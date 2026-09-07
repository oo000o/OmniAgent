"""Deterministic observability cases for run traces and aggregate metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from nanobot.agent.hook import AgentHookContext, AgentTurnHookContext
from nanobot.bus.runtime_events import RuntimeEventContext, SessionTurnStarted, TurnCompleted
from nanobot.observability import AgentErrorCode, RunStore, create_run_observability_hook
from nanobot.observability.errors import classify_exception, format_error
from nanobot.providers.base import LLMUsage, ToolCallRequest


@dataclass(frozen=True)
class ObservabilityCaseResult:
    case_id: str
    group: str
    passed: bool
    detail: str


def _sqlite_bytes(path: Path) -> bytes:
    chunks = [path.read_bytes()] if path.is_file() else [b""]
    for suffix in ("-wal", "-shm"):
        extra = Path(str(path) + suffix)
        if extra.is_file():
            chunks.append(extra.read_bytes())
    return b"".join(chunks)


def _start_run(store: RunStore, session_key: str, chat_id: str) -> RuntimeEventContext:
    context = RuntimeEventContext(channel="webui", chat_id=chat_id, session_key=session_key)
    store.handle(SessionTurnStarted(context))
    return context


def _finish(store: RunStore, context: RuntimeEventContext, latency_ms: int, tokens: int) -> None:
    usage = LLMUsage(
        input_tokens=tokens // 2,
        output_tokens=tokens - tokens // 2,
        total_tokens=tokens,
        reported_tokens=tokens,
    )
    store.handle(
        TurnCompleted(
            context,
            latency_ms=latency_ms,
            runtime=SimpleNamespace(provider=object(), model="eval"),
            usage=usage,
        )
    )


async def evaluate_observability(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    store = RunStore(root / "runs.db")
    store.initialize()
    cases: list[ObservabilityCaseResult] = []

    success = _start_run(store, "obs:success", "c-ok")
    event_id = store.begin_tool_call(success.session_key, tool_name="career_workflow_get")
    assert event_id is not None
    store.finish_tool_call(event_id, status="succeeded")
    _finish(store, success, latency_ms=80, tokens=40)

    failed = _start_run(store, "obs:error", "c-err")
    store.record_error(
        failed.session_key,
        format_error(AgentErrorCode.VALIDATION_ERROR, "bad tool arguments"),
    )
    bad = store.begin_tool_call(failed.session_key, tool_name="career_workflow_transition")
    assert bad is not None
    store.finish_tool_call(
        bad, status="failed", error_code=AgentErrorCode.VALIDATION_ERROR
    )
    _finish(store, failed, latency_ms=200, tokens=60)

    recovered = _start_run(store, "obs:recovery", "c-rec")
    store.record_span(
        recovered.session_key,
        event_type="recovery_started",
        status="started",
        tool_name="career_workflow_task_manifest",
    )
    store.record_span(
        recovered.session_key,
        event_type="recovery_succeeded",
        status="succeeded",
        tool_name="career_workflow_task_manifest",
        flags=("recovery",),
    )
    _finish(store, recovered, latency_ms=120, tokens=50)

    replan_ok = _start_run(store, "obs:replan-ok", "c-replan")
    store.record_span(
        replan_ok.session_key, event_type="replan_started", status="started",
        tool_name="career_workflow_replan", flags=("replan",),
    )
    store.record_span(
        replan_ok.session_key, event_type="replan_accepted", status="succeeded",
        tool_name="career_workflow_replan", flags=("replan",),
    )
    _finish(store, replan_ok, latency_ms=90, tokens=70)

    replan_fail = _start_run(store, "obs:replan-fail", "c-replan-fail")
    store.record_span(
        replan_fail.session_key, event_type="replan_started", status="started",
        tool_name="career_workflow_replan", flags=("replan",),
    )
    store.record_span(
        replan_fail.session_key,
        event_type="replan_exhausted",
        status="failed",
        tool_name="career_workflow_replan",
        error_code=AgentErrorCode.REPLAN_EXHAUSTED,
        flags=("replan",),
    )
    store.record_error(
        replan_fail.session_key,
        format_error(AgentErrorCode.REPLAN_EXHAUSTED, "two revisions failed"),
    )
    _finish(store, replan_fail, latency_ms=150, tokens=55)

    fallback = _start_run(store, "obs:fallback", "c-fb")
    store.record_span(
        fallback.session_key,
        event_type="retrieval_fallback",
        status="fallback",
        tool_name="career_workflow_retrieve",
        flags=("fallback",),
    )
    _finish(store, fallback, latency_ms=110, tokens=45)

    summary = store.aggregate()
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    cases.append(
        ObservabilityCaseResult(
            "obs-run-success-rate",
            "observability",
            summary["sample_count"] == 6 and metrics["run_success_rate"] == 0.6667,
            str(metrics["run_success_rate"]),
        )
    )
    cases.append(
        ObservabilityCaseResult(
            "obs-p95-sample-count",
            "observability",
            summary["sample_count"] == 6 and metrics["p95_latency_ms"] == 200,
            f"n={summary['sample_count']} p95={metrics['p95_latency_ms']}",
        )
    )
    cases.append(
        ObservabilityCaseResult(
            "obs-tool-success-rate",
            "observability",
            metrics["tool_success_rate"] == 0.5,
            str(metrics["tool_success_rate"]),
        )
    )
    cases.append(
        ObservabilityCaseResult(
            "obs-replan-rates",
            "observability",
            metrics["replan_rate"] == 0.3333 and metrics["replan_success_rate"] == 0.5,
            f"rate={metrics['replan_rate']} success={metrics['replan_success_rate']}",
        )
    )
    cases.append(
        ObservabilityCaseResult(
            "obs-recovery-rate",
            "observability",
            metrics["recovery_success_rate"] == 1.0,
            str(metrics["recovery_success_rate"]),
        )
    )
    cases.append(
        ObservabilityCaseResult(
            "obs-fallback-rate",
            "observability",
            metrics["retrieval_fallback_rate"] == 0.1667
            and metrics["model_fallback_rate"] == 0.0,
            str(metrics["retrieval_fallback_rate"]),
        )
    )

    private = RunStore(root / "privacy.db")
    private.initialize()
    hook_factory = create_run_observability_hook(private)
    hook = hook_factory(AgentTurnHookContext(session_key="obs:privacy"))
    assert hook is not None
    secret_context = _start_run(private, "obs:privacy", "c-priv")
    call = ToolCallRequest(
        id="call-secret",
        name="career_workflow_retrieve",
        arguments={},
    )
    params = {
        "workflow_id": "11111111-2222-3333-4444-555555555555",
        "queries_json": "SECRET_RESUME_BODY",
        "api_key": "sk-secretvalue999",
    }
    result = (
        '{"workflow":{"workflow_id":"11111111-2222-3333-4444-555555555555",'
        '"checkpoint":{"plan_revision":2}},'
        '"evidence":"SECRET_RESUME_BODY","retrieval_fallback":true}'
    )
    await hook.before_execute_tool(AgentHookContext(iteration=0, messages=[]), call, None, params)
    await hook.after_execute_tool(
        AgentHookContext(iteration=0, messages=[]), call, None, params, result
    )
    _finish(private, secret_context, latency_ms=40, tokens=10)
    dumped = _sqlite_bytes(root / "privacy.db")
    leak = b"SECRET_RESUME_BODY" in dumped or b"sk-secretvalue999" in dumped
    cases.append(
        ObservabilityCaseResult(
            "obs-privacy-redaction",
            "observability",
            not leak,
            "trace omitted document text and secrets",
        )
    )

    conflict = classify_exception(
        RuntimeError("workflow changed; reload it and retry with the latest version")
    )
    # CareerWorkflowConflictError is classified by type name; RuntimeError falls through.
    from nanobot.career import CareerWorkflowConflictError

    version = classify_exception(
        CareerWorkflowConflictError("workflow changed; reload it and retry with the latest version")
    )
    exhausted = classify_exception(CareerWorkflowConflictError("REPLAN_EXHAUSTED"))
    cases.append(
        ObservabilityCaseResult(
            "obs-error-classification",
            "observability",
            version is AgentErrorCode.VERSION_CONFLICT
            and exhausted is AgentErrorCode.REPLAN_EXHAUSTED
            and conflict is AgentErrorCode.VALIDATION_ERROR,
            f"{version.value}/{exhausted.value}",
        )
    )

    passed = sum(case.passed for case in cases)
    return {
        "fixture": "deterministic-run-observability-v1",
        "scope": "offline trace and aggregate formulas; not production SLAs",
        "total": len(cases),
        "passed": passed,
        "pass_rate": round(passed / len(cases), 4) if cases else 0.0,
        "groups": {
            "observability": {
                "passed": passed,
                "total": len(cases),
                "pass_rate": round(passed / len(cases), 4) if cases else 0.0,
            }
        },
        "cases": [case.__dict__ for case in cases],
        "sample_aggregate": {
            "sample_count": summary["sample_count"],
            "metrics": metrics,
        },
    }
