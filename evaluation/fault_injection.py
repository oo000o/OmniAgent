"""Fault-injection acceptance for recovery, version binding, and privacy."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from nanobot.agent.hook import AgentHookContext, AgentTurnHookContext
from nanobot.agent.tools.career import (
    CareerToolsConfig,
    CareerWorkflowConfirmTool,
    CareerWorkflowGetTool,
    CareerWorkflowReplanTool,
    CareerWorkflowRetrieveTool,
    CareerWorkflowScheduleTool,
    CareerWorkflowStartTool,
    CareerWorkflowVerifyPlanTool,
)
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.knowledge import KnowledgeToolsConfig
from nanobot.bus.runtime_events import RuntimeEventContext, SessionTurnStarted
from nanobot.career import (
    CareerCheckpoint,
    CareerWorkflow,
    CareerWorkflowCreate,
    CareerWorkflowState,
    CareerWorkflowStore,
    CareerWorkflowTransition,
    EvidenceReference,
    GapItem,
    GapStatus,
    LearningPlanItem,
)
from nanobot.cron.service import CronService
from nanobot.knowledge.embeddings import EmbeddingProviderError
from nanobot.observability import RunStore, create_run_observability_hook
from nanobot.providers.base import ToolCallRequest
from nanobot.session.keys import UNIFIED_SESSION_KEY
from nanobot.tasking import TaskCreate, TaskStore


@dataclass(frozen=True)
class FaultCaseResult:
    case_id: str
    group: str
    passed: bool
    detail: str


class _FailingEmbeddingProvider:
    @property
    def model_name(self) -> str:
        return "failing-v1"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbeddingProviderError("backend unavailable")


def _base_checkpoint() -> CareerCheckpoint:
    return CareerCheckpoint(
        evidence=[
            EvidenceReference(
                evidence_id="K1", source_type="jd", source_name="jd.md", chunk_id="c1"
            )
        ],
        gaps=[
            GapItem(
                competency="RAG evaluation",
                status=GapStatus.MISSING,
                rationale="Required by the JD.",
                evidence_ids=["K1"],
            )
        ],
        plan=[LearningPlanItem(item_id="generic", title="Read documentation")],
    )


def _advance(
    store: CareerWorkflowStore,
    current: CareerWorkflow,
    *,
    prefix: str,
) -> CareerWorkflow:
    checkpoint = _base_checkpoint()
    for state, key in (
        (CareerWorkflowState.EVIDENCE_RETRIEVED, f"{prefix}-evidence"),
        (CareerWorkflowState.GAP_READY, f"{prefix}-gap"),
        (CareerWorkflowState.PLAN_VERIFYING, f"{prefix}-plan"),
    ):
        current = store.transition(
            current.workflow_id,
            CareerWorkflowTransition(target_state=state, checkpoint=checkpoint),
            expected_version=current.version,
            idempotency_key=key,
        )
    return current


def _sqlite_bytes(path: Path) -> bytes:
    chunks = [path.read_bytes()] if path.is_file() else [b""]
    for suffix in ("-wal", "-shm"):
        extra = Path(str(path) + suffix)
        if extra.is_file():
            chunks.append(extra.read_bytes())
    return b"".join(chunks)


async def evaluate_fault_injection(root: Path) -> dict[str, object]:
    """Deterministic crash/replay and privacy checks; not production availability."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "resume.md").write_text(
        "Built Python APIs with Docker and recoverable Agent workflows.", encoding="utf-8"
    )
    (root / "jd.md").write_text(
        "The role requires Python, Docker, and RAG evaluation.", encoding="utf-8"
    )
    config = CareerToolsConfig(
        database_path="state/career.db", task_database_path="state/tasks.db"
    )
    cases: list[FaultCaseResult] = []

    task_store = TaskStore(root / config.task_database_path)
    task_store.initialize()
    request = TaskCreate(title="Build a RAG evaluation set", source="career:fault:rag-eval")
    first = task_store.create(request, idempotency_key="career:fault:rag-eval")
    recovered_ids = [
        task_store.create(request, idempotency_key="career:fault:rag-eval").task_id
        for _ in range(3)
    ]
    cases.append(
        FaultCaseResult(
            "fault-task-replay-thrice",
            "fault_injection",
            recovered_ids == [first.task_id] * 3 and len(task_store.list()) == 1,
            first.task_id,
        )
    )

    store = CareerWorkflowStore(root / config.database_path)
    store.initialize()
    current = store.create(
        CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md"),
        idempotency_key="fault-cron-workflow",
    )
    current = _advance(store, current, prefix="fault-cron")
    verifier = CareerWorkflowVerifyPlanTool(workspace=root, config=config)
    rejected = json.loads(
        await verifier.execute(current.workflow_id, current.version, "fault-cron-reject")
    )
    revised_plan = [
        {
            "item_id": "rag-eval",
            "title": "Build a RAG evaluation set",
            "priority": 1,
            "addresses": ["RAG evaluation"],
        }
    ]
    replan = CareerWorkflowReplanTool(workspace=root, config=config)
    revised = json.loads(
        await replan.execute(
            current.workflow_id,
            json.dumps(revised_plan),
            "uncovered-required-competency",
            "Bind the missing competency.",
            rejected["version"],
            "fault-cron-replan",
        )
    )
    awaiting = json.loads(
        await verifier.execute(current.workflow_id, revised["version"], "fault-cron-verify")
    )
    confirm = CareerWorkflowConfirmTool(workspace=root, config=config)
    confirm.set_context(
        RequestContext(
            channel="feishu",
            chat_id="oc_fault",
            original_user_text="确认创建学习任务",
        )
    )
    creating = json.loads(
        await confirm.execute(current.workflow_id, awaiting["version"], "fault-cron-confirm")
    )
    task = task_store.create(
        TaskCreate(
            title="Build a RAG evaluation set",
            source=f"career:{current.workflow_id}:rag-eval",
        ),
        idempotency_key=f"career:{current.workflow_id}:rag-eval",
    )
    recorded_state = CareerWorkflowState.TASKS_CREATED
    creating_workflow = store.get(current.workflow_id)
    with_tasks = creating_workflow.checkpoint.model_copy(
        update={"task_ids": {"rag-eval": task.task_id}}
    )
    created = store.transition(
        current.workflow_id,
        CareerWorkflowTransition(target_state=recorded_state, checkpoint=with_tasks),
        expected_version=creating["version"],
        idempotency_key="fault-cron-record",
    )

    cron_path = root / "state" / "cron.json"
    cron = CronService(cron_path)
    schedule = CareerWorkflowScheduleTool(workspace=root, config=config, cron_service=cron)
    origin = RequestContext(
        channel="feishu", chat_id="oc_fault", session_key=UNIFIED_SESSION_KEY
    )
    original_transition = schedule._store.transition

    def fail_checkpoint(*args: object, **kwargs: object) -> CareerWorkflow:
        raise OSError("simulated checkpoint outage")

    schedule._store.transition = fail_checkpoint  # type: ignore[method-assign]
    with request_context(origin):
        interrupted = await schedule.execute(
            current.workflow_id, 3600, created.version, "fault-cron-schedule"
        )
    schedule._store.transition = original_transition  # type: ignore[method-assign]
    recovery = CareerWorkflowScheduleTool(
        workspace=root, config=config, cron_service=CronService(cron_path)
    )
    job_ids: list[str] = []
    with request_context(origin):
        for index in range(3):
            recovered = json.loads(
                await recovery.execute(
                    current.workflow_id, 3600, created.version, "fault-cron-schedule"
                )
            )
            job_ids.append(str(recovered["checkpoint"]["followup_job_id"]))
    cases.append(
        FaultCaseResult(
            "fault-cron-replay-thrice",
            "fault_injection",
            "CHECKPOINT_ERROR" in interrupted
            and len(set(job_ids)) == 1
            and len(CronService(cron_path).list_jobs()) == 1,
            job_ids[0] if job_ids else "missing",
        )
    )

    replan_store = CareerWorkflowStore(root / "state" / "replan.db")
    replan_store.initialize()
    replan_current = replan_store.create(
        CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md"),
        idempotency_key="fault-replan-workflow",
    )
    replan_current = _advance(replan_store, replan_current, prefix="fault-replan")
    replan_config = CareerToolsConfig(database_path="state/replan.db")
    replan_verifier = CareerWorkflowVerifyPlanTool(workspace=root, config=replan_config)
    replan_rejected = json.loads(
        await replan_verifier.execute(
            replan_current.workflow_id, replan_current.version, "fault-replan-reject"
        )
    )
    replan_tool = CareerWorkflowReplanTool(workspace=root, config=replan_config)
    original_replan = replan_tool._store.transition

    def fail_replan(*args: object, **kwargs: object) -> CareerWorkflow:
        raise OSError("simulated replan checkpoint outage")

    replan_tool._store.transition = fail_replan  # type: ignore[method-assign]
    interrupted_replan = await replan_tool.execute(
        replan_current.workflow_id,
        json.dumps(revised_plan),
        "uncovered-required-competency",
        "Bind the missing competency.",
        replan_rejected["version"],
        "fault-replan-1",
    )
    replan_tool._store.transition = original_replan  # type: ignore[method-assign]
    after_crash = json.loads(
        await CareerWorkflowGetTool(workspace=root, config=replan_config).execute(
            replan_current.workflow_id
        )
    )
    recovered_replan = json.loads(
        await replan_tool.execute(
            replan_current.workflow_id,
            json.dumps(revised_plan),
            "uncovered-required-competency",
            "Bind the missing competency.",
            replan_rejected["version"],
            "fault-replan-1",
        )
    )
    cases.append(
        FaultCaseResult(
            "fault-replan-interrupt",
            "fault_injection",
            "CHECKPOINT_ERROR" in interrupted_replan
            and after_crash["checkpoint"]["plan_revision"] == 1
            and recovered_replan["checkpoint"]["plan_revision"] == 2
            and recovered_replan["checkpoint"]["replan_count"] == 1,
            f"before={after_crash['checkpoint']['plan_revision']} after={recovered_replan['checkpoint']['plan_revision']}",
        )
    )

    stale_store = CareerWorkflowStore(root / "state" / "stale.db")
    stale_store.initialize()
    stale_current = stale_store.create(
        CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md"),
        idempotency_key="fault-stale-workflow",
    )
    stale_current = _advance(stale_store, stale_current, prefix="fault-stale")
    stale_config = CareerToolsConfig(database_path="state/stale.db")
    stale_verifier = CareerWorkflowVerifyPlanTool(workspace=root, config=stale_config)
    stale_rejected = json.loads(
        await stale_verifier.execute(
            stale_current.workflow_id, stale_current.version, "fault-stale-reject"
        )
    )
    seen_v1_version = int(stale_rejected["version"])
    stale_replan = CareerWorkflowReplanTool(workspace=root, config=stale_config)
    stale_revised = json.loads(
        await stale_replan.execute(
            stale_current.workflow_id,
            json.dumps(revised_plan),
            "uncovered-required-competency",
            "Bind the missing competency.",
            stale_rejected["version"],
            "fault-stale-replan",
        )
    )
    stale_awaiting = json.loads(
        await stale_verifier.execute(
            stale_current.workflow_id, stale_revised["version"], "fault-stale-verify"
        )
    )
    stale_confirm = CareerWorkflowConfirmTool(workspace=root, config=stale_config)
    stale_confirm.set_context(
        RequestContext(channel="feishu", chat_id="oc_fault", original_user_text="确认创建学习任务")
    )
    stale_result = await stale_confirm.execute(
        stale_current.workflow_id, seen_v1_version, "fault-stale-confirm"
    )
    current_after_stale = json.loads(
        await CareerWorkflowGetTool(workspace=root, config=stale_config).execute(
            stale_current.workflow_id
        )
    )
    cases.append(
        FaultCaseResult(
            "fault-stale-confirm",
            "fault_injection",
            "VERSION_CONFLICT" in stale_result
            and stale_awaiting["state"] == "awaiting_confirmation"
            and current_after_stale["state"] == "awaiting_confirmation"
            and current_after_stale["checkpoint"]["plan_revision"] == 2
            and current_after_stale["checkpoint"]["confirmed"] is False,
            stale_result,
        )
    )

    exhausted_store = CareerWorkflowStore(root / "state" / "exhausted.db")
    exhausted_store.initialize()
    exhausted_current = exhausted_store.create(
        CareerWorkflowCreate(resume_source="resume.md", jd_source="jd.md"),
        idempotency_key="fault-exhausted-workflow",
    )
    exhausted_current = _advance(exhausted_store, exhausted_current, prefix="fault-exhausted")
    exhausted_config = CareerToolsConfig(database_path="state/exhausted.db")
    exhausted_verifier = CareerWorkflowVerifyPlanTool(workspace=root, config=exhausted_config)
    exhausted_replan = CareerWorkflowReplanTool(workspace=root, config=exhausted_config)
    rejected_plan = json.loads(
        await exhausted_verifier.execute(
            exhausted_current.workflow_id, exhausted_current.version, "fault-ex-reject"
        )
    )
    for attempt in (1, 2):
        rejected_plan = json.loads(
            await exhausted_replan.execute(
                exhausted_current.workflow_id,
                json.dumps(
                    [{"item_id": "attempt", "title": f"Attempt {attempt}", "priority": 1}]
                ),
                "missing-gap-binding",
                f"Try revision {attempt}.",
                rejected_plan["version"],
                f"fault-ex-replan-{attempt}",
            )
        )
        rejected_plan = json.loads(
            await exhausted_verifier.execute(
                exhausted_current.workflow_id,
                rejected_plan["version"],
                f"fault-ex-verify-{attempt}",
            )
        )
    cases.append(
        FaultCaseResult(
            "fault-replan-exhausted",
            "fault_injection",
            rejected_plan["state"] == "failed"
            and str(rejected_plan["checkpoint"]["error"]).startswith("REPLAN_EXHAUSTED:"),
            str(rejected_plan["checkpoint"]["error"]),
        )
    )

    retrieve = CareerWorkflowRetrieveTool(
        workspace=root,
        config=config,
        knowledge_config=KnowledgeToolsConfig(
            database_path="state/knowledge.db",
            retrieval_mode="hybrid",
            embedding_model="failing-v1",
            candidate_results=10,
        ),
        embedding_provider=_FailingEmbeddingProvider(),
    )
    started = json.loads(
        await CareerWorkflowStartTool(workspace=root, config=config).execute(
            "resume.md", "jd.md", "fault-fallback-start"
        )
    )
    fallback = json.loads(
        await retrieve.execute(
            started["workflow_id"],
            json.dumps(["Python Docker", "RAG evaluation"]),
            int(started["version"]),
            "fault-fallback-retrieve",
        )
    )
    cases.append(
        FaultCaseResult(
            "fault-embedding-fallback",
            "fault_injection",
            fallback.get("retrieval_fallback") is True
            and fallback["workflow"]["state"] == "evidence_retrieved",
            "lexical fallback after embedding failure",
        )
    )

    secret = "SECRET_RESUME_BODY"
    token = "sk-secretvalue999"
    captured: list[str] = []
    handler = logger.add(lambda message: captured.append(str(message)), level="DEBUG")
    privacy_started = json.loads(
        await CareerWorkflowStartTool(workspace=root, config=config).execute(
            "resume.md", "jd.md", "fault-privacy-start"
        )
    )
    await retrieve.execute(
        privacy_started["workflow_id"],
        json.dumps(["Python", secret, token]),
        int(privacy_started["version"]),
        "fault-privacy-retrieve",
    )
    privacy_store = RunStore(root / "privacy-runs.db")
    privacy_store.initialize()
    session_key = "fault:privacy"
    privacy_store.handle(
        SessionTurnStarted(
            RuntimeEventContext(channel="webui", chat_id="privacy", session_key=session_key)
        )
    )
    hook = create_run_observability_hook(privacy_store)(
        AgentTurnHookContext(session_key=session_key)
    )
    assert hook is not None
    call = ToolCallRequest(id="privacy-retrieve", name="career_workflow_retrieve", arguments={})
    params = {
        "workflow_id": str(privacy_started["workflow_id"]),
        "queries_json": json.dumps([secret, token]),
        "api_key": token,
    }
    await hook.before_execute_tool(AgentHookContext(iteration=0, messages=[]), call, None, params)
    await hook.after_execute_tool(
        AgentHookContext(iteration=0, messages=[]),
        call,
        None,
        params,
        json.dumps(
            {
                "workflow": fallback["workflow"],
                "evidence": [{"text": secret}],
                "retrieval_fallback": True,
            }
        ),
    )
    logger.remove(handler)
    blob = "\n".join(captured).encode() + _sqlite_bytes(root / "privacy-runs.db")
    cases.append(
        FaultCaseResult(
            "fault-privacy-trace-logs",
            "fault_injection",
            secret.encode() not in blob and token.encode() not in blob,
            "trace and logs omitted document text and credentials",
        )
    )

    passed = sum(case.passed for case in cases)
    return {
        "fixture": "deterministic-fault-injection-v1",
        "scope": "offline crash/replay and privacy checks; not production availability",
        "total": len(cases),
        "passed": passed,
        "pass_rate": round(passed / len(cases), 4) if cases else 0.0,
        "groups": {
            "fault_injection": {
                "passed": passed,
                "total": len(cases),
                "pass_rate": round(passed / len(cases), 4) if cases else 0.0,
            }
        },
        "cases": [case.__dict__ for case in cases],
    }
