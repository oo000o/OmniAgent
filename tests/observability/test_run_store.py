from types import SimpleNamespace

from nanobot.bus.runtime_events import (
    RuntimeEventBus,
    RuntimeEventContext,
    SessionTurnPersisted,
    SessionTurnStarted,
    TurnCompleted,
    TurnRetryObserved,
    TurnRunStatusChanged,
    TurnRuntimeAdmitted,
)
from nanobot.observability import RunStore
from nanobot.providers.base import LLMUsage


async def test_runtime_events_persist_complete_run(tmp_path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.initialize()
    bus = RuntimeEventBus()
    store.subscribe(bus)
    context = RuntimeEventContext(
        channel="telegram", chat_id="tg-42", session_key="unified:default"
    )
    runtime = SimpleNamespace(provider=object(), model="qwen-plus")
    usage = LLMUsage(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        reported_tokens=120,
    )

    await bus.publish(SessionTurnStarted(context))
    await bus.publish(TurnRuntimeAdmitted(context, runtime))
    await bus.publish(TurnRunStatusChanged(context, "executing_tools"))
    store.increment_tool_calls("unified:default")
    store.increment_tool_calls("unified:default")
    await bus.publish(TurnRetryObserved(context, "retrying in one second"))
    await bus.publish(SessionTurnPersisted(context, turn_id="turn-7", sender_id="user-1"))
    await bus.publish(TurnCompleted(context, latency_ms=321, runtime=runtime, usage=usage))

    records = store.list(session_key="unified:default")
    assert len(records) == 1
    record = records[0]
    assert record.channel == "telegram"
    assert record.status == "completed"
    assert record.turn_id == "turn-7"
    assert record.model == "qwen-plus"
    assert record.latency_ms == 321
    assert record.total_tokens == 120
    assert record.tool_calls == 2
    assert record.retries == 1
    assert store.get(record.run_id) == record


async def test_unknown_events_and_limits_are_safe(tmp_path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.initialize()
    bus = RuntimeEventBus()
    store.subscribe(bus)

    await bus.publish(
        TurnRunStatusChanged(
            RuntimeEventContext(channel="telegram", chat_id="x", session_key="missing"),
            "running",
        )
    )

    assert store.list() == []


async def test_error_survives_completion_event(tmp_path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.initialize()
    context = RuntimeEventContext(channel="telegram", chat_id="x", session_key="session-x")
    store.handle(SessionTurnStarted(context))
    store.record_error("session-x", "provider unavailable")
    store.handle(TurnCompleted(context, latency_ms=50))

    record = store.list()[0]
    assert record.status == "error"
    assert record.error == "provider unavailable"


async def test_tool_trace_and_aggregate_metrics(tmp_path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.initialize()
    context = RuntimeEventContext(channel="webui", chat_id="c1", session_key="obs:metrics")
    store.handle(SessionTurnStarted(context))
    started = store.begin_tool_call(
        "obs:metrics",
        tool_name="career_workflow_replan",
        workflow_id="11111111-2222-3333-4444-555555555555",
        plan_revision=1,
    )
    assert started is not None
    store.finish_tool_call(started, status="succeeded", flags=("replan",), plan_revision=2)
    store.record_span(
        "obs:metrics",
        event_type="replan_accepted",
        status="succeeded",
        tool_name="career_workflow_replan",
        flags=("replan",),
    )
    store.handle(
        TurnCompleted(
            context,
            latency_ms=40,
            usage=LLMUsage(
                input_tokens=10, output_tokens=10, total_tokens=20, reported_tokens=20
            ),
        )
    )

    events = store.list_events(store.list()[0].run_id)
    types = [event.event_type for event in events]
    assert "tool_succeeded" in types
    assert "replan_accepted" in types
    summary = store.aggregate(session_key="obs:metrics")
    assert summary["sample_count"] == 1
    assert summary["metrics"]["run_success_rate"] == 1.0
    assert summary["metrics"]["tool_success_rate"] == 1.0
    assert summary["metrics"]["replan_success_rate"] == 1.0
    assert "p95_latency_ms" in summary["metrics"]
    dumped = (tmp_path / "runs.db").read_bytes()
    assert b"resume" not in dumped or b"11111111-2222-3333-4444-555555555555" in dumped

