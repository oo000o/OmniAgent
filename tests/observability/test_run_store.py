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
