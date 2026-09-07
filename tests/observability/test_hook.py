from nanobot.agent.hook import AgentHookContext, AgentTurnHookContext
from nanobot.agent.tools.base import ToolResult
from nanobot.bus.runtime_events import RuntimeEventContext, SessionTurnStarted, TurnCompleted
from nanobot.observability import AgentErrorCode, RunStore, create_run_observability_hook
from nanobot.providers.base import ToolCallRequest


async def test_hook_records_tool_success_failure_and_omits_private_payloads(tmp_path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.initialize()
    session_key = "unified:trace"
    store.handle(
        SessionTurnStarted(
            RuntimeEventContext(channel="webui", chat_id="c1", session_key=session_key)
        )
    )
    hook = create_run_observability_hook(store)(AgentTurnHookContext(session_key=session_key))
    assert hook is not None
    context = AgentHookContext(iteration=0, messages=[])
    success_call = ToolCallRequest(id="t1", name="career_workflow_get", arguments={})
    fail_call = ToolCallRequest(id="t2", name="career_workflow_replan", arguments={})
    params = {
        "workflow_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "queries_json": "PRIVATE_JD_TEXT",
        "plan_json": "PRIVATE_PLAN_TEXT",
    }

    await hook.before_execute_tool(context, success_call, None, params)
    await hook.after_execute_tool(
        context,
        success_call,
        None,
        params,
        '{"workflow_id":"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee","checkpoint":{"plan_revision":1}}',
    )
    await hook.before_execute_tool(context, fail_call, None, params)
    await hook.on_execute_tool_error(
        context,
        fail_call,
        None,
        params,
        ToolResult.error("REPLAN_EXHAUSTED: uncovered required competencies"),
    )
    store.handle(TurnCompleted(RuntimeEventContext(channel="webui", chat_id="c1", session_key=session_key), latency_ms=12))

    events = store.list_events(store.list()[0].run_id)
    types = [event.event_type for event in events]
    assert "tool_succeeded" in types
    assert "tool_failed" in types
    assert "replan_started" in types
    assert "replan_exhausted" in types
    failed = next(event for event in events if event.event_type == "tool_failed")
    assert failed.error_code == AgentErrorCode.REPLAN_EXHAUSTED.value
    dumped = (tmp_path / "runs.db").read_bytes()
    wal = tmp_path / "runs.db-wal"
    blob = dumped + (wal.read_bytes() if wal.is_file() else b"")
    assert b"PRIVATE_JD_TEXT" not in blob
    assert b"PRIVATE_PLAN_TEXT" not in blob
