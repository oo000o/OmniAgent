"""Per-turn hook that enriches durable run records with tool and error data."""

from __future__ import annotations

from typing import Any

from nanobot.agent.hook import (
    AgentHook,
    AgentHookContext,
    AgentRunHookContext,
    AgentTurnHookContext,
    AgentTurnHookFactory,
)
from nanobot.observability.errors import classify_exception, extract_error_code
from nanobot.observability.run_store import RunStore
from nanobot.observability.trace import extract_trace_context, starting_events
from nanobot.providers.base import ToolCallRequest


class RunObservabilityHook(AgentHook):
    def __init__(self, store: RunStore, session_key: str) -> None:
        super().__init__()
        self._store = store
        self._session_key = session_key
        self._open_tools: dict[str, str] = {}

    def _call_key(self, tool_call: ToolCallRequest) -> str:
        raw_id = tool_call.id.strip() if tool_call.id else ""
        if raw_id:
            return raw_id
        return f"{tool_call.name}:{id(tool_call)}"

    async def before_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
    ) -> None:
        self._store.increment_tool_calls(self._session_key)
        trace = extract_trace_context(tool_call.name, params)
        event_id = self._store.begin_tool_call(
            self._session_key,
            tool_name=tool_call.name,
            workflow_id=trace.workflow_id,
            plan_revision=trace.plan_revision,
            step=trace.step,
        )
        if event_id is not None:
            self._open_tools[self._call_key(tool_call)] = event_id
        for event_type in starting_events(tool_call.name):
            self._store.record_span(
                self._session_key,
                event_type=event_type,
                status="started",
                tool_name=tool_call.name,
                workflow_id=trace.workflow_id,
                plan_revision=trace.plan_revision,
                parent_event_id=event_id,
                flags=trace.flags,
            )

    async def after_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
        result: Any,
    ) -> None:
        self._finish_tool(tool_call, params, result=result, failed=False, error=None)

    async def on_execute_tool_error(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
        error: Any,
    ) -> None:
        self._finish_tool(tool_call, params, result=None, failed=True, error=error)

    def _finish_tool(
        self,
        tool_call: ToolCallRequest,
        params: Any,
        *,
        result: Any,
        failed: bool,
        error: Any,
    ) -> None:
        event_id = self._open_tools.pop(self._call_key(tool_call), None)
        if event_id is None:
            return
        trace = extract_trace_context(
            tool_call.name, params, result, failed=failed, error=error
        )
        error_code = trace.error_code
        if failed and error_code is None and isinstance(error, BaseException):
            error_code = classify_exception(error)
        if failed and error_code is None:
            error_code = extract_error_code(str(error))
        self._store.finish_tool_call(
            event_id,
            status="failed" if failed else "succeeded",
            error_code=error_code,
            flags=trace.flags,
            workflow_id=trace.workflow_id,
            plan_revision=trace.plan_revision,
        )
        extra_status = {
            "replan_accepted": "succeeded",
            "replan_exhausted": "failed",
            "recovery_started": "started",
            "recovery_succeeded": "succeeded",
            "retrieval_fallback": "fallback",
            "model_fallback": "fallback",
        }
        for event_type in trace.extra_events:
            self._store.record_span(
                self._session_key,
                event_type=event_type,
                status=extra_status.get(event_type, "recorded"),
                tool_name=tool_call.name,
                workflow_id=trace.workflow_id,
                plan_revision=trace.plan_revision,
                error_code=error_code if event_type.endswith("exhausted") else None,
                parent_event_id=event_id,
                flags=trace.flags,
            )

    async def on_error(self, context: AgentRunHookContext) -> None:
        error = context.error or (
            str(context.exception) if context.exception is not None else "Agent run failed"
        )
        code = extract_error_code(error)
        if code is None and context.exception is not None:
            code = classify_exception(context.exception)
        self._store.record_error(self._session_key, error, code)


def create_run_observability_hook(store: RunStore) -> AgentTurnHookFactory:
    def _factory(context: AgentTurnHookContext) -> AgentHook | None:
        if not context.session_key:
            return None
        return RunObservabilityHook(store, context.session_key)

    return _factory
