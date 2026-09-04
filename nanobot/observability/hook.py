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
from nanobot.observability.run_store import RunStore
from nanobot.providers.base import ToolCallRequest


class RunObservabilityHook(AgentHook):
    def __init__(self, store: RunStore, session_key: str) -> None:
        super().__init__()
        self._store = store
        self._session_key = session_key

    async def before_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
    ) -> None:
        self._store.increment_tool_calls(self._session_key)

    async def on_error(self, context: AgentRunHookContext) -> None:
        error = context.error or (
            str(context.exception) if context.exception is not None else "Agent run failed"
        )
        self._store.record_error(self._session_key, error)


def create_run_observability_hook(store: RunStore) -> AgentTurnHookFactory:
    def _factory(context: AgentTurnHookContext) -> AgentHook | None:
        if not context.session_key:
            return None
        return RunObservabilityHook(store, context.session_key)

    return _factory
