"""Privacy-safe trace fields extracted from tool calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, cast

from nanobot.observability.errors import AgentErrorCode, extract_error_code

_WORKFLOW_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
_FALLBACK_MARKER = "[Retrieval mode: lexical fallback]"
_MAX_JSON_CHARS = 80_000


@dataclass(frozen=True, slots=True)
class TraceContext:
    workflow_id: str | None = None
    plan_revision: int | None = None
    step: str | None = None
    flags: tuple[str, ...] = ()
    error_code: AgentErrorCode | None = None
    extra_events: tuple[str, ...] = ()


def extract_trace_context(
    tool_name: str,
    params: Any,
    result: Any | None = None,
    *,
    failed: bool = False,
    error: Any | None = None,
) -> TraceContext:
    """Copy only identifiers and flags; never keep queries, documents, or arguments."""

    workflow_id = _workflow_id_from_params(params)
    plan_revision: int | None = None
    flags: list[str] = []
    extra: list[str] = []
    error_code = extract_error_code(str(error)) if error is not None else None

    parsed = _json_object(result)
    if parsed is not None:
        nested = _string_keyed(parsed.get("workflow"))
        body = {**parsed, **nested} if nested is not None else parsed
        workflow_id = _safe_workflow_id(body.get("workflow_id")) or workflow_id
        checkpoint = _string_keyed(body.get("checkpoint"))
        if checkpoint is not None:
            plan_revision = _safe_revision(checkpoint.get("plan_revision"))
            checkpoint_error = checkpoint.get("error")
            if isinstance(checkpoint_error, str):
                error_code = extract_error_code(checkpoint_error) or error_code
            if body.get("state") == "failed" and error_code == AgentErrorCode.REPLAN_EXHAUSTED:
                extra.append("replan_exhausted")
        if parsed.get("retrieval_fallback") is True:
            flags.append("fallback")
            extra.append("retrieval_fallback")
        completed = parsed.get("completed_task_ids")
        pending = parsed.get("pending_calls")
        if isinstance(completed, dict) and completed:
            flags.append("recovery")
            extra.append("recovery_started")
            if not pending:
                extra.append("recovery_succeeded")

    if isinstance(result, str) and _FALLBACK_MARKER in result:
        flags.append("fallback")
        extra.append("retrieval_fallback")

    if tool_name == "career_workflow_replan":
        flags.append("replan")
        if failed and error_code == AgentErrorCode.REPLAN_EXHAUSTED:
            extra.append("replan_exhausted")
        elif not failed:
            extra.append("replan_accepted")

    extra_events = tuple(dict.fromkeys(item for item in extra if item))
    return TraceContext(
        workflow_id=workflow_id,
        plan_revision=plan_revision,
        step=tool_name or None,
        flags=tuple(dict.fromkeys(flags)),
        error_code=error_code,
        extra_events=extra_events,
    )


def starting_events(tool_name: str) -> tuple[str, ...]:
    if tool_name == "career_workflow_replan":
        return ("replan_started",)
    return ()


def _workflow_id_from_params(params: Any) -> str | None:
    keyed = _string_keyed(params)
    if keyed is None:
        return None
    return _safe_workflow_id(keyed.get("workflow_id"))


def _safe_workflow_id(value: object) -> str | None:
    if isinstance(value, str) and _WORKFLOW_ID_RE.fullmatch(value):
        return value
    return None


def _safe_revision(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if 1 <= value <= 100:
        return value
    return None


def _string_keyed(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    typed: dict[str, Any] = {}
    for key, item in cast(dict[object, object], value).items():
        if isinstance(key, str):
            typed[key] = item
    return typed


def _json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return _string_keyed(cast(dict[object, object], value))
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text.startswith("{") or len(text) > _MAX_JSON_CHARS:
        return None
    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _string_keyed(parsed)
