"""Unified Agent error codes and privacy-safe error text."""

from __future__ import annotations

import json
import re
import sqlite3
from enum import StrEnum


class AgentErrorCode(StrEnum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    REPLAN_EXHAUSTED = "REPLAN_EXHAUSTED"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    MCP_ERROR = "MCP_ERROR"
    CHECKPOINT_ERROR = "CHECKPOINT_ERROR"


ERROR_POLICY: dict[AgentErrorCode, str] = {
    AgentErrorCode.VALIDATION_ERROR: "reject",
    AgentErrorCode.VERSION_CONFLICT: "retry",
    AgentErrorCode.REPLAN_EXHAUSTED: "human",
    AgentErrorCode.RETRIEVAL_ERROR: "degrade",
    AgentErrorCode.MCP_ERROR: "retry",
    AgentErrorCode.CHECKPOINT_ERROR: "recover",
}

_ERROR_PREFIX = re.compile(
    r"^(VALIDATION_ERROR|VERSION_CONFLICT|REPLAN_EXHAUSTED|RETRIEVAL_ERROR|"
    r"MCP_ERROR|CHECKPOINT_ERROR):"
)
_QUERY_CREDENTIAL_RE = re.compile(
    r"([?&](?:access_key|access_token|api_key|authorization|client_secret|"
    r"password|secret|ticket|token)=)[^&\s\]]+",
    re.IGNORECASE,
)
_ASSIGNED_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)\b\s*[:=]\s*\S+"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+")
_EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I)
_SK_RE = re.compile(r"sk-[A-Za-z0-9]{8,}")
_VERSION_CONFLICT_MARKERS = (
    "workflow changed; reload",
    "reload it and retry",
    "latest version",
)


def format_error(code: AgentErrorCode, message: str) -> str:
    compact = " ".join(message.split())
    if compact.startswith(f"{code.value}:"):
        return compact
    return f"{code.value}: {compact}"


def extract_error_code(message: str) -> AgentErrorCode | None:
    match = _ERROR_PREFIX.match(message.strip())
    if match:
        return AgentErrorCode(match.group(1))
    if "REPLAN_EXHAUSTED" in message:
        return AgentErrorCode.REPLAN_EXHAUSTED
    return None


def sanitize_error_message(message: str, *, limit: int = 400) -> str:
    text = _QUERY_CREDENTIAL_RE.sub(r"\1<redacted>", message)
    text = _ASSIGNED_SECRET_RE.sub(r"\1=<redacted>", text)
    text = _BEARER_RE.sub("bearer <redacted>", text)
    text = _EMAIL_RE.sub("<redacted-email>", text)
    text = _SK_RE.sub("sk-<redacted>", text)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[:limit]
    return text


def classify_exception(exc: BaseException) -> AgentErrorCode:
    name = type(exc).__name__
    message = str(exc)
    prefixed = extract_error_code(message)
    if prefixed is not None:
        return prefixed
    if name in {"EmbeddingProviderError", "KnowledgeIngestionError"}:
        return AgentErrorCode.RETRIEVAL_ERROR
    if name in {"TaskNotFoundError", "TaskConflictError"}:
        return AgentErrorCode.MCP_ERROR
    if isinstance(exc, sqlite3.Error) or name in {"OperationalError", "DatabaseError"}:
        return AgentErrorCode.CHECKPOINT_ERROR
    if isinstance(exc, FileNotFoundError) or name == "WorkspaceBoundaryError":
        return AgentErrorCode.VALIDATION_ERROR
    if isinstance(exc, PermissionError):
        return AgentErrorCode.VALIDATION_ERROR
    if isinstance(exc, OSError):
        return AgentErrorCode.CHECKPOINT_ERROR
    if name == "CareerWorkflowConflictError":
        lowered = message.casefold()
        if any(marker in lowered for marker in _VERSION_CONFLICT_MARKERS):
            return AgentErrorCode.VERSION_CONFLICT
        return AgentErrorCode.VALIDATION_ERROR
    if isinstance(exc, (json.JSONDecodeError, ValueError)) or name in {
        "ValidationError",
        "CareerWorkflowNotFoundError",
    }:
        return AgentErrorCode.VALIDATION_ERROR
    return AgentErrorCode.VALIDATION_ERROR
