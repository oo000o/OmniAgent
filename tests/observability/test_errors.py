from nanobot.career import CareerWorkflowConflictError
from nanobot.knowledge.embeddings import EmbeddingProviderError
from nanobot.observability.errors import (
    AgentErrorCode,
    classify_exception,
    extract_error_code,
    format_error,
    sanitize_error_message,
)
from nanobot.tasking import TaskNotFoundError


def test_format_and_extract_error_codes() -> None:
    message = format_error(AgentErrorCode.VERSION_CONFLICT, "reload and retry")
    assert message.startswith("VERSION_CONFLICT:")
    assert extract_error_code(message) is AgentErrorCode.VERSION_CONFLICT
    assert extract_error_code("REPLAN_EXHAUSTED: uncovered gap") is AgentErrorCode.REPLAN_EXHAUSTED


def test_classify_career_and_mcp_exceptions() -> None:
    assert (
        classify_exception(
            CareerWorkflowConflictError("workflow changed; reload it and retry with the latest version")
        )
        is AgentErrorCode.VERSION_CONFLICT
    )
    assert (
        classify_exception(CareerWorkflowConflictError("invalid workflow transition: gap_ready -> completed"))
        is AgentErrorCode.VALIDATION_ERROR
    )
    assert classify_exception(CareerWorkflowConflictError("REPLAN_EXHAUSTED")) is AgentErrorCode.REPLAN_EXHAUSTED
    assert classify_exception(EmbeddingProviderError("timeout")) is AgentErrorCode.RETRIEVAL_ERROR
    assert classify_exception(TaskNotFoundError("missing")) is AgentErrorCode.MCP_ERROR
    assert classify_exception(ValueError("queries_json must contain 1-20 non-empty strings")) is (
        AgentErrorCode.VALIDATION_ERROR
    )
    from nanobot.security.workspace_policy import WorkspaceBoundaryError

    assert classify_exception(WorkspaceBoundaryError("outside")) is AgentErrorCode.VALIDATION_ERROR


def test_sanitize_error_message_redacts_secrets_and_truncates() -> None:
    text = sanitize_error_message(
        "token=abcd1234 user@example.com used sk-secretvalue999 and api_key=supersecret " + ("x" * 500)
    )
    assert "abcd1234" not in text
    assert "sk-secretvalue999" not in text
    assert "supersecret" not in text
    assert "@example.com" not in text
    assert len(text) <= 400
