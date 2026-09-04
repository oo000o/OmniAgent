"""Tests for credential-safe third-party logging."""

from nanobot.utils.logging_bridge import _redact_log_message


def test_redact_log_message_hides_url_query_credentials() -> None:
    message = (
        "connected to wss://example.test/ws?v=2&access_key=private-key"
        "&service_id=42&ticket=private-ticket [conn_id=7]"
    )

    redacted = _redact_log_message(message)

    assert "private-key" not in redacted
    assert "private-ticket" not in redacted
    assert "access_key=<redacted>" in redacted
    assert "ticket=<redacted>" in redacted
    assert "service_id=42" in redacted
    assert "conn_id=7" in redacted


def test_redact_log_message_preserves_noncredential_text() -> None:
    message = "connected to wss://example.test/ws?v=2&service_id=42"

    assert _redact_log_message(message) == message
