"""Privacy-safe retrieval telemetry emitted at the knowledge boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger


@dataclass(frozen=True, slots=True)
class RetrievalEvent:
    """One retrieval outcome without query text or document contents."""

    mode: str
    status: str
    latency_ms: int
    result_count: int
    requested_limit: int
    candidate_limit: int


RetrievalObserver = Callable[[RetrievalEvent], None]


def log_retrieval_event(event: RetrievalEvent) -> None:
    """Write a structured event while keeping private query content out of logs."""

    logger.bind(
        event="knowledge_retrieval",
        retrieval_mode=event.mode,
        retrieval_status=event.status,
        latency_ms=event.latency_ms,
        result_count=event.result_count,
        requested_limit=event.requested_limit,
        candidate_limit=event.candidate_limit,
    ).info(
        "Knowledge retrieval mode={} status={} latency_ms={} results={}",
        event.mode,
        event.status,
        event.latency_ms,
        event.result_count,
    )
