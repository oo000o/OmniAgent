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
    query_type: str | None = None
    rewrite_used: bool = False
    rewrite_query_count: int = 0
    rewrite_fallback: bool = False
    rewrite_latency_ms: float = 0.0
    rerank_used: bool = False
    rerank_fallback: bool = False
    rerank_latency_ms: float = 0.0
    rerank_candidate_count: int = 0


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
        query_type=event.query_type,
        rewrite_used=event.rewrite_used,
        rewrite_query_count=event.rewrite_query_count,
        rewrite_fallback=event.rewrite_fallback,
        rewrite_latency_ms=event.rewrite_latency_ms,
        rerank_used=event.rerank_used,
        rerank_fallback=event.rerank_fallback,
        rerank_latency_ms=event.rerank_latency_ms,
        rerank_candidate_count=event.rerank_candidate_count,
    ).info(
        "Knowledge retrieval mode={} status={} latency_ms={} results={} "
        "rewrite_used={} rerank_used={}",
        event.mode,
        event.status,
        event.latency_ms,
        event.result_count,
        event.rewrite_used,
        event.rerank_used,
    )
