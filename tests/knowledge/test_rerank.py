"""Unit tests for rerank fallback and citation stability."""

from __future__ import annotations

from pathlib import Path

from nanobot.knowledge.fusion import FusedSearchResult
from nanobot.knowledge.models import KnowledgeChunk, KnowledgeSearchResult
from nanobot.knowledge.rerank import LexicalOverlapScorer, rerank_fused_results


def _fused(chunk_id: str, text: str, score: float) -> FusedSearchResult:
    chunk = KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="doc",
        text=text,
        start_char=0,
        end_char=len(text),
        heading=None,
    )
    result = KnowledgeSearchResult(
        chunk=chunk,
        source_path=Path(f"{chunk_id}.md"),
        source_name=f"{chunk_id}.md",
        score=score,
        rank=1,
    )
    return FusedSearchResult(result=result, fused_score=score, contributing_lists=1)


def test_successful_rerank_prefers_overlapping_passage() -> None:
    candidates = [
        _fused("a", "theme preview retry button", 0.9),
        _fused("b", "idempotency keys prevent duplicate task creation", 0.8),
        _fused("c", "css variables light surface", 0.7),
    ]
    reranked, trace = rerank_fused_results(
        "how to avoid duplicate task creation on retry",
        candidates,
        top_k=2,
        scorer=LexicalOverlapScorer(),
    )
    assert trace.rerank_used is True
    assert [item.result.chunk.chunk_id for item in reranked][0] == "b"
    assert {item.result.chunk.chunk_id for item in reranked} <= {"a", "b", "c"}


def test_reranker_exception_falls_back_to_rrf_order() -> None:
    class Boom:
        def score(self, query: str, passages: list[str]) -> list[float]:
            raise RuntimeError("unavailable")

    candidates = [_fused("a", "one", 0.9), _fused("b", "two", 0.8)]
    reranked, trace = rerank_fused_results(
        "query", candidates, top_k=2, scorer=Boom()  # type: ignore[arg-type]
    )
    assert trace.rerank_fallback is True
    assert [item.result.chunk.chunk_id for item in reranked] == ["a", "b"]


def test_candidate_boundary_and_empty() -> None:
    empty, trace = rerank_fused_results("q", [], top_k=3, scorer=LexicalOverlapScorer())
    assert empty == []
    assert trace.rerank_candidate_count == 0
    candidates = [_fused("a", "alpha", 1.0)]
    reranked, _trace = rerank_fused_results(
        "alpha", candidates, top_k=1, scorer=LexicalOverlapScorer()
    )
    assert reranked[0].result.chunk.chunk_id == "a"
