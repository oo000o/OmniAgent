"""Rank fusion for lexical and semantic retrieval results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from nanobot.knowledge.models import KnowledgeSearchResult


@dataclass(frozen=True, slots=True)
class FusedSearchResult:
    """A source result ranked across one or more retrieval strategies."""

    result: KnowledgeSearchResult
    fused_score: float
    contributing_lists: int


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[KnowledgeSearchResult]],
    *,
    limit: int = 5,
    rank_constant: int = 60,
) -> list[FusedSearchResult]:
    """Fuse ranked lists without requiring comparable raw score scales."""

    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")

    scores: dict[str, float] = defaultdict(float)
    appearances: dict[str, int] = defaultdict(int)
    representatives: dict[str, KnowledgeSearchResult] = {}
    for ranking in rankings:
        seen_in_ranking: set[str] = set()
        for position, result in enumerate(ranking, 1):
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(chunk_id)
            scores[chunk_id] += 1.0 / (rank_constant + position)
            appearances[chunk_id] += 1
            representatives.setdefault(chunk_id, result)

    ordered_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            -appearances[chunk_id],
            chunk_id,
        ),
    )
    return [
        FusedSearchResult(
            result=representatives[chunk_id],
            fused_score=scores[chunk_id],
            contributing_lists=appearances[chunk_id],
        )
        for chunk_id in ordered_ids[:limit]
    ]
