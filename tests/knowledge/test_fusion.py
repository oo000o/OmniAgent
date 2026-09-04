from pathlib import Path

import pytest

from nanobot.knowledge.fusion import reciprocal_rank_fusion
from nanobot.knowledge.models import KnowledgeChunk, KnowledgeSearchResult


def _result(chunk_id: str, rank: int) -> KnowledgeSearchResult:
    return KnowledgeSearchResult(
        chunk=KnowledgeChunk(chunk_id, "doc", chunk_id, 0, len(chunk_id)),
        source_path=Path("source.md"),
        source_name="source.md",
        score=1.0 / rank,
        rank=rank,
    )


def test_result_in_both_lists_ranks_first() -> None:
    lexical = [_result("lexical", 1), _result("shared", 2)]
    semantic = [_result("semantic", 1), _result("shared", 2)]

    fused = reciprocal_rank_fusion([lexical, semantic])

    assert fused[0].result.chunk.chunk_id == "shared"
    assert fused[0].contributing_lists == 2


def test_duplicate_in_one_ranking_only_contributes_once() -> None:
    duplicate = _result("same", 1)

    fused = reciprocal_rank_fusion([[duplicate, duplicate]], rank_constant=10)

    assert fused[0].fused_score == pytest.approx(1 / 11)
    assert fused[0].contributing_lists == 1


@pytest.mark.parametrize(("limit", "rank_constant"), [(0, 60), (101, 60), (5, 0)])
def test_invalid_fusion_settings_are_rejected(limit: int, rank_constant: int) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], limit=limit, rank_constant=rank_constant)
