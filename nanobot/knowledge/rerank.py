"""Cross-encoder style reranking with RRF fallback."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Protocol

from pydantic import Field

from nanobot.config_base import Base
from nanobot.knowledge.fusion import FusedSearchResult

_TOKEN_RE = re.compile(r"[A-Za-z0-9_+.-]+|[\u4e00-\u9fff]{2,}")


class RerankTrace(Base):
    """Privacy-safe rerank telemetry."""

    rerank_used: bool = False
    rerank_fallback: bool = False
    rerank_latency_ms: float = 0.0
    rerank_candidate_count: int = 0
    scorer: str = ""


class CrossEncoderScorer(Protocol):
    """Score (query, passage) pairs; higher is better."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class LexicalOverlapScorer:
    """Deterministic stand-in for CI/eval when sentence-transformers is absent."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        query_tokens = set(_TOKEN_RE.findall(query.casefold()))
        if not query_tokens:
            return [0.0 for _ in passages]
        scores: list[float] = []
        for passage in passages:
            passage_tokens = set(_TOKEN_RE.findall(passage.casefold()))
            if not passage_tokens:
                scores.append(0.0)
                continue
            overlap = len(query_tokens & passage_tokens)
            scores.append(overlap / max(len(query_tokens), 1) + 0.01 * overlap)
        return scores


class SentenceTransformerCrossEncoder:
    """Optional local CrossEncoder; import errors become unavailable."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for neural rerank"
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        model = self._load()
        pairs = [(query, passage) for passage in passages]
        predict = getattr(model, "predict")
        raw = predict(pairs)
        return [float(value) for value in raw]


def build_default_scorer(model_name: str, *, prefer_neural: bool = True) -> CrossEncoderScorer:
    """Prefer local CrossEncoder; fall back to lexical overlap scorer."""

    if prefer_neural:
        try:
            import importlib.util

            if importlib.util.find_spec("sentence_transformers") is not None:
                return SentenceTransformerCrossEncoder(model_name)
        except Exception:
            pass
    return LexicalOverlapScorer()


def scorer_label(scorer: CrossEncoderScorer) -> str:
    if isinstance(scorer, SentenceTransformerCrossEncoder):
        return f"cross-encoder:{scorer.model_name}"
    if isinstance(scorer, LexicalOverlapScorer):
        return "lexical-overlap"
    return type(scorer).__name__


def rerank_fused_results(
    query: str,
    candidates: Sequence[FusedSearchResult],
    *,
    top_k: int,
    scorer: CrossEncoderScorer | None = None,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    prefer_neural: bool = True,
) -> tuple[list[FusedSearchResult], RerankTrace]:
    """Rerank RRF candidates; on any failure return the original RRF order."""

    started = time.perf_counter()
    candidate_count = len(candidates)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not candidates:
        return (
            [],
            RerankTrace(
                rerank_used=False,
                rerank_fallback=False,
                rerank_latency_ms=0.0,
                rerank_candidate_count=0,
                scorer="",
            ),
        )
    active = scorer or build_default_scorer(model_name, prefer_neural=prefer_neural)
    label = scorer_label(active)
    try:
        passages = [item.result.chunk.text for item in candidates]
        scores = active.score(query, passages)
        if len(scores) != len(candidates):
            raise RuntimeError("reranker returned the wrong score count")
        ordered = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: (-pair[1], -pair[0].fused_score, pair[0].result.chunk.chunk_id),
        )
        reranked = [item for item, _score in ordered[:top_k]]
        return (
            reranked,
            RerankTrace(
                rerank_used=True,
                rerank_fallback=False,
                rerank_latency_ms=round((time.perf_counter() - started) * 1_000, 3),
                rerank_candidate_count=candidate_count,
                scorer=label,
            ),
        )
    except Exception:
        return (
            list(candidates[:top_k]),
            RerankTrace(
                rerank_used=False,
                rerank_fallback=True,
                rerank_latency_ms=round((time.perf_counter() - started) * 1_000, 3),
                rerank_candidate_count=candidate_count,
                scorer=label,
            ),
        )


class RerankSettings(Base):
    """Serializable rerank knobs for config and ablation."""

    enabled: bool = False
    candidate_k: int = Field(default=15, ge=1, le=50)
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    prefer_neural: bool = True
