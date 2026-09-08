"""Retrieval ablation across lexical, dense, hybrid, rewrite, and rerank."""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from pathlib import Path

from evaluation.retrieval_corpus import CASES, DOCUMENTS, LabeledRetrievalCase
from nanobot.knowledge import HybridKnowledgeRetriever, KnowledgeStore
from nanobot.knowledge.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from nanobot.knowledge.query_rewrite import (
    OpenAICompatibleQueryRewriter,
    QueryRewriter,
    deterministic_rewrite,
)
from nanobot.knowledge.rerank import (
    CrossEncoderScorer,
    LexicalOverlapScorer,
    RerankSettings,
    SentenceTransformerCrossEncoder,
    scorer_label,
)
from nanobot.knowledge.retrieval import RetrievalMode

PIPELINES: tuple[tuple[str, RetrievalMode], ...] = (
    ("BM25", "bm25"),
    ("Vector", "vector"),
    ("Hybrid + RRF", "hybrid"),
    ("+ Rewrite", "hybrid_rewrite"),
    ("+ Rerank", "hybrid_rerank"),
    ("Final", "hybrid_rewrite_rerank"),
)

_CONCEPT_TERMS = (
    ("idempotency", "duplicate", "replay", "幂等", "重复", "两次任务", "副作用"),
    ("optimistic", "version", "concurrent", "locking", "乐观", "并发", "旧版本"),
    ("checkpoint", "recovery", "crash", "恢复", "崩溃", "对账"),
    ("rrf", "fusion", "hybrid", "融合", "合成"),
    ("bm25", "fts5", "lexical", "词法", "关键词"),
    ("vector", "embedding", "cosine", "向量", "语义"),
    ("citation", "offset", "provenance", "引用", "出处", "[k"),
    ("confirm", "human", "hitl", "plan_revision", "确认", "同意"),
    ("replan", "bounded", "exhausted", "重规划", "无限"),
    ("feishu", "webhook", "cron", "飞书", "长连接", "会话"),
    ("mcp", "pydantic", "stdio", "工具边界"),
    ("trace", "runstore", "privacy", "简历", "日志"),
    ("evaluation", "baseline", "fixture", "评测", "门禁", "优化"),
    ("docker", "compose", "health", "交付", "工程化"),
    ("theme", "css", "preview", "settings panel"),
    ("marketing", "another question", "product copy"),
)


class AblationSemanticEmbedding:
    """Transparent bag-of-terms embedding for the labelled ablation corpus."""

    @property
    def model_name(self) -> str:
        return "ablation-semantic-v1"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            normalized = text.casefold()
            vector = [
                float(any(term in normalized for term in concept))
                for concept in _CONCEPT_TERMS
            ]
            vector.append(0.01)
            vectors.append(vector)
        return vectors


def _aggregate(
    rankings: list[tuple[str, ...]],
    *,
    k: int,
    cases: Sequence[LabeledRetrievalCase] = CASES,
) -> dict[str, float]:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    hits: list[float] = []
    for case, ranking in zip(cases, rankings, strict=True):
        relevant = set(case.relevant_sources)
        recalls.append(recall_at_k(ranking, relevant, k=k))
        reciprocal_ranks.append(reciprocal_rank(ranking, relevant))
        ndcgs.append(ndcg_at_k(ranking, relevant, k=k))
        hits.append(1.0 if set(ranking[:k]) & relevant else 0.0)
    if not recalls:
        return {
            f"recall_at_{k}": 0.0,
            "mrr": 0.0,
            f"ndcg_at_{k}": 0.0,
            f"hit_at_{k}": 0.0,
            "case_count": 0,
        }
    return {
        f"recall_at_{k}": round(sum(recalls) / len(recalls), 4),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
        f"ndcg_at_{k}": round(sum(ndcgs) / len(ndcgs), 4),
        f"hit_at_{k}": round(sum(hits) / len(hits), 4),
        "case_count": len(recalls),
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(values)

    def pct(p: float) -> float:
        rank = (len(ordered) - 1) * p
        low = int(rank)
        high = min(low + 1, len(ordered) - 1)
        weight = rank - low
        return round(ordered[low] * (1 - weight) + ordered[high] * weight, 3)

    return {
        "count": len(values),
        "avg_ms": round(statistics.fmean(values), 3),
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
    }


_CATEGORY_PIPELINES: tuple[str, ...] = ("BM25", "Vector", "Hybrid + RRF")


def _metrics_by_category(
    rankings_by_pipeline: dict[str, list[tuple[str, ...]]],
    *,
    k: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Split BM25 / Vector / Hybrid metrics by labelled query category."""

    categories = sorted({case.category for case in CASES})
    out: dict[str, dict[str, dict[str, float]]] = {}
    for category in categories:
        indices = [i for i, case in enumerate(CASES) if case.category == category]
        subset_cases = tuple(CASES[i] for i in indices)
        out[category] = {}
        for label in _CATEGORY_PIPELINES:
            subset_rankings = [rankings_by_pipeline[label][i] for i in indices]
            out[category][label] = _aggregate(subset_rankings, k=k, cases=subset_cases)
    return out


async def evaluate_retrieval_ablation(
    root: Path,
    *,
    k: int = 3,
    live: bool = False,
    rewrite_model: str = "",
    rewrite_api_key: str = "",
    rewrite_base_url: str | None = None,
    rewrite_timeout_s: float = 45.0,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict[str, object]:
    """Run labelled corpus through each ablation pipeline and report metrics."""

    store = KnowledgeStore(root / "ablation.db")
    store.initialize()
    provider = AblationSemanticEmbedding()

    rewriter: QueryRewriter
    scorer: CrossEncoderScorer
    if live:
        if not rewrite_model.strip() or not rewrite_api_key.strip():
            raise RuntimeError("live ablation requires rewrite model and api key")
        rewriter = OpenAICompatibleQueryRewriter(
            model=rewrite_model,
            api_key=rewrite_api_key,
            base_url=rewrite_base_url,
            timeout_s=rewrite_timeout_s,
        )
        scorer = SentenceTransformerCrossEncoder(rerank_model)
        # Fail fast if the neural model cannot load.
        _ = scorer.score("warmup", ["warmup passage about idempotency keys"])
        backend_notes = [
            f"live rewrite model={rewrite_model}",
            f"live rerank scorer={scorer_label(scorer)}",
            "vector path still uses ablation-semantic-v1 fixture embeddings",
        ]
    else:
        rewriter = deterministic_rewrite
        scorer = LexicalOverlapScorer()
        backend_notes = [
            "offline rewrite=deterministic_rewrite",
            "offline rerank=LexicalOverlapScorer",
            "CI 12-case deterministic fixture remains unchanged in evaluation.retrieval",
        ]

    retriever = HybridKnowledgeRetriever(
        store,
        provider,
        query_rewrite_enabled=False,
        query_rewriter=rewriter,
        rerank=RerankSettings(
            enabled=False,
            candidate_k=15,
            model=rerank_model,
            prefer_neural=live,
        ),
        rerank_scorer=scorer,
    )
    for name, content in DOCUMENTS.items():
        path = root / name
        path.write_text(content, encoding="utf-8")
        await retriever.index_document(path)

    metrics: dict[str, dict[str, float]] = {}
    rankings_by_pipeline: dict[str, list[tuple[str, ...]]] = {}
    stage_latency: dict[str, dict[str, object]] = {}
    case_rows: list[dict[str, object]] = []
    for label, mode in PIPELINES:
        rankings: list[tuple[str, ...]] = []
        rewrite_ms: list[float] = []
        rewrite_triggered_ms: list[float] = []
        rerank_ms: list[float] = []
        for case in CASES:
            outcome = await retriever.search_with_trace(
                case.query,
                limit=k,
                candidate_limit=max(15, k),
                mode=mode,
            )
            ranking = tuple(item.result.source_name for item in outcome.results)
            rankings.append(ranking)
            latency = float(outcome.trace.rewrite.rewrite_latency_ms)
            rewrite_ms.append(latency)
            if outcome.trace.rewrite.rewrite_used:
                rewrite_triggered_ms.append(latency)
            rerank_ms.append(float(outcome.trace.rerank.rerank_latency_ms))
            if label == "Final":
                case_rows.append(
                    {
                        "case_id": case.case_id,
                        "category": case.category,
                        "query": case.query,
                        "relevant_sources": list(case.relevant_sources),
                        "ranking": list(ranking),
                        "query_type": outcome.trace.rewrite.query_type,
                        "rewrite_used": outcome.trace.rewrite.rewrite_used,
                        "rewrite_fallback": outcome.trace.rewrite.rewrite_fallback,
                        "rerank_used": outcome.trace.rerank.rerank_used,
                        "rerank_fallback": outcome.trace.rerank.rerank_fallback,
                        "rerank_scorer": outcome.trace.rerank.scorer,
                        "rewrite_latency_ms": outcome.trace.rewrite.rewrite_latency_ms,
                        "rerank_latency_ms": outcome.trace.rerank.rerank_latency_ms,
                    }
                )
        rankings_by_pipeline[label] = rankings
        metrics[label] = _aggregate(rankings, k=k)
        stage_latency[label] = {
            "rewrite": _latency_summary(rewrite_ms),
            "rewrite_triggered_only": _latency_summary(rewrite_triggered_ms),
            "rerank": _latency_summary(rerank_ms),
        }

    report: dict[str, object] = {
        "fixture": "omniagent-retrieval-ablation-v1",
        "mode": "live" if live else "offline",
        "scope": "local labelled ablation on a small manual corpus; not production SLA",
        "case_count": len(CASES),
        "document_count": len(DOCUMENTS),
        "k": k,
        "pipelines": [label for label, _mode in PIPELINES],
        "metrics": metrics,
        "metrics_by_category": _metrics_by_category(rankings_by_pipeline, k=k),
        "stage_latency_ms": stage_latency,
        "cases": case_rows,
        "notes": backend_notes,
    }
    return report


def write_ablation_report(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_ablation_table(report: dict[str, object]) -> str:
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    k = int(report["k"])
    lines = [
        f"{'Pipeline':<22} {f'Recall@{k}':>10} {'MRR':>8} {f'NDCG@{k}':>10}",
        "-" * 54,
    ]
    for label, _mode in PIPELINES:
        row = metrics[label]
        assert isinstance(row, dict)
        lines.append(
            f"{label:<22} {row[f'recall_at_{k}']:>10.4f} {row['mrr']:>8.4f} "
            f"{row[f'ndcg_at_{k}']:>10.4f}"
        )
    return "\n".join(lines)


def format_category_table(report: dict[str, object]) -> str:
    by_cat = report.get("metrics_by_category")
    assert isinstance(by_cat, dict)
    k = int(report["k"])
    lines = [
        f"{'Category':<14} {'Pipeline':<14} {'n':>3} {f'Recall@{k}':>10} {'MRR':>8} "
        f"{f'NDCG@{k}':>10}",
        "-" * 64,
    ]
    for category in sorted(by_cat):
        pipelines = by_cat[category]
        assert isinstance(pipelines, dict)
        for label in _CATEGORY_PIPELINES:
            row = pipelines[label]
            assert isinstance(row, dict)
            lines.append(
                f"{category:<14} {label:<14} {int(row['case_count']):>3} "
                f"{row[f'recall_at_{k}']:>10.4f} {row['mrr']:>8.4f} "
                f"{row[f'ndcg_at_{k}']:>10.4f}"
            )
    return "\n".join(lines)
