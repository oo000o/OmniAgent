"""Lightweight reproducible local benchmarks (not production SLA)."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from evaluation.career_workflow import evaluate_career_workflow
from evaluation.fault_injection import evaluate_fault_injection
from evaluation.retrieval_ablation import AblationSemanticEmbedding
from evaluation.retrieval_corpus import CASES, DOCUMENTS
from nanobot.knowledge import HybridKnowledgeRetriever, KnowledgeStore
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


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 3)
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 3)


def _summary(latencies_ms: list[float], *, successes: int, runs: int) -> dict[str, object]:
    return {
        "runs": runs,
        "success_rate": round(successes / runs, 4) if runs else 0.0,
        "avg_ms": round(statistics.fmean(latencies_ms), 3) if latencies_ms else 0.0,
        "p50_ms": _percentile(latencies_ms, 0.50),
        "p95_ms": _percentile(latencies_ms, 0.95),
    }


async def _bench_retrieval(
    root: Path,
    *,
    live: bool = False,
    rewrite_model: str = "",
    rewrite_api_key: str = "",
    rewrite_base_url: str | None = None,
    rewrite_timeout_s: float = 45.0,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict[str, object]:
    store = KnowledgeStore(root / "bench-retrieval.db")
    store.initialize()
    provider = AblationSemanticEmbedding()

    rewriter: QueryRewriter
    scorer: CrossEncoderScorer
    if live:
        rewriter = OpenAICompatibleQueryRewriter(
            model=rewrite_model,
            api_key=rewrite_api_key,
            base_url=rewrite_base_url,
            timeout_s=rewrite_timeout_s,
        )
        scorer = SentenceTransformerCrossEncoder(rerank_model)
        _ = scorer.score("warmup", ["warmup passage about idempotency keys"])
    else:
        rewriter = deterministic_rewrite
        scorer = LexicalOverlapScorer()

    base = HybridKnowledgeRetriever(
        store,
        provider,
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
        await base.index_document(path)

    queries = [case.query for case in CASES]
    modes = {
        "RRF only": "hybrid",
        "RRF + Rerank": "hybrid_rerank",
        "Final rewrite+rerank": "hybrid_rewrite_rerank",
    }
    sections: dict[str, object] = {"scorer": scorer_label(scorer), "live": live}
    for label, mode in modes.items():
        latencies: list[float] = []
        rewrite_ms: list[float] = []
        rewrite_triggered_ms: list[float] = []
        rerank_ms: list[float] = []
        successes = 0
        for query in queries:
            started = time.perf_counter()
            outcome = await base.search_with_trace(
                query, limit=3, candidate_limit=15, mode=mode  # type: ignore[arg-type]
            )
            total_ms = (time.perf_counter() - started) * 1_000
            latencies.append(total_ms)
            rewrite_latency = float(outcome.trace.rewrite.rewrite_latency_ms)
            rewrite_ms.append(rewrite_latency)
            if outcome.trace.rewrite.rewrite_used:
                rewrite_triggered_ms.append(rewrite_latency)
            rerank_ms.append(float(outcome.trace.rerank.rerank_latency_ms))
            if outcome.results:
                successes += 1
        summary = _summary(latencies, successes=successes, runs=len(queries))
        summary["rewrite_avg_ms"] = round(statistics.fmean(rewrite_ms), 3) if rewrite_ms else 0.0
        summary["rewrite_p50_ms"] = _percentile(rewrite_ms, 0.50)
        summary["rewrite_p95_ms"] = _percentile(rewrite_ms, 0.95)
        summary["rewrite_triggered_count"] = len(rewrite_triggered_ms)
        summary["rewrite_triggered_p50_ms"] = _percentile(rewrite_triggered_ms, 0.50)
        summary["rewrite_triggered_p95_ms"] = _percentile(rewrite_triggered_ms, 0.95)
        summary["rerank_avg_ms"] = round(statistics.fmean(rerank_ms), 3) if rerank_ms else 0.0
        summary["rerank_p50_ms"] = _percentile(rerank_ms, 0.50)
        summary["rerank_p95_ms"] = _percentile(rerank_ms, 0.95)
        summary["retrieval_avg_ms"] = round(
            max(
                0.0,
                float(summary["avg_ms"])
                - float(summary["rewrite_avg_ms"])
                - float(summary["rerank_avg_ms"]),
            ),
            3,
        )
        sections[label] = summary
    return sections


async def evaluate_benchmark(
    root: Path,
    *,
    live: bool = False,
    rewrite_model: str = "",
    rewrite_api_key: str = "",
    rewrite_base_url: str | None = None,
    rewrite_timeout_s: float = 45.0,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict[str, object]:
    """Run local timing benches over fixed offline fixtures."""

    retrieval = await _bench_retrieval(
        root / "retrieval",
        live=live,
        rewrite_model=rewrite_model,
        rewrite_api_key=rewrite_api_key,
        rewrite_base_url=rewrite_base_url,
        rewrite_timeout_s=rewrite_timeout_s,
        rerank_model=rerank_model,
    )
    career = await evaluate_career_workflow(root / "career")
    fault = await evaluate_fault_injection(root / "fault")

    def _group_summary(report: dict[str, object], *, label: str) -> dict[str, object]:
        total = int(report["total"])
        passed = int(report["passed"])
        return {
            "label": label,
            "runs": total,
            "success_rate": round(passed / total, 4) if total else 0.0,
            "note": "pass/fail fixture timing is not wall-clock SLA; see retrieval section for p50/p95",
        }

    return {
        "fixture": "omniagent-local-benchmark-v1",
        "mode": "live" if live else "offline",
        "scope": "local/repository benchmark on labelled fixtures; not production SLA",
        "environment": "developer machine live or offline fixtures",
        "retrieval": retrieval,
        "task_mcp_and_career_workflow": _group_summary(career, label="career_workflow"),
        "recovery_fault_injection": _group_summary(fault, label="fault_injection"),
        "career_workflow_pass_rate": career.get("pass_rate"),
        "fault_injection_pass_rate": fault.get("pass_rate"),
    }


def write_benchmark_report(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
