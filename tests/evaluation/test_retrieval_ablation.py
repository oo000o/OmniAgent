"""Tests for ablation metrics and seal evaluation helpers."""

from __future__ import annotations

from pathlib import Path

from evaluation.benchmark import _percentile
from evaluation.retrieval_ablation import evaluate_retrieval_ablation
from evaluation.retrieval_corpus import CASES
from nanobot.knowledge.metrics import recall_at_k


def test_metric_helpers_handle_multi_relevant_and_empty_prefix() -> None:
    assert recall_at_k(("a", "b", "c"), {"b", "z"}, k=3) == 0.5
    assert recall_at_k((), {"a"}, k=3) == 0.0


def test_percentile_calculation() -> None:
    assert _percentile([10.0, 20.0, 30.0, 40.0], 0.5) == 25.0
    assert _percentile([10.0], 0.95) == 10.0


async def test_ablation_runs_all_pipelines(tmp_path: Path) -> None:
    report = await evaluate_retrieval_ablation(tmp_path)
    assert report["case_count"] == len(CASES)
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    assert set(metrics) >= {
        "BM25",
        "Vector",
        "Hybrid + RRF",
        "+ Rewrite",
        "+ Rerank",
        "Final",
    }
    for row in metrics.values():
        assert isinstance(row, dict)
        assert "mrr" in row
        assert "recall_at_3" in row

    by_category = report["metrics_by_category"]
    assert isinstance(by_category, dict)
    assert "keyword" in by_category
    for category_metrics in by_category.values():
        assert isinstance(category_metrics, dict)
        assert set(category_metrics) == {"BM25", "Vector", "Hybrid + RRF"}

    final_latency = report["stage_latency_ms"]["Final"]
    assert isinstance(final_latency, dict)
    assert "rewrite" in final_latency
    assert "rewrite_triggered_only" in final_latency
    assert "count" in final_latency["rewrite_triggered_only"]
