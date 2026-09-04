from pathlib import Path

from evaluation.retrieval import CASES, evaluate_retrieval


async def test_retrieval_evaluation_compares_all_strategies(tmp_path: Path) -> None:
    report = await evaluate_retrieval(tmp_path)

    assert report["case_count"] == len(CASES)
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    assert set(metrics) == {"bm25", "vector", "hybrid_rrf"}
    assert metrics["vector"]["recall_at_3"] >= metrics["bm25"]["recall_at_3"]
    assert metrics["hybrid_rrf"]["recall_at_3"] >= metrics["bm25"]["recall_at_3"]
