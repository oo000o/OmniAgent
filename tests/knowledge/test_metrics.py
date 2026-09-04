import math
from collections.abc import Callable

import pytest

from nanobot.knowledge.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_retrieval_metrics_reward_early_relevant_results() -> None:
    retrieved = ["wrong", "relevant", "also-relevant"]
    relevant = {"relevant", "also-relevant"}

    assert recall_at_k(retrieved, relevant, k=2) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(
        (1 / math.log2(3) + 1 / math.log2(4))
        / (1 / math.log2(2) + 1 / math.log2(3))
    )


def test_retrieval_metrics_return_zero_without_a_hit() -> None:
    retrieved = ["wrong"]
    relevant = {"relevant"}

    assert recall_at_k(retrieved, relevant, k=3) == 0.0
    assert reciprocal_rank(retrieved, relevant) == 0.0
    assert ndcg_at_k(retrieved, relevant, k=3) == 0.0


@pytest.mark.parametrize(
    ("metric", "message"),
    [
        (lambda: recall_at_k([], set(), k=1), "relevant"),
        (lambda: reciprocal_rank([], set()), "relevant"),
        (lambda: ndcg_at_k([], {"item"}, k=0), "positive"),
    ],
)
def test_retrieval_metrics_reject_invalid_labels(
    metric: Callable[[], float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        metric()
