"""Standard information-retrieval metrics for labelled query sets."""

from __future__ import annotations

import math
from collections.abc import Sequence, Set


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], *, k: int) -> float:
    """Return the fraction of relevant items found in the first ``k`` results."""

    _validate(relevant, k)
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str]) -> float:
    """Return the reciprocal rank of the first relevant result, or zero."""

    if not relevant:
        raise ValueError("relevant items must not be empty")
    for rank, item_id in enumerate(retrieved, 1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Set[str], *, k: int) -> float:
    """Return binary normalized discounted cumulative gain at ``k``."""

    _validate(relevant, k)
    gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, item_id in enumerate(retrieved[:k], 1)
        if item_id in relevant
    )
    ideal_count = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return gain / ideal


def _validate(relevant: Set[str], k: int) -> None:
    if not relevant:
        raise ValueError("relevant items must not be empty")
    if k < 1:
        raise ValueError("k must be positive")
