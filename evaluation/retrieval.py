"""Deterministic lexical, vector, and hybrid retrieval comparison."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from nanobot.knowledge import HybridKnowledgeRetriever, KnowledgeStore
from nanobot.knowledge.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    query: str
    relevant_source: str


DOCUMENTS = {
    "memory.md": "Durable memory preserves user facts across separate conversations.",
    "tools.md": "Tool execution lets an agent perform actions in external systems.",
    "locking.md": "Optimistic locking rejects stale concurrent task updates.",
    "fusion.md": "Reciprocal rank fusion merges lexical and semantic result lists.",
    "citations.md": "Source citations make retrieved evidence traceable.",
    "scheduling.md": "Cron schedules trigger reminders at a specified time.",
}

CASES = (
    RetrievalCase("durable memory", "memory.md"),
    RetrievalCase("remember details between chats", "memory.md"),
    RetrievalCase("tool execution", "tools.md"),
    RetrievalCase("take action outside the agent", "tools.md"),
    RetrievalCase("optimistic locking", "locking.md"),
    RetrievalCase("prevent two writers overwriting changes", "locking.md"),
    RetrievalCase("rank fusion", "fusion.md"),
    RetrievalCase("combine keyword and meaning search", "fusion.md"),
    RetrievalCase("source citations", "citations.md"),
    RetrievalCase("show where an answer came from", "citations.md"),
    RetrievalCase("cron schedules", "scheduling.md"),
    RetrievalCase("send a reminder later", "scheduling.md"),
)

_CONCEPT_TERMS = (
    ("memory", "remember", "facts", "between chats", "conversations"),
    ("tool", "action", "external", "outside the agent"),
    ("locking", "concurrent", "two writers", "overwriting", "stale"),
    ("fusion", "keyword and meaning", "lexical and semantic", "combine"),
    ("citation", "citations", "where an answer came from", "traceable"),
    ("cron", "schedule", "reminder", "later", "specified time"),
)


class DeterministicSemanticEmbedding:
    """Transparent semantic fixture used by CI; it is not a production model."""

    @property
    def model_name(self) -> str:
        return "deterministic-semantic-v1"

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


async def evaluate_retrieval(root: Path, *, k: int = 3) -> dict[str, object]:
    """Index the labelled corpus and compare three retrieval strategies."""

    store = KnowledgeStore(root / "retrieval.db")
    store.initialize()
    provider = DeterministicSemanticEmbedding()
    retriever = HybridKnowledgeRetriever(store, provider)
    for name, content in DOCUMENTS.items():
        path = root / name
        path.write_text(content, encoding="utf-8")
        await retriever.index_document(path)

    strategies: dict[str, list[tuple[str, ...]]] = {
        "bm25": [],
        "vector": [],
        "hybrid_rrf": [],
    }
    cases: list[dict[str, object]] = []
    for case in CASES:
        query_vector = (await provider.embed([case.query]))[0]
        lexical = store.search_lexical(case.query, limit=k)
        vector = store.search_vector(query_vector, model_name=provider.model_name, limit=k)
        hybrid = await retriever.search(case.query, limit=k, candidate_limit=10)
        ranked = {
            "bm25": tuple(item.source_name for item in lexical),
            "vector": tuple(item.source_name for item in vector),
            "hybrid_rrf": tuple(item.result.source_name for item in hybrid),
        }
        for strategy, source_names in ranked.items():
            strategies[strategy].append(source_names)
        cases.append(
            {
                "query": case.query,
                "relevant_source": case.relevant_source,
                "results": ranked,
            }
        )

    metrics: dict[str, dict[str, float]] = {}
    for strategy, rankings in strategies.items():
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        ndcgs: list[float] = []
        for case, ranking in zip(CASES, rankings, strict=True):
            relevant = {case.relevant_source}
            recalls.append(recall_at_k(ranking, relevant, k=k))
            reciprocal_ranks.append(reciprocal_rank(ranking, relevant))
            ndcgs.append(ndcg_at_k(ranking, relevant, k=k))
        metrics[strategy] = {
            f"recall_at_{k}": round(sum(recalls) / len(recalls), 4),
            "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
            f"ndcg_at_{k}": round(sum(ndcgs) / len(ndcgs), 4),
        }
    return {
        "fixture": "deterministic-semantic-v1",
        "case_count": len(CASES),
        "k": k,
        "metrics": metrics,
        "cases": cases,
    }
