from collections.abc import Sequence
from pathlib import Path

from nanobot.knowledge import ChunkingConfig, HybridKnowledgeRetriever, KnowledgeStore


class FakeEmbeddingProvider:
    @property
    def model_name(self) -> str:
        return "fake-v1"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [
                float("memory" in text.casefold()),
                float("tool" in text.casefold()),
                1.0,
            ]
            for text in texts
        ]


async def test_hybrid_retriever_indexes_and_searches(tmp_path: Path) -> None:
    source = tmp_path / "agent.md"
    source.write_text(
        "# Agent\n\nLong-term memory persists facts.\n\n# Tools\n\nTools perform actions.",
        encoding="utf-8",
    )
    store = KnowledgeStore(tmp_path / "knowledge.db")
    store.initialize()
    retriever = HybridKnowledgeRetriever(
        store,
        FakeEmbeddingProvider(),
        chunking=ChunkingConfig(chunk_size=100, overlap=10),
    )

    count = await retriever.index_document(source)
    results = await retriever.search("memory", limit=2)

    assert count >= 1
    assert results
    assert "memory" in results[0].result.chunk.text.casefold()
    assert results[0].result.source_name == "agent.md"
