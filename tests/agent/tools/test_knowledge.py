from collections.abc import Sequence
from pathlib import Path

from nanobot.agent.tools.knowledge import (
    KnowledgeAddTool,
    KnowledgeSearchTool,
    KnowledgeToolsConfig,
)
from nanobot.knowledge import EmbeddingProviderError
from nanobot.knowledge.observability import RetrievalEvent


class FakeEmbeddingProvider:
    @property
    def model_name(self) -> str:
        return "fake-v1"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [float("durable" in text.casefold()), float("memory" in text.casefold()), 1.0]
            for text in texts
        ]


class FailingEmbeddingProvider:
    @property
    def model_name(self) -> str:
        return "failing-v1"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbeddingProviderError("backend unavailable")


async def test_add_then_search_returns_cited_evidence(tmp_path: Path) -> None:
    source = tmp_path / "memory.md"
    source.write_text("Long-term memory persists facts across sessions.", encoding="utf-8")
    config = KnowledgeToolsConfig(database_path="state/knowledge.db")
    add = KnowledgeAddTool(workspace=tmp_path, config=config, restrict_to_workspace=True)
    search = KnowledgeSearchTool(workspace=tmp_path, config=config, restrict_to_workspace=True)

    add_result = await add.execute("memory.md")
    search_result = await search.execute("memory")

    assert "Indexed 'memory.md'" in add_result
    assert "[K1]" in search_result
    assert "persists facts" in search_result


def test_database_path_cannot_escape_workspace() -> None:
    try:
        KnowledgeToolsConfig(database_path="../outside.db")
    except ValueError as exc:
        assert "inside the workspace" in str(exc)
    else:
        raise AssertionError("escaping database path should be rejected")


async def test_hybrid_tool_indexes_vectors_and_returns_semantic_evidence(tmp_path: Path) -> None:
    source = tmp_path / "memory.md"
    source.write_text("Durable state survives process restarts.", encoding="utf-8")
    config = KnowledgeToolsConfig(
        database_path="state/knowledge.db",
        retrieval_mode="hybrid",
        embedding_model="fake-v1",
    )
    provider = FakeEmbeddingProvider()
    add = KnowledgeAddTool(
        workspace=tmp_path,
        config=config,
        restrict_to_workspace=True,
        embedding_provider=provider,
    )
    events: list[RetrievalEvent] = []
    search = KnowledgeSearchTool(
        workspace=tmp_path,
        config=config,
        restrict_to_workspace=True,
        embedding_provider=provider,
        retrieval_observer=events.append,
    )

    add_result = await add.execute("memory.md")
    search_result = await search.execute("durable")

    assert "(hybrid)" in add_result
    assert "[K1]" in search_result
    assert "survives process restarts" in search_result
    assert len(events) == 1
    assert events[0].mode == "hybrid"
    assert events[0].status == "ok"
    assert events[0].result_count == 1


async def test_hybrid_tool_degrades_to_lexical_when_embeddings_fail(tmp_path: Path) -> None:
    source = tmp_path / "memory.md"
    source.write_text("Long-term memory persists facts across sessions.", encoding="utf-8")
    config = KnowledgeToolsConfig(
        database_path="state/knowledge.db",
        retrieval_mode="hybrid",
        embedding_model="failing-v1",
    )
    provider = FailingEmbeddingProvider()
    add = KnowledgeAddTool(
        workspace=tmp_path,
        config=config,
        restrict_to_workspace=True,
        embedding_provider=provider,
    )
    events: list[RetrievalEvent] = []
    search = KnowledgeSearchTool(
        workspace=tmp_path,
        config=config,
        restrict_to_workspace=True,
        embedding_provider=provider,
        retrieval_observer=events.append,
    )

    add_result = await add.execute("memory.md")
    search_result = await search.execute("memory")

    assert "lexical fallback" in add_result
    assert "[Retrieval mode: lexical fallback]" in search_result
    assert "[K1]" in search_result
    assert events[0].mode == "lexical"
    assert events[0].status == "fallback"


async def test_hybrid_tool_reports_failure_when_fallback_is_disabled(tmp_path: Path) -> None:
    source = tmp_path / "memory.md"
    source.write_text("Long-term memory persists facts across sessions.", encoding="utf-8")
    config = KnowledgeToolsConfig(
        database_path="state/knowledge.db",
        retrieval_mode="hybrid",
        embedding_model="failing-v1",
        fallback_to_lexical=False,
    )
    provider = FailingEmbeddingProvider()
    add = KnowledgeAddTool(
        workspace=tmp_path,
        config=config,
        restrict_to_workspace=True,
        embedding_provider=provider,
    )
    search = KnowledgeSearchTool(
        workspace=tmp_path,
        config=config,
        restrict_to_workspace=True,
        embedding_provider=provider,
    )

    add_result = await add.execute("memory.md")
    search_result = await search.execute("memory")

    assert add_result.startswith("Knowledge ingestion failed:")
    assert search_result == "Knowledge search failed: embeddings were unavailable"


def test_hybrid_config_requires_embedding_model() -> None:
    try:
        KnowledgeToolsConfig(retrieval_mode="hybrid")
    except ValueError as exc:
        assert "embedding_model" in str(exc)
    else:
        raise AssertionError("hybrid mode should require an embedding model")
