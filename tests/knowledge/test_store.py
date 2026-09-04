from pathlib import Path

import pytest

from nanobot.knowledge import ChunkingConfig, KnowledgeStore
from nanobot.knowledge.ingest import KnowledgeIngestionError, ingest_document


def test_ingest_and_lexical_search_returns_source(tmp_path: Path) -> None:
    source = tmp_path / "memory.md"
    source.write_text(
        "# Agent Memory\n\nLong-term memory persists useful facts across sessions.",
        encoding="utf-8",
    )
    store = KnowledgeStore(tmp_path / "knowledge.db")
    store.initialize()

    stored = ingest_document(
        source,
        store,
        config=ChunkingConfig(chunk_size=200, overlap=20),
    )
    results = store.search_lexical("long-term memory")

    assert stored == 1
    assert store.document_count() == 1
    assert store.chunk_count() == 1
    assert results[0].source_name == "memory.md"
    assert "persists useful facts" in results[0].chunk.text


def test_reingest_replaces_old_chunks_and_fts_rows(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("obsolete phrase " * 30, encoding="utf-8")
    store = KnowledgeStore(tmp_path / "knowledge.db")
    store.initialize()
    config = ChunkingConfig(chunk_size=120, overlap=10)
    ingest_document(source, store, config=config)

    source.write_text("current phrase " * 5, encoding="utf-8")
    ingest_document(source, store, config=config)

    assert store.document_count() == 1
    assert store.search_lexical("obsolete") == []
    assert store.search_lexical("current")


def test_rejects_unsupported_document(tmp_path: Path) -> None:
    source = tmp_path / "archive.bin"
    source.write_bytes(b"binary")
    store = KnowledgeStore(tmp_path / "knowledge.db")
    store.initialize()

    with pytest.raises(KnowledgeIngestionError, match="unsupported document type"):
        ingest_document(source, store)


def test_search_limit_is_bounded(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    store.initialize()

    with pytest.raises(ValueError, match="between 1 and 100"):
        store.search_lexical("query", limit=101)
