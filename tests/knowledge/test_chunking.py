from pathlib import Path

import pytest

from nanobot.knowledge.chunking import ChunkingConfig, chunk_document
from nanobot.knowledge.models import SourceDocument


def _document(text: str) -> SourceDocument:
    return SourceDocument("doc-1", Path("guide.md"), text, "text/markdown")


def test_empty_document_has_no_chunks() -> None:
    assert chunk_document(_document("  \n")) == []


def test_chunk_offsets_reconstruct_chunk_text() -> None:
    text = "# Intro\n\n" + "alpha beta gamma. " * 50
    chunks = chunk_document(_document(text), ChunkingConfig(chunk_size=160, overlap=20))

    assert len(chunks) > 1
    for chunk in chunks:
        assert text[chunk.start_char : chunk.end_char] == chunk.text


def test_chunk_ids_are_stable() -> None:
    document = _document("# Stable\n\n" + "content " * 40)
    config = ChunkingConfig(chunk_size=120, overlap=10)

    first = chunk_document(document, config)
    second = chunk_document(document, config)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_chunk_tracks_nearest_markdown_heading() -> None:
    text = "# First\n\n" + "a " * 80 + "\n\n## Second\n\n" + "b " * 80
    chunks = chunk_document(_document(text), ChunkingConfig(chunk_size=120, overlap=10))

    assert chunks[0].heading is None
    assert any(chunk.heading == "Second" for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(99, 0), (100, -1), (100, 100), (100, 101)],
)
def test_invalid_chunking_config_is_rejected(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=chunk_size, overlap=overlap)
