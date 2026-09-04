"""Typed records shared by knowledge ingestion, retrieval, and citations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One extracted local document before it is split for retrieval."""

    document_id: str
    path: Path
    text: str
    media_type: str | None = None

    @property
    def source_name(self) -> str:
        return self.path.name

    @property
    def updated_at(self) -> str:
        """Return a stable UTC timestamp when the source exists."""
        try:
            timestamp = self.path.stat().st_mtime
        except OSError:
            return ""
        return datetime.fromtimestamp(timestamp, UTC).isoformat()


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    """A stable, source-addressable unit stored in the retrieval index."""

    chunk_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    heading: str | None = None

    def __post_init__(self) -> None:
        if self.start_char < 0:
            raise ValueError("start_char must be non-negative")
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if len(self.text) != self.end_char - self.start_char:
            raise ValueError("chunk text length must match its source character range")


@dataclass(frozen=True, slots=True)
class KnowledgeSearchResult:
    """One ranked chunk plus enough metadata to render a source citation."""

    chunk: KnowledgeChunk
    source_path: Path
    source_name: str
    score: float
    rank: int
