"""Deterministic document chunking with stable source offsets."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from nanobot.knowledge.models import KnowledgeChunk, SourceDocument

_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*$")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Character-based limits used by the first local retrieval index."""

    chunk_size: int = 1_000
    overlap: int = 150

    def __post_init__(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")


def _chunk_id(document_id: str, start: int, end: int, text: str) -> str:
    digest = hashlib.sha256(
        f"{document_id}\0{start}\0{end}\0{text}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{document_id}:{digest}"


def _heading_at(text: str, position: int) -> str | None:
    heading: str | None = None
    for match in _HEADING_RE.finditer(text, 0, position + 1):
        heading = match.group(1).strip()
    return heading


def _prefer_boundary(text: str, start: int, proposed_end: int) -> int:
    if proposed_end >= len(text):
        return len(text)
    minimum = start + max(1, (proposed_end - start) // 2)
    for marker in ("\n\n", "\n", "。", ". ", " "):
        boundary = text.rfind(marker, minimum, proposed_end)
        if boundary >= minimum:
            return boundary + len(marker)
    return proposed_end


def chunk_document(
    document: SourceDocument,
    config: ChunkingConfig | None = None,
) -> list[KnowledgeChunk]:
    """Split a document without losing the offsets required for citations."""

    settings = config or ChunkingConfig()
    if not document.text.strip():
        return []

    chunks: list[KnowledgeChunk] = []
    start = 0
    text_length = len(document.text)
    while start < text_length:
        proposed_end = min(text_length, start + settings.chunk_size)
        end = _prefer_boundary(document.text, start, proposed_end)
        if end <= start:
            end = proposed_end
        chunk_text = document.text[start:end]
        if chunk_text.strip():
            chunks.append(
                KnowledgeChunk(
                    chunk_id=_chunk_id(document.document_id, start, end, chunk_text),
                    document_id=document.document_id,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    heading=_heading_at(document.text, start),
                )
            )
        if end == text_length:
            break
        start = max(start + 1, end - settings.overlap)
    return chunks
