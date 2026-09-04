"""Safe local document ingestion into the knowledge store."""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

from nanobot.knowledge.chunking import ChunkingConfig, chunk_document
from nanobot.knowledge.models import KnowledgeChunk, SourceDocument
from nanobot.knowledge.store import KnowledgeStore
from nanobot.utils.document import extract_text


class KnowledgeIngestionError(ValueError):
    """Raised when a source cannot be converted into searchable text."""


def document_id_for_path(path: Path) -> str:
    """Return a stable identifier without exposing an absolute path in tool output."""

    resolved = str(path.resolve()).casefold().encode("utf-8")
    return hashlib.sha256(resolved).hexdigest()[:24]


def ingest_document(
    path: Path,
    store: KnowledgeStore,
    *,
    config: ChunkingConfig | None = None,
) -> int:
    """Extract, chunk, and atomically replace one already-authorized local file."""

    document, chunks = prepare_document(path, config=config)
    return store.replace_document(document, chunks)


def prepare_document(
    path: Path,
    *,
    config: ChunkingConfig | None = None,
) -> tuple[SourceDocument, list[KnowledgeChunk]]:
    """Extract and chunk one already-authorized file without mutating an index."""

    extracted = extract_text(path)
    if extracted is None:
        raise KnowledgeIngestionError(f"unsupported document type: {path.suffix or '<none>'}")
    if extracted.startswith("[error:"):
        raise KnowledgeIngestionError(extracted)
    if not extracted.strip():
        raise KnowledgeIngestionError("document does not contain searchable text")

    document = SourceDocument(
        document_id=document_id_for_path(path),
        path=path,
        text=extracted,
        media_type=mimetypes.guess_type(path.name)[0],
    )
    return document, chunk_document(document, config)
