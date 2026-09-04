"""SQLite persistence and lexical search for the local knowledge base."""

from __future__ import annotations

import math
import re
import sqlite3
import struct
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path

from nanobot.knowledge.models import (
    KnowledgeChunk,
    KnowledgeSearchResult,
    SourceDocument,
)

_SCHEMA_VERSION = 2
_QUERY_TERM_RE = re.compile(r"[\w-]+", re.UNICODE)


class KnowledgeStore:
    """Persist documents and chunks in a small local SQLite database."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    document_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    media_type TEXT,
                    updated_at TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    heading TEXT,
                    FOREIGN KEY(document_id) REFERENCES knowledge_documents(document_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document
                    ON knowledge_chunks(document_id);
                CREATE TABLE IF NOT EXISTS knowledge_embeddings (
                    chunk_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY(chunk_id, model_name),
                    FOREIGN KEY(chunk_id) REFERENCES knowledge_chunks(chunk_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_embeddings_model
                    ON knowledge_embeddings(model_name);
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    chunk_text,
                    heading,
                    tokenize='unicode61'
                );
                """
            )
            connection.execute(
                """
                INSERT INTO schema_meta(key, value) VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(_SCHEMA_VERSION),),
            )
            connection.commit()

    def replace_document(
        self,
        document: SourceDocument,
        chunks: Iterable[KnowledgeChunk],
    ) -> int:
        """Replace one document and its index atomically; return stored chunk count."""

        materialized = list(chunks)
        for chunk in materialized:
            if chunk.document_id != document.document_id:
                raise ValueError("all chunks must belong to the replaced document")

        with closing(self._connect()) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            try:
                connection.execute("BEGIN IMMEDIATE")
                old_ids = [
                    str(row[0])
                    for row in connection.execute(
                        "SELECT chunk_id FROM knowledge_chunks WHERE document_id = ?",
                        (document.document_id,),
                    )
                ]
                if old_ids:
                    connection.executemany(
                        "DELETE FROM knowledge_chunks_fts WHERE chunk_id = ?",
                        ((chunk_id,) for chunk_id in old_ids),
                    )
                connection.execute(
                    """
                    INSERT INTO knowledge_documents(
                        document_id, source_path, source_name, media_type, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(document_id) DO UPDATE SET
                        source_path = excluded.source_path,
                        source_name = excluded.source_name,
                        media_type = excluded.media_type,
                        updated_at = excluded.updated_at
                    """,
                    (
                        document.document_id,
                        str(document.path),
                        document.source_name,
                        document.media_type,
                        document.updated_at,
                    ),
                )
                connection.execute(
                    "DELETE FROM knowledge_chunks WHERE document_id = ?",
                    (document.document_id,),
                )
                connection.executemany(
                    """
                    INSERT INTO knowledge_chunks(
                        chunk_id, document_id, chunk_text, start_char, end_char, heading
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.text,
                            chunk.start_char,
                            chunk.end_char,
                            chunk.heading,
                        )
                        for chunk in materialized
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO knowledge_chunks_fts(chunk_id, chunk_text, heading)
                    VALUES (?, ?, ?)
                    """,
                    (
                        (chunk.chunk_id, chunk.text, chunk.heading or "")
                        for chunk in materialized
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return len(materialized)

    def replace_embeddings(
        self,
        model_name: str,
        embeddings: Iterable[tuple[str, list[float]]],
    ) -> int:
        """Atomically replace vectors for the supplied chunks and model."""

        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        materialized = list(embeddings)
        dimensions: int | None = None
        packed: list[tuple[str, str, int, bytes]] = []
        for chunk_id, vector in materialized:
            if not vector:
                raise ValueError("embedding vectors must not be empty")
            if any(not math.isfinite(value) for value in vector):
                raise ValueError("embedding vectors must contain only finite values")
            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise ValueError("embedding vectors must have consistent dimensions")
            packed.append(
                (
                    chunk_id,
                    model_name,
                    len(vector),
                    struct.pack(f"<{len(vector)}f", *vector),
                )
            )

        with closing(self._connect()) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            try:
                connection.execute("BEGIN IMMEDIATE")
                if packed:
                    connection.executemany(
                        "DELETE FROM knowledge_embeddings WHERE chunk_id = ? AND model_name = ?",
                        ((chunk_id, model_name) for chunk_id, _, _, _ in packed),
                    )
                    connection.executemany(
                        """
                        INSERT INTO knowledge_embeddings(
                            chunk_id, model_name, dimensions, vector
                        ) VALUES (?, ?, ?, ?)
                        """,
                        packed,
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return len(packed)

    def search_vector(
        self,
        query_vector: list[float],
        *,
        model_name: str,
        limit: int = 5,
    ) -> list[KnowledgeSearchResult]:
        """Rank stored vectors by cosine similarity for a small local index."""

        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        if not query_vector or any(not math.isfinite(value) for value in query_vector):
            raise ValueError("query_vector must contain finite values")
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if query_norm == 0:
            raise ValueError("query_vector must not be a zero vector")

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    c.chunk_id,
                    c.document_id,
                    c.chunk_text,
                    c.start_char,
                    c.end_char,
                    c.heading,
                    d.source_path,
                    d.source_name,
                    e.dimensions,
                    e.vector
                FROM knowledge_embeddings AS e
                JOIN knowledge_chunks AS c USING(chunk_id)
                JOIN knowledge_documents AS d USING(document_id)
                WHERE e.model_name = ? AND e.dimensions = ?
                """,
                (model_name, len(query_vector)),
            ).fetchall()

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            dimensions = int(row[8])
            vector = struct.unpack(f"<{dimensions}f", bytes(row[9]))
            vector_norm = math.sqrt(sum(value * value for value in vector))
            if vector_norm == 0:
                continue
            score = sum(left * right for left, right in zip(query_vector, vector, strict=True))
            score /= query_norm * vector_norm
            scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], str(item[1][0])))

        return [
            KnowledgeSearchResult(
                chunk=KnowledgeChunk(
                    chunk_id=str(row[0]),
                    document_id=str(row[1]),
                    text=str(row[2]),
                    start_char=int(row[3]),
                    end_char=int(row[4]),
                    heading=str(row[5]) if row[5] is not None else None,
                ),
                source_path=Path(str(row[6])),
                source_name=str(row[7]),
                score=score,
                rank=rank,
            )
            for rank, (score, row) in enumerate(scored[:limit], 1)
        ]

    def search_lexical(self, query: str, *, limit: int = 5) -> list[KnowledgeSearchResult]:
        """Search chunks with SQLite BM25 and return stable source metadata."""

        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        terms = _QUERY_TERM_RE.findall(query.casefold())
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    c.chunk_id,
                    c.document_id,
                    c.chunk_text,
                    c.start_char,
                    c.end_char,
                    c.heading,
                    d.source_path,
                    d.source_name,
                    -bm25(knowledge_chunks_fts) AS score
                FROM knowledge_chunks_fts
                JOIN knowledge_chunks AS c USING(chunk_id)
                JOIN knowledge_documents AS d USING(document_id)
                WHERE knowledge_chunks_fts MATCH ?
                ORDER BY score DESC, c.chunk_id ASC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()

        return [
            KnowledgeSearchResult(
                chunk=KnowledgeChunk(
                    chunk_id=str(row[0]),
                    document_id=str(row[1]),
                    text=str(row[2]),
                    start_char=int(row[3]),
                    end_char=int(row[4]),
                    heading=str(row[5]) if row[5] is not None else None,
                ),
                source_path=Path(str(row[6])),
                source_name=str(row[7]),
                score=float(row[8]),
                rank=rank,
            )
            for rank, row in enumerate(rows, 1)
        ]

    def document_count(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT COUNT(*) FROM knowledge_documents").fetchone()
        return int(row[0]) if row is not None else 0

    def chunk_count(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()
        return int(row[0]) if row is not None else 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection
