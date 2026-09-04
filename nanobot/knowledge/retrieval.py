"""End-to-end ingestion and hybrid retrieval service."""

from __future__ import annotations

from pathlib import Path

from nanobot.knowledge.chunking import ChunkingConfig
from nanobot.knowledge.embeddings import EmbeddingProvider, EmbeddingProviderError
from nanobot.knowledge.fusion import FusedSearchResult, reciprocal_rank_fusion
from nanobot.knowledge.ingest import prepare_document
from nanobot.knowledge.store import KnowledgeStore


class HybridKnowledgeRetriever:
    """Combine lexical and semantic retrieval behind one small service."""

    def __init__(
        self,
        store: KnowledgeStore,
        embedding_provider: EmbeddingProvider,
        *,
        chunking: ChunkingConfig | None = None,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.chunking = chunking or ChunkingConfig()

    async def index_document(self, path: Path) -> int:
        """Replace a source document and its model-specific vectors."""

        document, chunks = prepare_document(path, config=self.chunking)
        vectors = await self.embedding_provider.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise EmbeddingProviderError("embedding provider returned the wrong vector count")
        self.store.replace_document(document, chunks)
        self.store.replace_embeddings(
            self.embedding_provider.model_name,
            [(chunk.chunk_id, vector) for chunk, vector in zip(chunks, vectors, strict=True)],
        )
        return len(chunks)

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
    ) -> list[FusedSearchResult]:
        """Retrieve lexical and semantic candidates, then fuse their ranks."""

        if not query.strip():
            return []
        if candidate_limit < limit or candidate_limit > 100:
            raise ValueError("candidate_limit must be between limit and 100")
        query_vectors = await self.embedding_provider.embed([query])
        if len(query_vectors) != 1:
            raise EmbeddingProviderError(
                "embedding provider returned the wrong query vector count"
            )
        lexical = self.store.search_lexical(query, limit=candidate_limit)
        semantic = self.store.search_vector(
            query_vectors[0],
            model_name=self.embedding_provider.model_name,
            limit=candidate_limit,
        )
        return reciprocal_rank_fusion([lexical, semantic], limit=limit)
