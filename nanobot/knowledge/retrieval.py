"""End-to-end ingestion and hybrid retrieval service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nanobot.knowledge.chunking import ChunkingConfig
from nanobot.knowledge.embeddings import EmbeddingProvider, EmbeddingProviderError
from nanobot.knowledge.fusion import FusedSearchResult, reciprocal_rank_fusion
from nanobot.knowledge.ingest import prepare_document
from nanobot.knowledge.models import KnowledgeSearchResult
from nanobot.knowledge.query_rewrite import (
    QueryRewriter,
    QueryRewriteTrace,
    prepare_search_queries,
)
from nanobot.knowledge.rerank import (
    CrossEncoderScorer,
    RerankSettings,
    RerankTrace,
    rerank_fused_results,
)
from nanobot.knowledge.store import KnowledgeStore

RetrievalMode = Literal[
    "bm25",
    "vector",
    "hybrid",
    "hybrid_rewrite",
    "hybrid_rerank",
    "hybrid_rewrite_rerank",
]


@dataclass(frozen=True, slots=True)
class RetrievalPipelineTrace:
    """Stage telemetry without query or document bodies."""

    mode: str
    rewrite: QueryRewriteTrace
    rerank: RerankTrace


@dataclass(frozen=True, slots=True)
class RetrievalPipelineResult:
    results: list[FusedSearchResult]
    trace: RetrievalPipelineTrace


class HybridKnowledgeRetriever:
    """Combine lexical and semantic retrieval behind one small service."""

    def __init__(
        self,
        store: KnowledgeStore,
        embedding_provider: EmbeddingProvider,
        *,
        chunking: ChunkingConfig | None = None,
        query_rewrite_enabled: bool = False,
        max_rewritten_queries: int = 3,
        query_rewriter: QueryRewriter | None = None,
        rerank: RerankSettings | None = None,
        rerank_scorer: CrossEncoderScorer | None = None,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.chunking = chunking or ChunkingConfig()
        self.query_rewrite_enabled = query_rewrite_enabled
        self.max_rewritten_queries = max_rewritten_queries
        self.query_rewriter = query_rewriter
        self.rerank = rerank or RerankSettings()
        self.rerank_scorer = rerank_scorer

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
        mode: RetrievalMode = "hybrid",
    ) -> list[FusedSearchResult]:
        """Retrieve and optionally rewrite/rerank; returns fused results only."""

        outcome = await self.search_with_trace(
            query, limit=limit, candidate_limit=candidate_limit, mode=mode
        )
        return outcome.results

    async def search_with_trace(
        self,
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
        mode: RetrievalMode = "hybrid",
    ) -> RetrievalPipelineResult:
        """Full pipeline with stage traces for eval and observability."""

        if not query.strip():
            empty_trace = RetrievalPipelineTrace(
                mode=mode,
                rewrite=QueryRewriteTrace(query_type="simple"),
                rerank=RerankTrace(),
            )
            return RetrievalPipelineResult(results=[], trace=empty_trace)
        if candidate_limit < limit or candidate_limit > 100:
            raise ValueError("candidate_limit must be between limit and 100")

        use_rewrite = mode in {"hybrid_rewrite", "hybrid_rewrite_rerank"}
        use_rerank = mode in {"hybrid_rerank", "hybrid_rewrite_rerank"}
        if mode == "hybrid":
            use_rewrite = self.query_rewrite_enabled
            use_rerank = self.rerank.enabled

        queries, rewrite_trace = await prepare_search_queries(
            query,
            enabled=use_rewrite,
            max_rewritten_queries=self.max_rewritten_queries,
            rewriter=self.query_rewriter,
        )

        if mode == "bm25":
            lexical = self.store.search_lexical(queries[0], limit=limit)
            results = [
                FusedSearchResult(result=item, fused_score=float(len(lexical) - index), contributing_lists=1)
                for index, item in enumerate(lexical)
            ]
            return RetrievalPipelineResult(
                results=results,
                trace=RetrievalPipelineTrace(
                    mode=mode, rewrite=rewrite_trace, rerank=RerankTrace()
                ),
            )

        if mode == "vector":
            query_vectors = await self.embedding_provider.embed([queries[0]])
            if len(query_vectors) != 1:
                raise EmbeddingProviderError(
                    "embedding provider returned the wrong query vector count"
                )
            semantic = self.store.search_vector(
                query_vectors[0],
                model_name=self.embedding_provider.model_name,
                limit=limit,
            )
            results = [
                FusedSearchResult(result=item, fused_score=float(len(semantic) - index), contributing_lists=1)
                for index, item in enumerate(semantic)
            ]
            return RetrievalPipelineResult(
                results=results,
                trace=RetrievalPipelineTrace(
                    mode=mode, rewrite=rewrite_trace, rerank=RerankTrace()
                ),
            )

        pool_limit = candidate_limit
        if use_rerank:
            pool_limit = max(candidate_limit, self.rerank.candidate_k, limit)

        rankings: list[list[KnowledgeSearchResult]] = []
        for search_query in queries:
            query_vectors = await self.embedding_provider.embed([search_query])
            if len(query_vectors) != 1:
                raise EmbeddingProviderError(
                    "embedding provider returned the wrong query vector count"
                )
            rankings.append(self.store.search_lexical(search_query, limit=pool_limit))
            rankings.append(
                self.store.search_vector(
                    query_vectors[0],
                    model_name=self.embedding_provider.model_name,
                    limit=pool_limit,
                )
            )

        fused_pool = reciprocal_rank_fusion(
            rankings, limit=self.rerank.candidate_k if use_rerank else limit
        )
        if not use_rerank:
            return RetrievalPipelineResult(
                results=fused_pool[:limit],
                trace=RetrievalPipelineTrace(
                    mode=mode, rewrite=rewrite_trace, rerank=RerankTrace()
                ),
            )

        reranked, rerank_trace = rerank_fused_results(
            query,
            fused_pool,
            top_k=limit,
            scorer=self.rerank_scorer,
            model_name=self.rerank.model,
            prefer_neural=self.rerank.prefer_neural,
        )
        return RetrievalPipelineResult(
            results=reranked,
            trace=RetrievalPipelineTrace(
                mode=mode, rewrite=rewrite_trace, rerank=rerank_trace
            ),
        )
