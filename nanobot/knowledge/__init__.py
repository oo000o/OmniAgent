"""Local knowledge-base primitives for retrieval-augmented agent runs."""

from nanobot.knowledge.chunking import ChunkingConfig, chunk_document
from nanobot.knowledge.citations import KnowledgeCitation, render_retrieval_context
from nanobot.knowledge.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
    OpenAICompatibleEmbeddingProvider,
)
from nanobot.knowledge.fusion import FusedSearchResult, reciprocal_rank_fusion
from nanobot.knowledge.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from nanobot.knowledge.models import KnowledgeChunk, KnowledgeSearchResult, SourceDocument
from nanobot.knowledge.retrieval import HybridKnowledgeRetriever
from nanobot.knowledge.store import KnowledgeStore

__all__ = [
    "ChunkingConfig",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "FusedSearchResult",
    "HybridKnowledgeRetriever",
    "KnowledgeChunk",
    "KnowledgeCitation",
    "KnowledgeSearchResult",
    "KnowledgeStore",
    "OpenAICompatibleEmbeddingProvider",
    "SourceDocument",
    "chunk_document",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank_fusion",
    "reciprocal_rank",
    "render_retrieval_context",
]
