"""Local knowledge-base primitives for retrieval-augmented agent runs."""

from nanobot.knowledge.chunking import ChunkingConfig, chunk_document
from nanobot.knowledge.citations import KnowledgeCitation, render_retrieval_context
from nanobot.knowledge.embeddings import EmbeddingProvider, OpenAICompatibleEmbeddingProvider
from nanobot.knowledge.fusion import FusedSearchResult, reciprocal_rank_fusion
from nanobot.knowledge.models import KnowledgeChunk, KnowledgeSearchResult, SourceDocument
from nanobot.knowledge.retrieval import HybridKnowledgeRetriever
from nanobot.knowledge.store import KnowledgeStore

__all__ = [
    "ChunkingConfig",
    "EmbeddingProvider",
    "FusedSearchResult",
    "HybridKnowledgeRetriever",
    "KnowledgeChunk",
    "KnowledgeCitation",
    "KnowledgeSearchResult",
    "KnowledgeStore",
    "OpenAICompatibleEmbeddingProvider",
    "SourceDocument",
    "chunk_document",
    "reciprocal_rank_fusion",
    "render_retrieval_context",
]
