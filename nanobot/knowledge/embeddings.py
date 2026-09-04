"""Embedding interfaces kept independent from chat-provider implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from openai import AsyncOpenAI, OpenAIError


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding backend cannot complete a request."""


class EmbeddingProvider(Protocol):
    """Minimal async contract required by the knowledge index."""

    @property
    def model_name(self) -> str: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbeddingProvider:
    """Generate embeddings through an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 64,
    ) -> None:
        if not model.strip():
            raise ValueError("embedding model must not be empty")
        if batch_size < 1 or batch_size > 2_048:
            raise ValueError("batch_size must be between 1 and 2048")
        if dimensions is not None and dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("embedding inputs must be non-empty strings")

        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), self._batch_size):
                batch = list(texts[start : start + self._batch_size])
                if self._dimensions is None:
                    response = await self._client.embeddings.create(
                        input=batch,
                        model=self._model,
                    )
                else:
                    response = await self._client.embeddings.create(
                        input=batch,
                        model=self._model,
                        dimensions=self._dimensions,
                    )
                ordered = sorted(response.data, key=lambda item: item.index)
                if len(ordered) != len(batch):
                    raise EmbeddingProviderError(
                        "embedding endpoint returned an incomplete batch"
                    )
                vectors.extend([list(item.embedding) for item in ordered])
        except OpenAIError as exc:
            raise EmbeddingProviderError(
                f"embedding request failed: {type(exc).__name__}"
            ) from exc
        return vectors
