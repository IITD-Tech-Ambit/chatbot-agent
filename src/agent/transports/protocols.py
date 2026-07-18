"""Transport ports (Protocol classes). Business code depends on these shapes,
never on httpx / grpc directly — adapters live next to this module."""

from __future__ import annotations

from typing import Any, Protocol


class EmbeddingTransport(Protocol):
    """Raw wire access to the embedding service (no caching here)."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        """(original_index, score) pairs sorted by score descending."""
        ...

    async def health(self) -> bool: ...


class FacultySearchClient(Protocol):
    """faculty-for-query aggregation served by search-api."""

    async def faculty_for_query(self, query: str) -> dict[str, Any]:
        """Returns {"departments": [...], "total_matching_papers": int}."""
        ...
