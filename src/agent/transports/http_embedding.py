"""HTTP adapter for EmbeddingTransport (local dev without Envoy)."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class HttpEmbeddingTransport:
    def __init__(self, base_url: str, timeout_ms: int = 10_000) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_ms / 1000.0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}/embed", json={"texts": texts})
            resp.raise_for_status()
            return resp.json()["embeddings"]

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        payload: dict = {"query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}/rerank", json=payload)
            if resp.status_code == 404:
                logger.debug("Reranker endpoint not available (404)")
                return []
            resp.raise_for_status()
            return [(r["index"], r["score"]) for r in resp.json().get("results", [])]

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                return resp.is_success
        except Exception:
            return False
