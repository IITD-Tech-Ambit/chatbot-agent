"""Embedding service client with Redis cache. Ports embeddingService.js."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import httpx
import redis.asyncio as aioredis

from agent import metrics as _metrics

logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        redis_client: aioredis.Redis,
        timeout_ms: int = 10_000,
        cache_ttl: int = 86400,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._redis = redis_client
        self._timeout = timeout_ms / 1000.0
        self._cache_ttl = cache_ttl

    def _cache_key(self, text: str) -> str:
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"embed:{h}"

    async def embed_query(self, text: str) -> list[float]:
        key = self._cache_key(text)

        # Try cache
        try:
            cached = await self._redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning("Redis cache read failed: %s", e)

        # Call embedding service
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/embed",
                json={"texts": [text]},
            )
            resp.raise_for_status()
            embedding = resp.json()["embeddings"][0]

        # Write cache
        try:
            await self._redis.setex(key, self._cache_ttl, json.dumps(embedding))
        except Exception as e:
            logger.warning("Redis cache write failed: %s", e)

        return embedding

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        """Call the BGE cross-encoder reranker.

        Returns (original_index, score) pairs sorted by score descending.
        Returns [] when the reranker is unavailable or errors out.
        """
        if not documents:
            _metrics.CHATBOT_RERANK_REQUESTS_TOTAL.labels(outcome="skipped").inc()
            return []
        payload: dict[str, Any] = {"query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        t_start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self._base_url}/rerank", json=payload)
                if resp.status_code == 404:
                    logger.debug("Reranker endpoint not available (404)")
                    _metrics.CHATBOT_RERANK_REQUESTS_TOTAL.labels(outcome="skipped").inc()
                    return []
                resp.raise_for_status()
                results = resp.json().get("results", [])
                _metrics.CHATBOT_RERANK_REQUESTS_TOTAL.labels(outcome="success").inc()
                _metrics.CHATBOT_RERANK_DURATION_SECONDS.observe(time.perf_counter() - t_start)
                return [(r["index"], r["score"]) for r in results]
        except Exception as exc:
            _metrics.CHATBOT_RERANK_REQUESTS_TOTAL.labels(outcome="error").inc()
            _metrics.CHATBOT_RERANK_DURATION_SECONDS.observe(time.perf_counter() - t_start)
            logger.debug("Rerank failed: %s", exc)
            return []

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                return resp.is_success
        except Exception:
            return False
