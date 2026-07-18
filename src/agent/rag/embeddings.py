"""Embedding client: Redis-cached facade over an injected EmbeddingTransport.

The wire protocol (gRPC via Envoy in production, HTTP in local dev) lives in
agent.transports; this class owns only caching and metrics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

import redis.asyncio as aioredis

from agent import metrics as _metrics
from agent.transports.protocols import EmbeddingTransport

logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(
        self,
        transport: EmbeddingTransport,
        redis_client: aioredis.Redis,
        cache_ttl: int = 86400,
    ) -> None:
        self._transport = transport
        self._redis = redis_client
        self._cache_ttl = cache_ttl

    def _cache_key(self, text: str) -> str:
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"embed:{h}"

    async def embed_query(self, text: str) -> list[float]:
        key = self._cache_key(text)

        try:
            cached = await self._redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning("Redis cache read failed: %s", e)

        embedding = (await self._transport.embed([text]))[0]

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
        t_start = time.perf_counter()
        try:
            results = await self._transport.rerank(query, documents, top_n)
            outcome = "success" if results else "skipped"
            _metrics.CHATBOT_RERANK_REQUESTS_TOTAL.labels(outcome=outcome).inc()
            _metrics.CHATBOT_RERANK_DURATION_SECONDS.observe(time.perf_counter() - t_start)
            return results
        except Exception as exc:
            _metrics.CHATBOT_RERANK_REQUESTS_TOTAL.labels(outcome="error").inc()
            _metrics.CHATBOT_RERANK_DURATION_SECONDS.observe(time.perf_counter() - t_start)
            logger.debug("Rerank failed: %s", exc)
            return []

    async def health(self) -> bool:
        try:
            return await self._transport.health()
        except Exception:
            return False
