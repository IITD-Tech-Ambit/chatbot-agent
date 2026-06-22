"""Embedding service client with Redis cache. Ports embeddingService.js."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import httpx
import redis.asyncio as aioredis

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

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                return resp.is_success
        except Exception:
            return False
