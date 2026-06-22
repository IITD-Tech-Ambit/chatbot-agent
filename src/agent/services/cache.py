"""Redis LLM-response cache. Short TTL to absorb burst traffic."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class LLMCache:
    def __init__(self, redis_client: aioredis.Redis, ttl: int = 90) -> None:
        self._redis = redis_client
        self._ttl = ttl

    def _key(self, query: str) -> str:
        normalized = query.strip().lower()
        h = hashlib.sha256(normalized.encode()).hexdigest()
        return f"llm:v1:{h}"

    async def get(self, query: str) -> dict[str, Any] | None:
        if self._ttl <= 0:
            return None
        try:
            raw = await self._redis.get(self._key(query))
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.warning("LLM cache read failed: %s", e)
        return None

    async def set(self, query: str, response: dict[str, Any]) -> None:
        if self._ttl <= 0:
            return
        try:
            await self._redis.setex(
                self._key(query),
                self._ttl,
                json.dumps(response, default=str),
            )
        except Exception as e:
            logger.warning("LLM cache write failed: %s", e)
