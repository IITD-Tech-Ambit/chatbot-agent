"""Redis-backed cache for resolved IPC code meanings."""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RedisIpcCache:
    def __init__(self, redis_client: aioredis.Redis, ttl: int) -> None:
        self._redis = redis_client
        self._ttl = ttl

    def _key(self, code: str) -> str:
        return f"ipc:def:{code.strip().upper()}"

    async def get(self, code: str) -> str | None:
        try:
            return await self._redis.get(self._key(code))
        except Exception as exc:
            logger.warning("IPC cache read failed: %s", exc)
            return None

    async def set(self, code: str, meaning: str) -> None:
        try:
            await self._redis.setex(self._key(code), self._ttl, meaning)
        except Exception as exc:
            logger.warning("IPC cache write failed: %s", exc)
