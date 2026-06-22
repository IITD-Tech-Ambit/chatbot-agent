"""Redis fixed-window rate limiter, mirroring the Node INCR+EXPIRE pattern. Fails open."""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, redis_client: aioredis.Redis, window_sec: int = 60, max_requests: int = 20) -> None:
        self._redis = redis_client
        self._window = window_sec
        self._max = max_requests

    async def is_allowed(self, ip: str) -> bool:
        try:
            key = f"chat:rl:{ip}"
            count = await self._redis.incr(key)
            if count == 1:
                await self._redis.expire(key, self._window)
            return count <= self._max
        except Exception as e:
            logger.warning("Rate-limit check failed (allowing): %s", e)
            return True
