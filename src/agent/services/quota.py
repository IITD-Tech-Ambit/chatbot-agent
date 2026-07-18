"""Per-user daily chat quota (IST calendar day), backed by Redis.

QuotaStore is the port; RedisQuotaStore the adapter. Key layout:
    chat:quota:{user_id}:{YYYY-MM-DD}   (date in Asia/Kolkata)
INCR + EXPIREAT next IST midnight, so counters reset with the local day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

import redis.asyncio as aioredis

from agent.config import settings

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")


def is_quota_exempt(kerberos: str, category: str) -> bool:
    """Only categories listed in CHAT_QUOTA_LIMITED_CATEGORIES (env,
    comma-separated, case-insensitive substring match — e.g. "student"
    matches "UG Student"/"PG Student") are subject to the daily quota;
    every other category (faculty, staff, anything unlisted) is unlimited.
    Whitelisted kerberos IDs are always unlimited regardless of category.
    An empty/unrecognized category (e.g. missing header) is NOT exempt —
    fail toward applying the limit, not toward unlimited chats."""
    whitelist = {
        k.strip().lower()
        for k in settings.CHAT_QUOTA_WHITELIST_KERBEROS.split(",")
        if k.strip()
    }
    if kerberos.strip().lower() in whitelist:
        return True

    category = category.strip().lower()
    if not category:
        return False

    limited_categories = {
        c.strip().lower()
        for c in settings.CHAT_QUOTA_LIMITED_CATEGORIES.split(",")
        if c.strip()
    }
    is_limited = any(c in category for c in limited_categories)
    return not is_limited


@dataclass(frozen=True)
class QuotaState:
    allowed: bool
    limit: int
    used: int

    @property
    def remaining(self) -> int:
        return max(self.limit - self.used, 0)


class QuotaStore(Protocol):
    async def consume(self, user_id: str) -> QuotaState:
        """Count one message against today's quota and report the new state."""
        ...

    async def peek(self, user_id: str) -> QuotaState:
        """Report today's state without consuming."""
        ...


class RedisQuotaStore:
    def __init__(self, redis_client: aioredis.Redis, daily_limit: int = 5) -> None:
        self._redis = redis_client
        self._limit = daily_limit

    def _key(self, user_id: str, now: datetime) -> str:
        return f"chat:quota:{user_id}:{now.strftime('%Y-%m-%d')}"

    @staticmethod
    def _next_midnight_ist(now: datetime) -> datetime:
        return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    async def consume(self, user_id: str) -> QuotaState:
        now = datetime.now(IST)
        key = self._key(user_id, now)
        try:
            used = await self._redis.incr(key)
            if used == 1:
                await self._redis.expireat(key, int(self._next_midnight_ist(now).timestamp()))
        except Exception as exc:
            # Redis is also the LLM/embedding cache backbone; if it is down the
            # chat is degraded anyway. Fail open so quota never adds an outage.
            logger.warning("Quota consume failed (allowing): %s", exc)
            return QuotaState(allowed=True, limit=self._limit, used=0)

        if used > self._limit:
            return QuotaState(allowed=False, limit=self._limit, used=self._limit)
        return QuotaState(allowed=True, limit=self._limit, used=int(used))

    async def peek(self, user_id: str) -> QuotaState:
        now = datetime.now(IST)
        try:
            raw = await self._redis.get(self._key(user_id, now))
            used = min(int(raw or 0), self._limit)
        except Exception as exc:
            logger.warning("Quota peek failed (reporting empty): %s", exc)
            used = 0
        return QuotaState(allowed=used < self._limit, limit=self._limit, used=used)
