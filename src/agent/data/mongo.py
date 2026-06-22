"""Async MongoDB client via Motor."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect(uri: str) -> AsyncIOMotorDatabase:
    global _client, _db
    _client = AsyncIOMotorClient(
        uri,
        maxPoolSize=10,
        serverSelectionTimeoutMS=5000,
        socketTimeoutMS=45000,
    )
    db_name = uri.rsplit("/", 1)[-1].split("?")[0] or "research_db"
    _db = _client[db_name]
    # Force a connection test
    await _client.admin.command("ping")
    logger.info("MongoDB connected to %s", db_name)
    return _db


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("MongoDB not connected — call connect() first")
    return _db


async def close() -> None:
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed")
