"""Health check endpoint."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "mongodb": False,
        "opensearch": False,
        "redis": False,
        "embedding": False,
    }

    try:
        db = request.app.state.db
        await db.command("ping")
        checks["mongodb"] = True
    except Exception:
        pass

    try:
        os_client = request.app.state.opensearch
        info = await os_client.cluster.health()
        checks["opensearch"] = info.get("status") in ("green", "yellow")
    except Exception:
        pass

    try:
        redis_client = request.app.state.redis
        await redis_client.ping()
        checks["redis"] = True
    except Exception:
        pass

    try:
        embed = request.app.state.embedding_client
        checks["embedding"] = await embed.health()
    except Exception:
        pass

    healthy = all(checks.values())
    return {
        "status": "healthy" if healthy else "degraded",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
