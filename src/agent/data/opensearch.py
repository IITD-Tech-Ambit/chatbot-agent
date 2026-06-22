"""Async OpenSearch client wrapper."""

from __future__ import annotations

import logging
from typing import Any

from opensearchpy._async.client import AsyncOpenSearch

logger = logging.getLogger(__name__)

_client: AsyncOpenSearch | None = None


async def connect(
    node: str,
    user: str = "",
    password: str = "",
    verify_certs: bool = False,
    use_ssl: bool = False,
) -> AsyncOpenSearch:
    global _client
    auth = (user, password) if user else None
    _use_ssl = use_ssl or node.startswith("https")
    _verify = verify_certs if _use_ssl else False
    _client = AsyncOpenSearch(
        hosts=[node],
        http_auth=auth,
        use_ssl=_use_ssl,
        verify_certs=_verify,
        ssl_show_warn=False,
        timeout=30,
    )
    info = await _client.info()
    logger.info("OpenSearch connected: %s", info.get("version", {}).get("number", "?"))
    return _client


def get_client() -> AsyncOpenSearch:
    if _client is None:
        raise RuntimeError("OpenSearch not connected — call connect() first")
    return _client


async def close() -> None:
    global _client
    if _client:
        await _client.close()
        _client = None
        logger.info("OpenSearch connection closed")
