"""HTTP client for the search-api Explore endpoints (advanced paper + IP search).

Mirrors the Explore page: it calls the SAME `POST /api/v1/search` and
`POST /api/v1/ip/search` the frontend uses, so the bot gets the identical
hybrid BM25 + semantic engine (BM25 pre-check gate, cross-encoder rerank, fuzzy
fallback, facets, related-faculty aggregation) instead of reimplementing search.

Called directly over HTTP at SEARCH_API_URL (search-api serves REST on :3001
even when the east-west mesh transport is gRPC). `trust_env=False` keeps these
internal mesh calls off the campus LLM proxy.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ResearchSearchClient:
    def __init__(self, base_url: str, timeout_s: float = 25.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            return resp.json()

    async def search(self, body: dict[str, Any]) -> dict[str, Any]:
        """Advanced research-paper search (POST /api/v1/search)."""
        return await self._post("/api/v1/search", body)

    async def ip_search(self, body: dict[str, Any]) -> dict[str, Any]:
        """Advanced IP/patent search (POST /api/v1/ip/search)."""
        return await self._post("/api/v1/ip/search", body)

    async def faculty_for_query(
        self,
        query: str,
        *,
        mode: str = "advanced",
        search_in: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The Explore page's People sidebar (GET /api/v1/search/faculty-for-query).

        Aggregates faculty across the ENTIRE matching result set (not just the
        returned page), grouped by department with per-faculty paper counts.
        The same query/mode/search_in/filters must be passed as the paper search,
        otherwise the counts would describe a different corpus.
        """
        params: dict[str, Any] = {"query": query, "mode": mode}
        if search_in:
            params["search_in"] = ",".join(search_in)
        if filters:
            params["filters"] = json.dumps(filters)

        url = f"{self._base_url}/api/v1/search/faculty-for-query"
        async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
