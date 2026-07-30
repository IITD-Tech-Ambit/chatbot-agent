"""HTTP client for the main backend's Directory endpoints.

Wraps `GET /api/directory/grouped` — the SAME endpoints the Directory page uses
— so the bot lists a unit's faculty exactly as that page shows them. Called
directly at BACKEND_API_URL; `trust_env=False` keeps these internal calls off
the campus LLM proxy.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


class DirectoryClient:
    def __init__(self, base_url: str, timeout_s: float = 20.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        clean = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
        url = f"{self._base_url}/api/directory{path}"
        async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as client:
            resp = await client.get(url, params=clean)
            resp.raise_for_status()
            return resp.json()

    async def unit_groups(self, category: str) -> list[dict[str, Any]]:
        """The units (departments | centres | schools) with their ids + counts —
        exactly the list the Directory page's category tab shows."""
        data = await self._get("/grouped", {"category": category, "summaryOnly": "true"})
        body = data.get("data", data) or {}
        return body.get("departments", []) or []

    async def unit_faculty(self, unit_id: str, category: str) -> list[dict[str, Any]]:
        """The faculty within one unit, in the Directory page's order (h-index desc)."""
        data = await self._get(f"/grouped/{unit_id}/faculties", {"category": category})
        body = data.get("data", data) or {}
        return body.get("faculties", []) or []

    async def search_faculty(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """The Directory search box (GET /search?q=). Ranked results; the first is
        what a user would land on for the same query."""
        data = await self._get("/search", {"q": query, "limit": limit})
        body = data.get("data", data) or {}
        return body.get("faculties", []) or []

    async def faculty_profile(self, kerberos: str) -> dict[str, Any]:
        """One faculty member's profile core (GET /faculty/:kerberos/profile) —
        the same record the /faculty/<kerberos> page header shows."""
        kerb = quote(kerberos, safe="")
        data = await self._get(f"/faculty/{kerb}/profile")
        return data.get("data", data) or {}

    async def research_summary(self, kerberos: str, year_limit: int = 20) -> dict[str, Any]:
        """A faculty member's publication summary + per-year timeline
        (GET /faculty/:kerberos/research-summary) — the source of the profile
        page's Publications section, newest year first."""
        data = await self._get(
            f"/faculty/{quote(kerberos, safe='')}/research-summary",
            {"yearLimit": year_limit},
        )
        return data.get("data", data) or {}
