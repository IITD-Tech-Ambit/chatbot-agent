"""HTTP client for the search-api taxonomy endpoints (the Research Areas page).

Wraps `GET /api/v1/taxonomy/*` — the SAME cached endpoints the Research Areas
browser uses, so the bot's theme/domain/faculty counts match the page exactly.
Called directly over HTTP at SEARCH_API_URL; `trust_env=False` keeps these
internal mesh calls off the campus LLM proxy.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class TaxonomyClient:
    def __init__(self, base_url: str, timeout_s: float = 20.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        clean = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
        url = f"{self._base_url}/api/v1{path}"
        async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as client:
            resp = await client.get(url, params=clean)
            resp.raise_for_status()
            return resp.json()

    async def departments(self) -> list[dict[str, Any]]:
        data = await self._get("/taxonomy/departments")
        return data.get("departments", []) or []

    async def themes(self, department: str | None = None) -> list[dict[str, Any]]:
        data = await self._get("/taxonomy/themes", {"department": department})
        return data.get("themes", []) or []

    async def domains(self, theme: str | None = None, department: str | None = None) -> list[dict[str, Any]]:
        data = await self._get("/taxonomy/domains", {"theme": theme, "department": department})
        return data.get("domains", []) or []

    async def counts(
        self, theme: str | None = None, domain: str | None = None, department: str | None = None
    ) -> dict[str, Any]:
        return await self._get(
            "/taxonomy/counts", {"theme": theme, "domain": domain, "department": department}
        )

    async def faculty(
        self,
        theme: str | None = None,
        domain: str | None = None,
        department: str | None = None,
        page: int = 1,
        per_page: int = 50,
    ) -> dict[str, Any]:
        return await self._get(
            "/taxonomy/faculty",
            {"theme": theme, "domain": domain, "department": department,
             "page": page, "per_page": per_page},
        )
