"""Orchestrates IPC code resolution over injected collaborators.

Resolution order for a code lookup: bundled static table for coarse
(section/class/subclass) levels, then Redis cache, then a WIPO fetch for
group-level detail, caching successful fetches. Topic→prefix suggestions come
from the offline static table only.
"""

from __future__ import annotations

import re
from typing import Any

from agent.services.ipc.protocols import IpcCache, IpcStaticTable, WipoIpcClient


def _normalize(code: str) -> str:
    return re.sub(r"\s+", " ", (code or "").strip().upper())


class IpcClassificationService:
    def __init__(
        self,
        static_table: IpcStaticTable,
        cache: IpcCache,
        wipo_client: WipoIpcClient | None = None,
    ) -> None:
        self._static = static_table
        self._cache = cache
        self._wipo = wipo_client

    async def resolve_code(self, code: str) -> dict[str, Any]:
        norm = _normalize(code)
        static = self._static.describe(norm)
        is_group_level = "/" in norm

        if static and not is_group_level:
            return self._result(norm, static["meaning"], "static_table", static)

        cached = await self._cache.get(norm)
        if cached:
            return self._result(norm, cached, "cache", static)

        if self._wipo is not None:
            fetched = await self._wipo.fetch_definition(norm)
            if fetched:
                await self._cache.set(norm, fetched)
                return self._result(norm, fetched, "wipo", static)

        if static:
            return self._result(norm, static["meaning"], "static_table", static)

        return self._result(norm, None, "unknown", None)

    def suggest_prefixes(self, topic: str, limit: int = 8) -> list[dict[str, Any]]:
        return self._static.suggest(topic, limit=limit)

    @staticmethod
    def _result(
        code: str, meaning: str | None, source: str, static: dict[str, Any] | None
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": code,
            "meaning": meaning,
            "source": source,
            "found": meaning is not None,
        }
        if static:
            result["matched_prefix"] = static.get("matched_prefix")
            result["level"] = static.get("level")
            result["breakdown"] = static.get("breakdown")
        return result
