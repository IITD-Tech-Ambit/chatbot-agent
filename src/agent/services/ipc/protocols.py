"""Narrow contracts for the IPC classification collaborators (DIP + ISP).

The orchestrating service depends on these protocols, not concrete Redis/HTTP
clients, so each collaborator can be swapped or faked in tests.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IpcStaticTable(Protocol):
    def describe(self, code: str) -> dict[str, Any] | None: ...
    def suggest(self, topic: str, limit: int = 8) -> list[dict[str, Any]]: ...


@runtime_checkable
class IpcCache(Protocol):
    async def get(self, code: str) -> str | None: ...
    async def set(self, code: str, meaning: str) -> None: ...


@runtime_checkable
class WipoIpcClient(Protocol):
    async def fetch_definition(self, code: str) -> str | None: ...
