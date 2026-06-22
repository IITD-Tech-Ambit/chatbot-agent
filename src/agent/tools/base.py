"""Abstract base for new agent tools (SOLID — Single Responsibility).

Existing 7 tools use @tool decorators directly for backward compatibility.
New tools subclass AgentTool for explicit structure and testability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AgentTool(ABC):
    """One tool, one domain, one run() method."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    async def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool. Returns a domain dict; registry JSON-serializes it."""
        ...
