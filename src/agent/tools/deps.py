"""Shared dependencies injected into tool factories at startup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.repositories.protocols import IFacultyRepository, IResearchRepository

if TYPE_CHECKING:
    from agent.config import Settings
    from agent.rag.retriever import Retriever
    from agent.transports.protocols import FacultySearchClient


@dataclass(frozen=True)
class ToolDeps:
    retriever: Retriever
    faculty_repo: IFacultyRepository
    research_repo: IResearchRepository
    config: Settings
    search_client: FacultySearchClient | None = None
