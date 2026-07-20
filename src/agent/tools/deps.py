"""Shared dependencies injected into tool factories at startup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.repositories.protocols import (
    IFacultyRepository,
    IIpRepository,
    IResearchRepository,
    ITaxonomyRepository,
)

if TYPE_CHECKING:
    from agent.config import Settings
    from agent.rag.ip_retriever import IpRetriever
    from agent.rag.retriever import Retriever
    from agent.services.ipc.service import IpcClassificationService
    from agent.transports.protocols import FacultySearchClient


@dataclass(frozen=True)
class ToolDeps:
    retriever: Retriever
    faculty_repo: IFacultyRepository
    research_repo: IResearchRepository
    config: Settings
    search_client: FacultySearchClient | None = None
    ip_repo: IIpRepository | None = None
    ip_retriever: IpRetriever | None = None
    ipc_service: IpcClassificationService | None = None
    taxonomy_repo: ITaxonomyRepository | None = None
