"""Tool dependency registry.

Tools run inside LangGraph's ToolNode, which doesn't have direct access to the
FastAPI app state. This module holds references set at startup so tools can
retrieve the shared repositories, retriever, and config.

Auto-discovery: every BaseTool instance exported from agent/tools/*.py
(excluding _-prefixed modules) is collected at startup. No manual list needed.
"""

from __future__ import annotations

import importlib
import pkgutil
import logging
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from agent.config import Settings
    from agent.rag.retriever import Retriever
    from agent.repositories.faculty_repo import FacultyRepository
    from agent.repositories.research_repo import ResearchRepository

logger = logging.getLogger(__name__)

_retriever: Retriever | None = None
_faculty_repo: FacultyRepository | None = None
_research_repo: ResearchRepository | None = None
_config: Settings | None = None


def init(
    *,
    retriever: Retriever,
    faculty_repo: FacultyRepository,
    research_repo: ResearchRepository,
    config: Settings,
) -> None:
    global _retriever, _faculty_repo, _research_repo, _config
    _retriever = retriever
    _faculty_repo = faculty_repo
    _research_repo = research_repo
    _config = config


def get_retriever() -> Retriever:
    assert _retriever is not None, "Tool registry not initialized"
    return _retriever


def get_faculty_repo() -> FacultyRepository:
    assert _faculty_repo is not None, "Tool registry not initialized"
    return _faculty_repo


def get_research_repo() -> ResearchRepository:
    assert _research_repo is not None, "Tool registry not initialized"
    return _research_repo


def get_config() -> Settings:
    assert _config is not None, "Tool registry not initialized"
    return _config


# ── Auto-discovery: collect all BaseTool instances from tool modules ──

def all_tools() -> list[BaseTool]:
    import agent.tools as _tools_pkg

    tools: list[BaseTool] = []
    seen_names: set[str] = set()

    for _, modname, _ in pkgutil.iter_modules(_tools_pkg.__path__):
        if modname.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"agent.tools.{modname}")
        except Exception as exc:
            logger.warning("Could not import tool module '%s': %s", modname, exc)
            continue
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, BaseTool) and obj.name not in seen_names:
                tools.append(obj)
                seen_names.add(obj.name)
                logger.debug("Registered tool: %s (from %s)", obj.name, modname)

    logger.info("Loaded %d tools: %s", len(tools), [t.name for t in tools])
    return tools


TOOL_STATUS: dict[str, str] = {
    "search_papers": "Searching publications...",
    "find_faculty_for_topic": "Finding relevant faculty...",
    "get_faculty_profile": "Looking up faculty profile...",
    "get_publication_stats": "Computing statistics...",
    "compare_faculty": "Comparing faculty...",
    "find_similar_papers": "Finding similar papers...",
    "get_research_trends": "Analyzing research trends...",
    "get_department_profile": "Loading department overview...",
    "list_departments": "Listing departments...",
    "find_faculty_by_expertise": "Searching by expertise...",
    "find_interdisciplinary_papers": "Finding interdisciplinary research...",
    "get_top_faculty": "Ranking faculty...",
}


def status_for(tool_name: str) -> str:
    return TOOL_STATUS.get(tool_name, "Thinking...")
