"""Build LangChain tools with injected dependencies.

Each tool module exports ``build_tool(deps) -> BaseTool``. Tools declare
thinking labels / token caps via ``annotate_tool`` so chat/graph code can
extend without editing central switch statements (OCP).
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

from langchain_core.tools import BaseTool

from agent.tools.deps import ToolDeps

logger = logging.getLogger(__name__)

# The bot's capability surface has been narrowed to the Explore advanced-search
# tools plus the chart-producing tools. Only these tool modules are loaded; all
# other tool files remain on disk but are intentionally NOT registered. To bring
# one back, add its module name here.
ENABLED_TOOL_MODULES: frozenset[str] = frozenset({
    # Explore advanced search (papers + IP), wrapping the search-api
    "search_research",
    "search_ip",
    # Research Areas (classification taxonomy), wrapping the search-api.
    # Structural/naming questions are answered from the system-prompt reference;
    # this tool covers the dynamic experts + area counts.
    "experts_by_research_area",
    # Chart-producing tools (see agent.api.chart_builder._CHART_BUILDERS)
    "research_trends",       # get_research_trends
    "compare_faculty",       # compare_faculty
    "department_profile",    # get_department_profile
    "publication_stats",     # get_publication_stats
    "get_ip_stats",          # get_ip_stats
})


def build_tools(deps: ToolDeps) -> list[BaseTool]:
    """Discover ``build_tool`` factories under agent.tools and instantiate them.

    Restricted to ENABLED_TOOL_MODULES — the Explore search tools + chart tools.
    """
    import agent.tools as tools_pkg

    tools: list[BaseTool] = []
    seen: set[str] = set()

    for _, modname, _ in pkgutil.iter_modules(tools_pkg.__path__):
        if modname.startswith("_") or modname in {"deps", "meta"}:
            continue
        if modname not in ENABLED_TOOL_MODULES:
            continue
        try:
            mod = importlib.import_module(f"agent.tools.{modname}")
        except Exception as exc:
            logger.warning("Could not import tool module '%s': %s", modname, exc)
            continue
        factory = getattr(mod, "build_tool", None)
        if factory is None:
            continue
        try:
            tool = factory(deps)
        except Exception as exc:
            logger.warning("Could not build tool from '%s': %s", modname, exc)
            continue
        if not isinstance(tool, BaseTool) or tool.name in seen:
            continue
        tools.append(tool)
        seen.add(tool.name)
        logger.debug("Registered tool: %s (from %s)", tool.name, modname)

    logger.info("Loaded %d tools: %s", len(tools), [t.name for t in tools])
    return tools
