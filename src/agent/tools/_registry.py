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


def build_tools(deps: ToolDeps) -> list[BaseTool]:
    """Discover ``build_tool`` factories under agent.tools and instantiate them."""
    import agent.tools as tools_pkg

    tools: list[BaseTool] = []
    seen: set[str] = set()

    for _, modname, _ in pkgutil.iter_modules(tools_pkg.__path__):
        if modname.startswith("_") or modname in {"deps", "meta"}:
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
