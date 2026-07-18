"""Per-tool UI/budget metadata (OCP: declare beside the tool, not in central switches)."""

from __future__ import annotations

from langchain_core.tools import BaseTool

_META_THINKING = "thinking_label"
_META_TOKEN_CAP = "token_cap"


def annotate_tool(tool: BaseTool, *, thinking_label: str, token_cap: int) -> BaseTool:
    meta = dict(tool.metadata or {})
    meta[_META_THINKING] = thinking_label
    meta[_META_TOKEN_CAP] = token_cap
    tool.metadata = meta
    return tool


def thinking_label_for(tool_name: str, tools: list[BaseTool] | None = None) -> str:
    if tools:
        for t in tools:
            if t.name == tool_name:
                label = (t.metadata or {}).get(_META_THINKING)
                if label:
                    return str(label)
    return "Processing your query"


def token_caps_map(tools: list[BaseTool], default: int) -> dict[str, int]:
    caps: dict[str, int] = {}
    for t in tools:
        cap = (t.metadata or {}).get(_META_TOKEN_CAP)
        caps[t.name] = cap if isinstance(cap, int) and cap > 0 else default
    return caps
