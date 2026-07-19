"""lookup_ipc_classification tool — resolve IPC codes ↔ meanings for IP queries."""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent.tools.deps import ToolDeps
from agent.tools.meta import annotate_tool


class LookupIpcArgs(BaseModel):
    code: Optional[str] = Field(default=None, description='IPC classification code to explain, e.g. "A61K" or "A61K 38/00"')
    topic: Optional[str] = Field(default=None, description='Research topic to map to candidate IPC prefixes, e.g. "drug delivery"')


def build_tool(deps: ToolDeps) -> BaseTool:
    ipc_service = deps.ipc_service

    @tool(args_schema=LookupIpcArgs)
    async def lookup_ipc_classification(code: str | None = None, topic: str | None = None) -> str:
        """Resolve IPC patent classifications. Give a code to explain what it means, or a topic to
        get candidate IPC prefixes; then call search_ips/get_ip_stats with classification_prefix to
        find matching patents."""
        if ipc_service is None:
            return json.dumps({"error": "IPC lookup is not available"})
        if not code and not topic:
            return json.dumps({"error": "Provide an IPC code or a topic."})

        result: dict = {}
        if code:
            result["code_lookup"] = await ipc_service.resolve_code(code)
        if topic:
            result["topic"] = topic
            result["candidate_prefixes"] = ipc_service.suggest_prefixes(topic)

        return json.dumps(result, default=str)

    return annotate_tool(
        lookup_ipc_classification,
        thinking_label="Looking up IPC classification",
        token_cap=deps.config.TOKEN_CAP_IPC_LOOKUP,
    )
