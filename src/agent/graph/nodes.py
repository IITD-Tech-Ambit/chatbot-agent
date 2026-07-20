"""Graph nodes: agent (tool-selection), budget guard, answer stream.

Node callables are built via factories so LLMs/tools are injected at startup
rather than looked up from module globals.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Awaitable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from agent.config import settings
from agent.graph.state import AgentState
from agent.tools.meta import token_caps_map

AgentNode = Callable[[AgentState, RunnableConfig | None], Awaitable[dict[str, Any]]]


def make_agent_node(tool_llm: BaseChatModel, tools: list[BaseTool]) -> AgentNode:
    async def agent_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        """Invoke the tool-selection LLM. No forced tool injection — empty
        tool_calls routes to the answer node."""
        rounds = state.get("tool_rounds", 0)

        if rounds >= settings.MAX_TOOL_ROUNDS:
            return {"messages": [AIMessage(content="")]}

        llm_with_tools = tool_llm.bind_tools(tools)
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    return agent_node


def make_answer_node(
    answer_llm: BaseChatModel,
    tools: list[BaseTool] | None = None,
) -> AgentNode:
    caps = token_caps_map(tools or [], settings.TOKEN_CAP_DEFAULT)

    async def answer_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        """Generate the final grounded answer. Budget-guard truncates context first."""
        messages = _enforce_context_budget(state["messages"], caps)
        response = await answer_llm.ainvoke(messages)
        return {"messages": [response]}

    return answer_node


def route_after_agent(state: AgentState) -> str:
    """Conditional edge: go to tools if tool_calls present."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "answer"


def _enforce_context_budget(messages: list, caps: dict[str, int]) -> list:
    """Truncate tool result content to stay within budget."""
    available = settings.CONTEXT_ANSWER_RESERVE * 4

    total = 0
    result = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content or ""
            tool_cap = caps.get(msg.name or "", settings.TOKEN_CAP_DEFAULT)
            if len(content) > tool_cap:
                try:
                    data = json.loads(content)
                    truncated = False
                    if isinstance(data, dict):
                        for list_key in ("papers", "ips", "faculty", "similar_papers", "groups", "comparison", "trend", "results", "departments", "themes", "domains", "distribution"):
                            items = data.get(list_key)
                            if isinstance(items, list):
                                while items and len(json.dumps(data, default=str)) > tool_cap:
                                    items.pop()
                                truncated = True
                                break
                    content = json.dumps(data, default=str) if truncated else content[:tool_cap]
                except (json.JSONDecodeError, Exception):
                    content = content[:tool_cap]
            total += len(content)
            if total > available:
                content = '{"truncated": true}'
            result.append(ToolMessage(content=content, tool_call_id=msg.tool_call_id, name=msg.name))
        else:
            content_len = len(getattr(msg, "content", "") or "")
            total += content_len
            result.append(msg)

    return result
