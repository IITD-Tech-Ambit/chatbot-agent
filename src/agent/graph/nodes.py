"""Graph nodes: agent (tool-selection), force-tool fallback, budget guard, answer stream."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agent.config import settings
from agent.graph.state import AgentState

logger = logging.getLogger(__name__)

# Module-level references set at startup by builder.init_node_deps()
_tool_llm = None
_answer_llm = None
_tools: list = []


def init_node_deps(tool_llm, answer_llm, tools: list) -> None:
    global _tool_llm, _answer_llm, _tools
    _tool_llm = tool_llm
    _answer_llm = answer_llm
    _tools = tools


async def agent_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Invoke the tool-selection LLM. If no tool_calls, force a default search_papers.

    tool_rounds is NOT incremented here — it's incremented by route_after_agent
    when it decides to route to tools, so the count reflects actual tool executions.
    """
    rounds = state.get("tool_rounds", 0)

    if rounds >= settings.MAX_TOOL_ROUNDS:
        return {"messages": [AIMessage(content="")]}

    llm_with_tools = _tool_llm.bind_tools(_tools)
    response = await llm_with_tools.ainvoke(state["messages"])

    if not getattr(response, "tool_calls", None):
        user_msg = _extract_user_query(state["messages"])
        logger.info("No tool calls from LLM — forcing search_papers(%s)", user_msg[:80])

        forced = AIMessage(
            content="",
            tool_calls=[{
                "id": "forced_search",
                "name": "search_papers",
                "args": {"query": user_msg},
            }],
        )
        return {"messages": [forced]}

    return {"messages": [response]}


async def answer_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Generate the final grounded answer. Budget-guard truncates context first."""
    messages = _enforce_context_budget(state["messages"])
    response = await _answer_llm.ainvoke(messages)
    return {"messages": [response]}


def extract_sources(state: AgentState) -> list[dict[str, Any]]:
    """Pull paper sources from search_papers tool results in the message history."""
    sources: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for msg in state.get("messages", []):
        if not isinstance(msg, ToolMessage):
            continue
        if msg.name != "search_papers":
            continue
        try:
            data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            for p in data.get("papers", []):
                title = p.get("title", "")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    sources.append({
                        "citation_index": p.get("citation_index"),
                        "title": title,
                        "authors": p.get("authors", []),
                        "year": p.get("year"),
                        "field": p.get("field"),
                        "citations": p.get("citations", 0),
                        "link": p.get("link"),
                        "document_scopus_id": p.get("document_scopus_id"),
                        "document_eid": p.get("document_eid"),
                    })
        except (json.JSONDecodeError, AttributeError):
            pass
    return sources


# ── Routing ──

def route_after_agent(state: AgentState) -> str:
    """Conditional edge: go to tools if tool_calls present AND under round cap."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "answer"


# ── Helpers ──

def _extract_user_query(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _enforce_context_budget(messages: list) -> list:
    """Truncate tool result content to stay within budget."""
    available = settings.CONTEXT_ANSWER_RESERVE * 4

    total = 0
    result = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content or ""
            tool_cap = _tool_cap(msg.name)
            if len(content) > tool_cap:
                try:
                    data = json.loads(content)
                    truncated = False
                    if isinstance(data, dict):
                        for list_key in ("papers", "faculty", "departments", "results"):
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


def _tool_cap(name: str) -> int:
    caps = {
        "search_papers": settings.TOKEN_CAP_SEARCH_PAPERS,
        "get_faculty_profile": settings.TOKEN_CAP_FACULTY_PROFILE,
        "get_publication_stats": settings.TOKEN_CAP_PUBLICATION_STATS,
        "get_department_profile": settings.TOKEN_CAP_DEPARTMENT_PROFILE,
        "list_departments": settings.TOKEN_CAP_LIST_DEPARTMENTS,
        "find_faculty_by_expertise": settings.TOKEN_CAP_FACULTY_EXPERTISE,
        "find_faculty_for_topic": settings.TOKEN_CAP_FACULTY_EXPERTISE,
        "find_interdisciplinary_papers": settings.TOKEN_CAP_INTERDISCIPLINARY,
        "get_top_faculty": settings.TOKEN_CAP_TOP_FACULTY,
        "compare_faculty": settings.TOKEN_CAP_FACULTY_PROFILE,
        "find_similar_papers": settings.TOKEN_CAP_SEARCH_PAPERS,
        "get_research_trends": settings.TOKEN_CAP_PUBLICATION_STATS,
    }
    return caps.get(name, settings.TOKEN_CAP_DEFAULT)
