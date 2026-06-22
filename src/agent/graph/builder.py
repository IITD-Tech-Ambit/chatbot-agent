"""Build the LangGraph StateGraph for the agent.

Graph shape:
  agent -> route_after_agent -> {tools | answer}
  tools -> bump_rounds -> agent  (tool_rounds gate in agent_node prevents loops)

The agent node forces >=1 tool call (no-tool-call = inject search_papers).
When tool_rounds >= MAX_TOOL_ROUNDS, agent_node emits an empty AI message
(no tool_calls) which routes to answer.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent.graph.state import AgentState
from agent.graph.nodes import agent_node, answer_node, route_after_agent, init_node_deps
from agent.tools._registry import all_tools


def _bump_rounds(state: AgentState) -> dict[str, Any]:
    """Increment tool_rounds after tools execute, before re-entering agent."""
    return {"tool_rounds": state.get("tool_rounds", 0) + 1}


def build_graph(
    tool_llm: BaseChatModel,
    answer_llm: BaseChatModel,
) -> Any:
    """Construct and compile the agent graph."""
    tools = all_tools()
    tool_node = ToolNode(tools)

    init_node_deps(tool_llm, answer_llm, tools)

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("bump_rounds", _bump_rounds)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "answer": "answer"},
    )

    graph.add_edge("tools", "bump_rounds")
    graph.add_edge("bump_rounds", "agent")
    graph.add_edge("answer", END)

    return graph.compile()
