"""Tests for the LangGraph agent graph logic.

Uses FakeToolCallingChatModel — no Ollama required.
Tests: no-tool -> forced search, tool_rounds capped at MAX_TOOL_ROUNDS, budget guard.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from agent.graph.state import AgentState
from agent.graph.nodes import agent_node, answer_node, route_after_agent, _enforce_context_budget, init_node_deps
from agent.config import settings
from tests.conftest import FakeToolCallingLLM, FakeNoToolLLM


class TestRouteAfterAgent:
    def test_routes_to_tools_when_tool_calls(self):
        state: AgentState = {
            "messages": [AIMessage(content="", tool_calls=[{"id": "1", "name": "search_papers", "args": {"query": "x"}}])],
            "tool_rounds": 0,
            "paper_sources": [],
        }
        assert route_after_agent(state) == "tools"

    def test_routes_to_answer_when_no_tool_calls(self):
        state: AgentState = {
            "messages": [AIMessage(content="Here's your answer")],
            "tool_rounds": 0,
            "paper_sources": [],
        }
        assert route_after_agent(state) == "answer"

    def test_routes_to_answer_when_empty_content_no_calls(self):
        """After max rounds, agent_node returns empty AIMessage with no tool_calls."""
        state: AgentState = {
            "messages": [AIMessage(content="")],
            "tool_rounds": 1,
            "paper_sources": [],
        }
        assert route_after_agent(state) == "answer"


class TestAgentNode:
    @pytest.mark.asyncio
    async def test_with_tool_calls(self):
        llm = FakeToolCallingLLM(
            tool_calls=[{"id": "c1", "name": "search_papers", "args": {"query": "ML"}}],
        )
        init_node_deps(tool_llm=llm, answer_llm=llm, tools=[])
        state: AgentState = {
            "messages": [HumanMessage(content="Tell me about ML")],
            "tool_rounds": 0,
            "paper_sources": [],
        }
        result = await agent_node(state)
        last_msg = result["messages"][-1]
        assert hasattr(last_msg, "tool_calls")
        assert len(last_msg.tool_calls) > 0

    @pytest.mark.asyncio
    async def test_force_tool_when_no_calls(self):
        """When LLM returns no tool_calls, agent_node should inject a forced search_papers."""
        llm = FakeNoToolLLM()
        init_node_deps(tool_llm=llm, answer_llm=llm, tools=[])
        state: AgentState = {
            "messages": [HumanMessage(content="Tell me about quantum computing")],
            "tool_rounds": 0,
            "paper_sources": [],
        }
        result = await agent_node(state)
        last_msg = result["messages"][-1]
        assert hasattr(last_msg, "tool_calls")
        assert last_msg.tool_calls[0]["name"] == "search_papers"
        assert "quantum computing" in last_msg.tool_calls[0]["args"]["query"]

    @pytest.mark.asyncio
    async def test_early_exit_when_rounds_exhausted(self):
        """When tool_rounds >= MAX_TOOL_ROUNDS, agent_node returns empty message."""
        llm = FakeToolCallingLLM(
            tool_calls=[{"id": "c1", "name": "search_papers", "args": {"query": "ML"}}],
        )
        init_node_deps(tool_llm=llm, answer_llm=llm, tools=[])
        state: AgentState = {
            "messages": [HumanMessage(content="Tell me about ML")],
            "tool_rounds": settings.MAX_TOOL_ROUNDS,
            "paper_sources": [],
        }
        result = await agent_node(state)
        last_msg = result["messages"][-1]
        assert not getattr(last_msg, "tool_calls", None)


class TestBudgetGuard:
    def test_truncates_long_tool_output(self):
        long_content = "x" * 10000
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="query"),
            AIMessage(content="", tool_calls=[{"id": "1", "name": "search_papers", "args": {}}]),
            ToolMessage(content=long_content, tool_call_id="1", name="search_papers"),
        ]
        result = _enforce_context_budget(messages)
        tool_msg = [m for m in result if isinstance(m, ToolMessage)][0]
        assert len(tool_msg.content) < len(long_content)

    def test_preserves_short_tool_output(self):
        short_content = '{"papers": []}'
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="query"),
            AIMessage(content="", tool_calls=[{"id": "1", "name": "search_papers", "args": {}}]),
            ToolMessage(content=short_content, tool_call_id="1", name="search_papers"),
        ]
        result = _enforce_context_budget(messages)
        tool_msg = [m for m in result if isinstance(m, ToolMessage)][0]
        assert tool_msg.content == short_content
