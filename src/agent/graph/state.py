"""Agent state definition for the LangGraph StateGraph."""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_rounds: int
    paper_sources: list[dict[str, Any]]
