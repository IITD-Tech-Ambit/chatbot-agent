"""xAI Grok LLM factory via langchain-openai's ChatOpenAI.

xAI uses an OpenAI-compatible API at https://api.x.ai/v1.
Provides two instances:
- make_tool_llm(): temperature=0, deterministic tool selection.
- make_answer_llm(): temperature=0.3, tagged ["answer"] for SSE stream filtering.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

XAI_BASE_URL = "https://api.x.ai/v1"


def make_tool_llm(
    *,
    api_key: str,
    model: str = "grok-4.3",
    max_tokens: int = 1024,
) -> BaseChatModel:
    """LLM instance for tool-selection calls (temperature=0, deterministic)."""
    return ChatOpenAI(
        model=model,
        base_url=XAI_BASE_URL,
        api_key=api_key,
        temperature=0,
        max_tokens=max_tokens,
    )


def make_answer_llm(
    *,
    api_key: str,
    model: str = "grok-4.3",
    max_tokens: int = 1024,
) -> BaseChatModel:
    """LLM instance for the final answer stream (temperature=0.3, tagged)."""
    llm = ChatOpenAI(
        model=model,
        base_url=XAI_BASE_URL,
        api_key=api_key,
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return llm.with_config(tags=["answer"])
