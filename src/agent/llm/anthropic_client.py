"""Anthropic Claude LLM factory via langchain-anthropic's ChatAnthropic.

Provides two instances:
- make_tool_llm(): temperature=0, deterministic tool selection.
- make_answer_llm(): temperature=0, tagged ["answer"] for SSE stream filtering.

Outbound calls to api.anthropic.com go through the campus proxy when
LLM_HTTP_PROXY_URL is set. Deliberately NOT the generic HTTP_PROXY/HTTPS_PROXY
env vars: those are container-wide, and the underlying httpx client auto-reads
them when present (trust_env), which would also route anything else in the
container that honors those vars (e.g. Python's urllib, used by this image's own
HEALTHCHECK to hit its own /health over loopback) through the proxy. Building an
explicit httpx client here — with trust_env disabled and only our own proxy
applied — keeps proxy usage scoped to exactly this client.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel


def _build(api_key: str, model: str, max_tokens: int, proxy_url: str | None) -> ChatAnthropic:
    kwargs: dict = {
        "model": model,
        "api_key": api_key,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    # api.anthropic.com is external internet; route through the campus proxy in
    # prod (LLM_HTTP_PROXY_URL). ChatAnthropic's own `anthropic_proxy` handles
    # this scoped to just this client — the generic HTTP_PROXY/HTTPS_PROXY env
    # vars are deliberately kept blank in the container (see the module docstring).
    if proxy_url:
        kwargs["anthropic_proxy"] = proxy_url
    return ChatAnthropic(**kwargs)


def make_tool_llm(
    *,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 1024,
    proxy_url: str | None = None,
) -> BaseChatModel:
    """LLM instance for tool-selection calls (temperature=0, deterministic)."""
    return _build(api_key, model, max_tokens, proxy_url)


def make_answer_llm(
    *,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 1024,
    proxy_url: str | None = None,
) -> BaseChatModel:
    """LLM instance for the final answer stream (temperature=0, tagged).

    Temperature is 0 to keep the answer faithful to tool results — higher
    temperature lets the model embellish with plausible-but-unsupported detail
    (invented counts, "top contributors", inferred categories), which is the main
    source of post-tool hallucination.
    """
    return _build(api_key, model, max_tokens, proxy_url).with_config(tags=["answer"])
