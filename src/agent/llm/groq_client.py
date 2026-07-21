"""xAI Grok LLM factory via langchain-openai's ChatOpenAI.

xAI uses an OpenAI-compatible API at https://api.x.ai/v1.
Provides two instances:
- make_tool_llm(): temperature=0, deterministic tool selection.
- make_answer_llm(): temperature=0, tagged ["answer"] for SSE stream filtering.

Outbound calls to api.x.ai go through the campus proxy when LLM_HTTP_PROXY_URL
is set. Deliberately NOT the generic HTTP_PROXY/HTTPS_PROXY env vars: those
are container-wide, and langchain-openai's underlying httpx client auto-reads
them when present (trust_env), which would also route anything else in the
container that honors those vars (e.g. Python's urllib, used by this image's
own HEALTHCHECK to hit its own /health over loopback) through the proxy.
Building explicit httpx clients here — with trust_env disabled and only our
own proxy applied — keeps proxy usage scoped to exactly this client.
"""

from __future__ import annotations

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

XAI_BASE_URL = "https://api.x.ai/v1"


def _http_clients(proxy_url: str | None) -> tuple[httpx.Client, httpx.AsyncClient]:
    """Build sync/async httpx clients that ignore container-wide proxy env vars.

    trust_env=False stops httpx from auto-detecting HTTP_PROXY/HTTPS_PROXY;
    `proxy` is applied explicitly only when LLM_HTTP_PROXY_URL is configured.
    """
    return (
        httpx.Client(proxy=proxy_url, trust_env=False),
        httpx.AsyncClient(proxy=proxy_url, trust_env=False),
    )


def make_tool_llm(
    *,
    api_key: str,
    model: str = "grok-4.3",
    max_tokens: int = 1024,
    proxy_url: str | None = None,
) -> BaseChatModel:
    """LLM instance for tool-selection calls (temperature=0, deterministic)."""
    http_client, http_async_client = _http_clients(proxy_url)
    return ChatOpenAI(
        model=model,
        base_url=XAI_BASE_URL,
        api_key=api_key,
        temperature=0,
        max_tokens=max_tokens,
        http_client=http_client,
        http_async_client=http_async_client,
    )


def make_answer_llm(
    *,
    api_key: str,
    model: str = "grok-4.3",
    max_tokens: int = 1024,
    proxy_url: str | None = None,
) -> BaseChatModel:
    """LLM instance for the final answer stream (temperature=0, tagged).

    Temperature is 0 (not 0.3) to keep the answer faithful to tool results —
    higher temperature let the model embellish with plausible-but-unsupported
    detail (invented counts, "top contributors", inferred categories), which is
    the main source of post-tool hallucination.
    """
    http_client, http_async_client = _http_clients(proxy_url)
    llm = ChatOpenAI(
        model=model,
        base_url=XAI_BASE_URL,
        api_key=api_key,
        temperature=0,
        max_tokens=max_tokens,
        http_client=http_client,
        http_async_client=http_async_client,
    )
    return llm.with_config(tags=["answer"])
