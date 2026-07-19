"""Chat request pipeline: guardrails → quota → structured → cache → agent stream."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from agent.api.schemas import ChatRequest
from agent.api.source_mapper import ips_to_sources, papers_to_sources
from agent.api.sse_events import (
    ChartEvent,
    DoneEvent,
    ErrorEvent,
    ThinkingEvent,
    TokenEvent,
)
from agent.api.chart_builder import build_chart_for_tool
from agent.config import settings
from agent import metrics as _metrics
from agent.guardrails.guardrails import (
    sanitize_message,
    classify_meta,
    canned_reply,
    detect_injection,
)
from agent.services.quota import is_quota_exempt
from agent.llm.prompts import get_system_prompt
from agent.graph.state import AgentState
from agent.tools.meta import thinking_label_for

logger = logging.getLogger(__name__)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def require_user_id(request: Request) -> str:
    """Trusted identity injected by the api-gateway (x-user-id). The gateway
    strips any client-supplied copy, so an empty header means the request
    bypassed the gateway — reject it (defense in depth behind the 401 the
    gateway already returns)."""
    user_id = (request.headers.get("x-user-id") or "").strip()
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Login with your IITD account to use the chat assistant.",
        )
    return user_id


def is_requester_quota_exempt(request: Request) -> bool:
    """Trusted x-user-kerberos / x-user-category, also injected by the gateway."""
    kerberos = request.headers.get("x-user-kerberos") or ""
    category = request.headers.get("x-user-category") or ""
    return is_quota_exempt(kerberos, category)


def _trim_history(history: list[dict], budget: int = settings.HISTORY_TOKEN_BUDGET) -> list[dict]:
    """Keep recent history turns within the token budget."""
    kept: list[dict] = []
    chars = 0
    char_budget = budget * 4
    for turn in reversed(history):
        turn_len = len(turn.get("content", ""))
        if chars + turn_len > char_budget:
            break
        kept.insert(0, turn)
        chars += turn_len
    return kept


def _canned_stream(text: str, start_time: float) -> StreamingResponse:
    async def _gen() -> AsyncGenerator[str, None]:
        yield _sse("token", TokenEvent(text=text).model_dump())
        yield _sse("done", DoneEvent(took_ms=int((time.time() - start_time) * 1000)).model_dump())

    return StreamingResponse(_gen(), media_type="text/event-stream")


async def _structured_stream(
    *,
    message: str,
    structured_match,
    faculty_repo,
    research_repo,
    llm_cache,
    start_time: float,
) -> AsyncGenerator[str, None]:
    from agent.routing.structured import execute_structured

    yield _sse("thinking", ThinkingEvent(
        step="Looking up data directly", detail="Fast database lookup"
    ).model_dump())
    try:
        result = await execute_structured(structured_match, faculty_repo, research_repo)
        if "error" in result:
            text = result["error"]
        else:
            text = result.get("text", "")

        yield _sse("token", TokenEvent(text=text).model_dump())
        took_ms = int((time.time() - start_time) * 1000)
        yield _sse("done", DoneEvent(took_ms=took_ms).model_dump())

        if text and "error" not in result:
            await llm_cache.set(message, {"answer": text, "sources": None, "chart": None})
    except Exception as exc:
        logger.error("Structured fast-path error: %s", exc, exc_info=True)
        yield _sse("error", ErrorEvent(message="Something went wrong. Please try again.").model_dump())


async def _cached_stream(cached: dict, start_time: float) -> AsyncGenerator[str, None]:
    _metrics.CHATBOT_LLM_REQUESTS_TOTAL.labels(outcome="cached").inc()
    if cached.get("sources"):
        yield _sse("sources", cached["sources"])
    if cached.get("chart"):
        yield _sse("chart", cached["chart"])
    yield _sse("token", TokenEvent(text=cached["answer"]).model_dump())
    yield _sse("done", DoneEvent(
        took_ms=int((time.time() - start_time) * 1000), cached=True
    ).model_dump())


async def _agent_stream(
    *,
    message: str,
    body: ChatRequest,
    graph,
    tools: list[BaseTool],
    llm_cache,
    start_time: float,
) -> AsyncGenerator[str, None]:
    system_content = get_system_prompt()
    history = _trim_history(
        [t.model_dump() for t in body.history[-settings.CHAT_MAX_HISTORY_TURNS:]],
    )

    def _history_msg(turn: dict):
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            return HumanMessage(content=content)
        if role == "assistant":
            return AIMessage(content=content)
        return SystemMessage(content=content)

    messages = [
        SystemMessage(content=system_content),
        *[_history_msg(t) for t in history],
        HumanMessage(content=message),
    ]

    initial_state: AgentState = {
        "messages": messages,
        "tool_rounds": 0,
    }

    sources_emitted = False
    collected_answer = ""
    collected_sources: list[dict] = []
    collected_chart: dict | None = None
    first_token_recorded = False

    try:
        async for ev in graph.astream_events(initial_state, version="v2"):
            kind = ev.get("event", "")
            name = ev.get("name", "")
            tags = ev.get("tags") or []

            if kind == "on_tool_start":
                _metrics.CHATBOT_TOOL_CALLS_TOTAL.labels(tool=name).inc()
                label = thinking_label_for(name, tools)
                yield _sse("thinking", ThinkingEvent(
                    step=label, detail=None
                ).model_dump())

            elif kind == "on_tool_end":
                try:
                    raw_output = ev.get("data", {}).get("output", "")
                    output_str = (
                        getattr(raw_output, "content", raw_output)
                        if not isinstance(raw_output, str) else raw_output
                    )
                    data = json.loads(output_str) if isinstance(output_str, str) else output_str

                    chart_ev: ChartEvent | None = build_chart_for_tool(name, data)
                    if chart_ev:
                        chart_payload = chart_ev.model_dump()
                        yield _sse("chart", chart_payload)
                        collected_chart = chart_payload

                    if name == "search_papers":
                        deduped = papers_to_sources(data.get("papers", []))
                        if deduped and not sources_emitted:
                            yield _sse("sources", deduped)
                            collected_sources = deduped
                            sources_emitted = True

                    elif name in ("search_ips", "find_ips_by_faculty"):
                        deduped = ips_to_sources(data.get("ips", []))
                        if deduped and not sources_emitted:
                            yield _sse("sources", deduped)
                            collected_sources = deduped
                            sources_emitted = True

                    elif name == "get_ip_details" and data.get("ip"):
                        deduped = ips_to_sources([data["ip"]])
                        if deduped and not sources_emitted:
                            yield _sse("sources", deduped)
                            collected_sources = deduped
                            sources_emitted = True

                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass

            elif kind == "on_chat_model_stream" and "answer" in tags:
                chunk = ev.get("data", {}).get("chunk")
                if chunk:
                    token = getattr(chunk, "content", "")
                    if token:
                        if not first_token_recorded:
                            _metrics.CHATBOT_TIME_TO_FIRST_TOKEN_SECONDS.observe(
                                time.time() - start_time
                            )
                            first_token_recorded = True
                        yield _sse("token", TokenEvent(text=token).model_dump())
                        collected_answer += token

        took_ms = int((time.time() - start_time) * 1000)
        _metrics.CHATBOT_CHAT_DURATION_SECONDS.observe(took_ms / 1000)
        _metrics.CHATBOT_LLM_REQUESTS_TOTAL.labels(outcome="success").inc()
        yield _sse("done", DoneEvent(took_ms=took_ms).model_dump())

        if collected_answer.strip():
            await llm_cache.set(message, {
                "answer": collected_answer,
                "sources": collected_sources or None,
                "chart": collected_chart,
            })

    except Exception as e:
        logger.error("Chat stream error: %s", e, exc_info=True)
        _metrics.CHATBOT_LLM_REQUESTS_TOTAL.labels(outcome="error").inc()
        yield _sse("error", ErrorEvent(message="Something went wrong. Please try again.").model_dump())


async def handle_chat(request: Request, body: ChatRequest) -> StreamingResponse | JSONResponse:
    """Orchestrate guardrails → quota → structured → cache → agent stream."""
    start_time = time.time()
    app = request.app
    user_id = require_user_id(request)

    message = sanitize_message(body.message, settings.CHAT_MAX_MESSAGE_LENGTH)
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    meta = classify_meta(message)
    if meta:
        logger.info("Chat meta short-circuit: %s", meta)
        return _canned_stream(canned_reply(meta), start_time)

    if detect_injection(message):
        logger.warning("Injection attempt hard-blocked for message: %s", message[:80])
        return _canned_stream(canned_reply("refusal"), start_time)

    # Per-user daily quota (IST) — counted after guardrail short-circuits so
    # greetings/injection blocks don't burn messages. Faculty/staff and
    # whitelisted kerberos IDs are exempt and never touch the counter.
    quota_state = (
        None if is_requester_quota_exempt(request)
        else await app.state.quota_store.consume(user_id)
    )
    if quota_state is not None and not quota_state.allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Quota Exceeded",
                "message": (
                    f"You have used all {quota_state.limit} chat messages for today. "
                    "Your quota resets at midnight IST."
                ),
                "limit": quota_state.limit,
                "remaining": 0,
                "statusCode": 429,
            },
        )

    # Structured fast-path first — beats cache so stale LLM refusals can't intercept
    from agent.routing.structured import match_structured

    llm_cache = app.state.llm_cache
    structured_match = match_structured(message)
    if structured_match:
        return StreamingResponse(
            _structured_stream(
                message=message,
                structured_match=structured_match,
                faculty_repo=app.state.faculty_repo,
                research_repo=app.state.research_repo,
                llm_cache=llm_cache,
                start_time=start_time,
            ),
            media_type="text/event-stream",
        )

    cached = await llm_cache.get(message)
    if cached:
        return StreamingResponse(
            _cached_stream(cached, start_time),
            media_type="text/event-stream",
        )

    return StreamingResponse(
        _agent_stream(
            message=message,
            body=body,
            graph=app.state.graph,
            tools=getattr(app.state, "tools", []) or [],
            llm_cache=llm_cache,
            start_time=start_time,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
