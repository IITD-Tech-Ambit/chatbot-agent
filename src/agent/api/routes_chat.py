"""SSE chat endpoint.

Drives graph.astream_events(version="v2") and maps LangGraph events to the
frontend SSE contract: thinking | status | sources | chart | token | done | error.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage

from agent.api.schemas import ChatRequest
from agent.api.sse_events import (
    ChartEvent,
    DoneEvent,
    ErrorEvent,
    StatusEvent,
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
from agent.llm.prompts import get_system_prompt, SECURITY_NOTE
from agent.graph.nodes import extract_sources
from agent.graph.state import AgentState
from agent.tools._registry import status_for

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Friendly thinking labels — never expose internal tool names to frontend ──
_THINKING_LABELS: dict[str, str] = {
    "search_papers": "Searching indexed publications",
    "find_faculty_for_topic": "Identifying relevant researchers",
    "find_faculty_by_expertise": "Scanning faculty expertise profiles",
    "get_faculty_profile": "Loading faculty profile",
    "get_publication_stats": "Computing publication statistics",
    "compare_faculty": "Comparing researcher profiles",
    "find_similar_papers": "Finding related work",
    "get_research_trends": "Analyzing publication trends",
    "get_department_profile": "Loading department overview",
    "list_departments": "Retrieving department list",
    "find_interdisciplinary_papers": "Exploring interdisciplinary research",
}


def _thinking_label(tool_name: str) -> str:
    return _THINKING_LABELS.get(tool_name, "Processing your query")


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


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


@router.post("/chat")
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    start_time = time.time()
    app = request.app

    message = sanitize_message(body.message, settings.CHAT_MAX_MESSAGE_LENGTH)
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    # ── Guardrails: meta / capability / greeting ──
    meta = classify_meta(message)
    if meta:
        logger.info("Chat meta short-circuit: %s", meta)

        async def _meta_stream() -> AsyncGenerator[str, None]:
            yield _sse("token", TokenEvent(text=canned_reply(meta)).model_dump())
            yield _sse("done", DoneEvent(took_ms=int((time.time() - start_time) * 1000)).model_dump())

        return StreamingResponse(_meta_stream(), media_type="text/event-stream")

    # ── Hard-block injection attempts ──
    if detect_injection(message):
        logger.warning("Injection attempt hard-blocked for message: %s", message[:80])

        async def _block_stream() -> AsyncGenerator[str, None]:
            yield _sse("token", TokenEvent(text=canned_reply("refusal")).model_dump())
            yield _sse("done", DoneEvent(took_ms=int((time.time() - start_time) * 1000)).model_dump())

        return StreamingResponse(_block_stream(), media_type="text/event-stream")

    # ── Structured fast-path FIRST — beats cache so stale LLM refusals can't intercept ──
    from agent.routing.structured import match_structured, execute_structured

    llm_cache = app.state.llm_cache
    structured_match = match_structured(message)
    if structured_match:
        faculty_repo = app.state.faculty_repo
        research_repo = app.state.research_repo

        async def _structured_stream() -> AsyncGenerator[str, None]:
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

        return StreamingResponse(_structured_stream(), media_type="text/event-stream")

    # ── LLM response cache (only for queries that go to the LLM) ──
    cached = await llm_cache.get(message)
    if cached:
        async def _cached_stream() -> AsyncGenerator[str, None]:
            _metrics.CHATBOT_LLM_REQUESTS_TOTAL.labels(outcome="cached").inc()
            if cached.get("sources"):
                yield _sse("sources", cached["sources"])
            if cached.get("chart"):
                yield _sse("chart", cached["chart"])
            yield _sse("token", TokenEvent(text=cached["answer"]).model_dump())
            yield _sse("done", DoneEvent(
                took_ms=int((time.time() - start_time) * 1000), cached=True
            ).model_dump())

        return StreamingResponse(_cached_stream(), media_type="text/event-stream")

    # ── Full LLM graph path ──
    system_content = get_system_prompt()
    history = _trim_history(
        [t.model_dump() for t in body.history[-settings.CHAT_MAX_HISTORY_TURNS:]],
    )

    messages = [
        SystemMessage(content=system_content),
        *[HumanMessage(content=t["content"]) if t["role"] == "user"
          else SystemMessage(content=t["content"]) for t in history],
        HumanMessage(content=message),
    ]

    initial_state: AgentState = {
        "messages": messages,
        "tool_rounds": 0,
        "paper_sources": [],
    }

    graph = app.state.graph

    async def _event_stream() -> AsyncGenerator[str, None]:
        sources_emitted = False
        collected_answer = ""
        collected_sources: list[dict] = []
        collected_chart: dict | None = None
        active_tool: str | None = None
        first_token_recorded = False

        try:
            async for ev in graph.astream_events(initial_state, version="v2"):
                kind = ev.get("event", "")
                name = ev.get("name", "")
                tags = ev.get("tags") or []

                if kind == "on_tool_start":
                    active_tool = name
                    _metrics.CHATBOT_TOOL_CALLS_TOTAL.labels(tool=name).inc()
                    label = _thinking_label(name)
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

                        # ── Chart event for chart-capable tools ──
                        chart_ev: ChartEvent | None = build_chart_for_tool(name, data)
                        if chart_ev:
                            chart_payload = chart_ev.model_dump()
                            yield _sse("chart", chart_payload)
                            collected_chart = chart_payload

                        # ── Sources from search_papers ──
                        if name == "search_papers":
                            papers = data.get("papers", [])
                            seen: set[str] = set()
                            deduped: list[dict] = []
                            for p in papers:
                                t = p.get("title", "")
                                if t and t not in seen:
                                    seen.add(t)
                                    deduped.append({
                                        "index": p.get("citation_index"),
                                        "id": p.get("id", ""),
                                        "title": t,
                                        "authors": p.get("authors", []),
                                        "publication_year": p.get("year"),
                                        "document_type": p.get("document_type"),
                                        "field_associated": p.get("field"),
                                        "citation_count": p.get("citations", 0),
                                        "link": p.get("link"),
                                        "document_scopus_id": p.get("document_scopus_id"),
                                        "document_eid": p.get("document_eid"),
                                        "kerberos": p.get("kerberos"),
                                        "faculty_name": p.get("faculty_name"),
                                    })
                            if deduped and not sources_emitted:
                                yield _sse("sources", deduped)
                                collected_sources = deduped
                                sources_emitted = True

                    except (json.JSONDecodeError, AttributeError, TypeError):
                        pass

                    active_tool = None

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

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
