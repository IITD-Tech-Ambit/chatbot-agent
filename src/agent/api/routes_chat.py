"""SSE chat endpoint.

Thin HTTP adapter over the chat pipeline. Drives the frontend SSE contract:
thinking | status | sources | chart | token | done | error.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from agent.api.schemas import ChatRequest
from agent.api.chat_pipeline import handle_chat, require_user_id

router = APIRouter()


@router.get("/quota")
async def quota(request: Request) -> dict[str, int]:
    """Per-user daily quota state ({limit, used, remaining}), auth required."""
    user_id = require_user_id(request)
    state = await request.app.state.quota_store.peek(user_id)
    return {"limit": state.limit, "used": state.used, "remaining": state.remaining}


@router.post("/chat")
async def chat(request: Request, body: ChatRequest):
    return await handle_chat(request, body)
