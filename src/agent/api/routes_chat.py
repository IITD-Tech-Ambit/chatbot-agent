"""SSE chat endpoint.

Thin HTTP adapter over the chat pipeline. Drives the frontend SSE contract:
thinking | status | sources | chart | token | done | error.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from agent.api.schemas import ChatRequest
from agent.api.chat_pipeline import handle_chat, is_requester_quota_exempt, require_user_id

router = APIRouter()


@router.get("/quota")
async def quota(request: Request) -> dict[str, int | bool]:
    """Per-user daily quota state, auth required.

    Unlimited users (faculty/staff, whitelisted kerberos) get {"unlimited":
    true} and no limit/used/remaining — the frontend shows no counter at all.
    """
    user_id = require_user_id(request)
    if is_requester_quota_exempt(request):
        return {"unlimited": True}
    state = await request.app.state.quota_store.peek(user_id)
    return {"limit": state.limit, "used": state.used, "remaining": state.remaining, "unlimited": False}


@router.post("/chat")
async def chat(request: Request, body: ChatRequest):
    return await handle_chat(request, body)
