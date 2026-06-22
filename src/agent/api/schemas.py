"""Pydantic request/response schemas for the chat API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=12)
    user_id: str | None = Field(default=None, max_length=128, description="Optional authenticated user ID")
