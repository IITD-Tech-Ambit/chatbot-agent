"""gRPC transport for chat.v1.ChatService (CheckQuota).

Thin adapter over the QuotaStore port; served on the internal network via
Envoy. Each gunicorn worker binds with SO_REUSEPORT (grpc default on Linux).
"""

from __future__ import annotations

import logging

import grpc

from chat.v1 import chat_pb2, chat_pb2_grpc
from agent.services.quota import QuotaStore

logger = logging.getLogger(__name__)


class ChatServicer(chat_pb2_grpc.ChatServiceServicer):
    def __init__(self, quota_store: QuotaStore) -> None:
        self._quota = quota_store

    async def CheckQuota(self, request, context):
        if not request.user_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "user_id is required")
        state = await self._quota.peek(request.user_id)
        return chat_pb2.CheckQuotaResponse(
            limit=state.limit,
            used=state.used,
            remaining=state.remaining,
        )


async def start_grpc_server(quota_store: QuotaStore, port: int) -> grpc.aio.Server:
    server = grpc.aio.server()
    chat_pb2_grpc.add_ChatServiceServicer_to_server(ChatServicer(quota_store), server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    await server.start()
    logger.info("chat.v1 gRPC server listening on :%d", port)
    return server
