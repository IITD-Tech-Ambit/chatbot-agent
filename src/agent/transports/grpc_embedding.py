"""gRPC adapter for EmbeddingTransport — embedding.v1 through Envoy."""

from __future__ import annotations

import logging

import grpc

from embedding.v1 import embedding_pb2, embedding_pb2_grpc

logger = logging.getLogger(__name__)


class GrpcEmbeddingTransport:
    def __init__(self, envoy_target: str, timeout_ms: int = 10_000) -> None:
        self._channel = grpc.aio.insecure_channel(envoy_target)
        self._stub = embedding_pb2_grpc.EmbeddingServiceStub(self._channel)
        self._timeout = timeout_ms / 1000.0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._stub.Embed(
            embedding_pb2.EmbedRequest(texts=texts),
            timeout=self._timeout * (2 if len(texts) > 1 else 1),
        )
        return [list(e.values) for e in response.embeddings]

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        response = await self._stub.Rerank(
            embedding_pb2.RerankRequest(
                query=query, documents=documents, top_n=top_n or 0
            ),
            timeout=self._timeout,
        )
        return [(r.index, r.score) for r in response.results]

    async def health(self) -> bool:
        try:
            await self._stub.Embed(
                embedding_pb2.EmbedRequest(texts=["ping"]), timeout=2.0
            )
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._channel.close()
