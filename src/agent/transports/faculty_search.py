"""FacultySearchClient adapters: gRPC (search.v1 via Envoy) and HTTP (dev)."""

from __future__ import annotations

import logging
from typing import Any

import grpc
import httpx

from search.v1 import search_pb2, search_pb2_grpc

logger = logging.getLogger(__name__)


class GrpcFacultySearchClient:
    def __init__(self, envoy_target: str, timeout_s: float = 20.0) -> None:
        self._channel = grpc.aio.insecure_channel(envoy_target)
        self._stub = search_pb2_grpc.SearchServiceStub(self._channel)
        self._timeout = timeout_s

    async def faculty_for_query(self, query: str) -> dict[str, Any]:
        response = await self._stub.FacultyForQuery(
            search_pb2.FacultyForQueryRequest(query=query),
            timeout=self._timeout,
        )
        return {
            "departments": [
                {
                    "name": dept.name,
                    "total_paper_count": dept.total_paper_count,
                    "faculty": [
                        {
                            "name": f.name,
                            "author_id": f.author_id,
                            "paper_count": f.paper_count,
                            "relevance_score": f.relevance_score,
                        }
                        for f in dept.faculty
                    ],
                }
                for dept in response.departments
            ],
            "total_faculty": response.total_faculty,
            "total_matching_papers": response.total_matching_papers,
        }

    async def close(self) -> None:
        await self._channel.close()


class HttpFacultySearchClient:
    def __init__(self, base_url: str, timeout_s: float = 20.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s

    async def faculty_for_query(self, query: str) -> dict[str, Any]:
        url = f"{self._base_url}/api/v1/search/faculty-for-query"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params={"query": query})
            resp.raise_for_status()
            return resp.json()
