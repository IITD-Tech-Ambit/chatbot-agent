"""Integration tests for the SSE chat endpoint.

Tests SSE event ordering, rate limiting, meta short-circuits.
All backends mocked — no real DB/LLM required.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from tests.conftest import FakeRedis, FakeToolCallingLLM, FakeNoToolLLM

# Trusted identity header normally injected by the api-gateway
AUTH_HEADERS = {"x-user-id": "testuser"}


def _make_app():
    """Create a FastAPI app with mocked state for testing."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from agent.api.routes_health import router as health_router
    from agent.api.routes_chat import router as chat_router
    from agent.services.cache import LLMCache
    from agent.services.quota import RedisQuotaStore

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(health_router)
    app.include_router(chat_router, prefix="/api/v1")

    redis = FakeRedis()
    app.state.llm_cache = LLMCache(redis, ttl=0)
    app.state.quota_store = RedisQuotaStore(redis, daily_limit=5)

    fake_llm = FakeToolCallingLLM(answer="Test answer from LLM.")
    app.state.tool_llm = fake_llm
    app.state.answer_llm = fake_llm
    app.state.tools = []
    app.state.graph = _make_fake_graph()

    # Dummy state for health checks
    app.state.db = MagicMock()
    app.state.opensearch = MagicMock()
    app.state.redis = redis
    app.state.embedding_client = MagicMock()

    return app


def _make_fake_graph():
    """Create a minimal fake graph that yields expected SSE events."""

    class FakeGraph:
        async def astream_events(self, state, config=None, version="v2"):
            yield {
                "event": "on_tool_start",
                "name": "search_papers",
                "tags": [],
                "data": {},
            }
            yield {
                "event": "on_tool_end",
                "name": "search_papers",
                "tags": [],
                "data": {
                    "output": json.dumps({
                        "papers": [
                            {"citation_index": 1, "title": "Test Paper", "authors": ["Author A"], "year": 2023, "field": "CS", "citations": 10}
                        ]
                    })
                },
            }

            class FakeChunk:
                content = "This is a test answer."

            yield {
                "event": "on_chat_model_stream",
                "name": "answer_llm",
                "tags": ["answer"],
                "data": {"chunk": FakeChunk()},
            }

    return FakeGraph()


class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_sse_event_order(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/chat", headers=AUTH_HEADERS, json={"message": "Tell me about ML research"})

        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")

        events = []
        for line in lines:
            if line.startswith("event: "):
                events.append(line.split("event: ")[1])

        assert "thinking" in events
        assert "sources" in events
        assert "token" in events
        assert "done" in events
        assert events.index("thinking") < events.index("sources")
        assert events.index("sources") < events.index("token")
        assert events.index("token") < events.index("done")

    @pytest.mark.asyncio
    async def test_meta_short_circuit_no_llm(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/chat", headers=AUTH_HEADERS, json={"message": "who are you"})

        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        events = [l for l in lines if l.startswith("event: ")]
        event_names = [e.split("event: ")[1] for e in events]

        assert "token" in event_names
        assert "done" in event_names
        # No status or sources (no tool calls or graph execution)
        assert "status" not in event_names
        assert "sources" not in event_names

    @pytest.mark.asyncio
    async def test_empty_message_400(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/chat", headers=AUTH_HEADERS, json={"message": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_valid_request_returns_200(self):
        """Sanity: a well-formed request always gets a 200 stream (rate limiter removed)."""
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/chat", headers=AUTH_HEADERS, json={"message": "test query about IIT Delhi"})
        assert resp.status_code == 200


class TestChatAuthAndQuota:
    @pytest.mark.asyncio
    async def test_missing_identity_401(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/chat", json={"message": "test question"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_quota_exhaustion_429(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for _ in range(5):
                resp = await client.post(
                    "/api/v1/chat", headers=AUTH_HEADERS,
                    json={"message": "a research question about ML"},
                )
                assert resp.status_code == 200
            resp = await client.post(
                "/api/v1/chat", headers=AUTH_HEADERS,
                json={"message": "a research question about ML"},
            )
        assert resp.status_code == 429
        data = resp.json()
        assert data["remaining"] == 0
        assert data["limit"] == 5

    @pytest.mark.asyncio
    async def test_greetings_do_not_consume_quota(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/chat", headers=AUTH_HEADERS, json={"message": "who are you"}
            )
            assert resp.status_code == 200
            quota = (await client.get("/api/v1/quota", headers=AUTH_HEADERS)).json()
        assert quota == {"limit": 5, "used": 0, "remaining": 5, "unlimited": False}

    @pytest.mark.asyncio
    async def test_quota_endpoint_counts_usage(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/api/v1/chat", headers=AUTH_HEADERS,
                json={"message": "a research question about ML"},
            )
            quota = (await client.get("/api/v1/quota", headers=AUTH_HEADERS)).json()
        assert quota == {"limit": 5, "used": 1, "remaining": 4, "unlimited": False}

    @pytest.mark.asyncio
    async def test_quota_endpoint_requires_identity(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/quota")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_faculty_category_is_unlimited(self):
        app = _make_app()
        headers = {**AUTH_HEADERS, "x-user-kerberos": "prof123", "x-user-category": "Faculty"}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for _ in range(6):
                resp = await client.post(
                    "/api/v1/chat", headers=headers, json={"message": "a research question about ML"},
                )
                assert resp.status_code == 200
            quota = (await client.get("/api/v1/quota", headers=headers)).json()
        assert quota == {"unlimited": True}

    @pytest.mark.asyncio
    async def test_whitelisted_student_kerberos_is_unlimited(self, monkeypatch):
        from agent.config import settings

        monkeypatch.setattr(settings, "CHAT_QUOTA_WHITELIST_KERBEROS", "ch7221511")
        app = _make_app()
        headers = {**AUTH_HEADERS, "x-user-kerberos": "ch7221511", "x-user-category": "Student"}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for _ in range(6):
                resp = await client.post(
                    "/api/v1/chat", headers=headers, json={"message": "a research question about ML"},
                )
                assert resp.status_code == 200
            quota = (await client.get("/api/v1/quota", headers=headers)).json()
        assert quota == {"unlimited": True}

    @pytest.mark.asyncio
    async def test_missing_category_still_applies_student_limit(self):
        """No x-user-category header (e.g. older session) must not be treated as unlimited."""
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for _ in range(5):
                resp = await client.post(
                    "/api/v1/chat", headers=AUTH_HEADERS, json={"message": "a research question about ML"},
                )
                assert resp.status_code == 200
            resp = await client.post(
                "/api/v1/chat", headers=AUTH_HEADERS, json={"message": "a research question about ML"},
            )
        assert resp.status_code == 429


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self):
        app = _make_app()
        app.state.db = MagicMock()
        app.state.db.command = AsyncMock(return_value={"ok": 1})
        app.state.opensearch = MagicMock()
        app.state.opensearch.cluster = MagicMock()
        app.state.opensearch.cluster.health = AsyncMock(return_value={"status": "green"})
        app.state.embedding_client = MagicMock()
        app.state.embedding_client.health = AsyncMock(return_value=True)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
