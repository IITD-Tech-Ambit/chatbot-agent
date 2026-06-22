"""FastAPI application with lifespan for initializing all services."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.config import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: connect to all backends, build graph. Shutdown: close connections."""

    # ── Data layer ──
    from agent.data import mongo, opensearch, redis as redis_mod

    db = await mongo.connect(settings.MONGODB_URI)
    app.state.db = db

    os_client = await opensearch.connect(
        settings.OPENSEARCH_NODE,
        settings.OPENSEARCH_USER,
        settings.OPENSEARCH_PASSWORD,
        verify_certs=settings.OPENSEARCH_VERIFY_CERTS,
        use_ssl=settings.OPENSEARCH_USE_SSL,
    )
    app.state.opensearch = os_client

    redis_client = await redis_mod.connect(settings.REDIS_URL)
    app.state.redis = redis_client

    # ── Repositories ──
    from agent.repositories.faculty_repo import FacultyRepository
    from agent.repositories.research_repo import ResearchRepository

    faculty_repo = FacultyRepository(db)
    research_repo = ResearchRepository(db)

    # Expose repositories on app.state for fast-path routing
    app.state.faculty_repo = faculty_repo
    app.state.research_repo = research_repo

    # ── RAG ──
    from agent.rag.embeddings import EmbeddingClient
    from agent.rag.retriever import Retriever

    embedding_client = EmbeddingClient(
        base_url=settings.EMBEDDING_SERVICE_URL,
        redis_client=redis_client,
        timeout_ms=settings.EMBEDDING_TIMEOUT_MS,
        cache_ttl=settings.EMBEDDING_CACHE_TTL,
    )
    app.state.embedding_client = embedding_client

    retriever = Retriever(
        opensearch=os_client,
        index_name=settings.OPENSEARCH_INDEX,
        research_repo=research_repo,
        embedding_client=embedding_client,
        top_k=settings.CHAT_TOP_K,
    )

    # ── Tool registry ──
    from agent.tools import _registry

    _registry.init(
        retriever=retriever,
        faculty_repo=faculty_repo,
        research_repo=research_repo,
        config=settings,
    )
    app.state.tools = _registry.all_tools()

    # ── LLM (xAI Grok) ──
    from agent.llm.groq_client import make_tool_llm, make_answer_llm

    if not settings.GROQ_API_KEY:
        logger.error("GROQ_API_KEY (xAI API key) not set — LLM calls will fail")

    tool_llm = make_tool_llm(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        max_tokens=settings.MAX_ANSWER_TOKENS,
    )
    answer_llm = make_answer_llm(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        max_tokens=settings.MAX_ANSWER_TOKENS,
    )
    app.state.tool_llm = tool_llm
    app.state.answer_llm = answer_llm

    # ── Graph ──
    from agent.graph.builder import build_graph

    app.state.graph = build_graph(tool_llm, answer_llm)

    # ── Services ──
    from agent.services.cache import LLMCache

    app.state.llm_cache = LLMCache(redis_client, ttl=settings.LLM_CACHE_TTL)

    logger.info(
        "Chatbot agent started on http://%s:%s (model: %s via xAI)",
        settings.HOST, settings.PORT, settings.GROQ_MODEL,
    )
    yield

    # ── Shutdown ──
    await opensearch.close()
    await redis_mod.close()
    await mongo.close()
    logger.info("Chatbot agent shut down")


app = FastAPI(
    title="IIT Delhi Research Chatbot Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.ALLOWED_ORIGINS != ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": str(exc), "statusCode": 500},
    )


# ── Routes ──
from agent.api.routes_health import router as health_router
from agent.api.routes_chat import router as chat_router

app.include_router(health_router)
app.include_router(chat_router, prefix="/api/v1")

from agent import metrics as _metrics
_metrics.setup_metrics(app)
