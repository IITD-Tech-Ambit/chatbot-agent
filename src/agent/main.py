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

    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required but not set — cannot start without an xAI API key")

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

    from agent.repositories.faculty_repo import FacultyRepository
    from agent.repositories.research_repo import ResearchRepository

    faculty_repo = FacultyRepository(db)
    research_repo = ResearchRepository(db)

    app.state.faculty_repo = faculty_repo
    app.state.research_repo = research_repo

    # Mesh transports (composition root: gRPC via Envoy or HTTP for dev)
    from agent.rag.embeddings import EmbeddingClient
    from agent.rag.query_parser import QueryParser
    from agent.rag.retriever import Retriever

    use_grpc = settings.MESH_TRANSPORT.lower() == "grpc"
    if use_grpc:
        from agent.transports.grpc_embedding import GrpcEmbeddingTransport
        from agent.transports.faculty_search import GrpcFacultySearchClient

        embedding_transport = GrpcEmbeddingTransport(
            settings.ENVOY_GRPC_TARGET, timeout_ms=settings.EMBEDDING_TIMEOUT_MS
        )
        search_client = GrpcFacultySearchClient(settings.ENVOY_GRPC_TARGET)
    else:
        from agent.transports.http_embedding import HttpEmbeddingTransport
        from agent.transports.faculty_search import HttpFacultySearchClient

        embedding_transport = HttpEmbeddingTransport(
            settings.EMBEDDING_SERVICE_URL, timeout_ms=settings.EMBEDDING_TIMEOUT_MS
        )
        search_client = HttpFacultySearchClient(settings.SEARCH_API_URL)

    embedding_client = EmbeddingClient(
        transport=embedding_transport,
        redis_client=redis_client,
        cache_ttl=settings.EMBEDDING_CACHE_TTL,
    )
    app.state.embedding_client = embedding_client

    if settings.GROQ_API_KEY:
        query_parser = QueryParser(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_EXTRACT_MODEL,
            proxy_url=settings.LLM_HTTP_PROXY_URL or None,
        )
    else:
        query_parser = None
        logger.warning(
            "GROQ_API_KEY not set — QueryParser disabled; faculty/dept kerberos filtering will not work"
        )

    retriever = Retriever(
        opensearch=os_client,
        index_name=settings.OPENSEARCH_INDEX,
        research_repo=research_repo,
        embedding_client=embedding_client,
        top_k=settings.CHAT_TOP_K,
        faculty_repo=faculty_repo,
        query_parser=query_parser,
    )

    from agent.tools.deps import ToolDeps
    from agent.tools._registry import build_tools

    tool_deps = ToolDeps(
        retriever=retriever,
        faculty_repo=faculty_repo,
        research_repo=research_repo,
        config=settings,
        search_client=search_client,
    )
    tools = build_tools(tool_deps)
    app.state.tools = tools

    from agent.llm.groq_client import make_tool_llm, make_answer_llm

    tool_llm = make_tool_llm(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        max_tokens=settings.MAX_ANSWER_TOKENS,
        proxy_url=settings.LLM_HTTP_PROXY_URL or None,
    )
    answer_llm = make_answer_llm(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        max_tokens=settings.MAX_ANSWER_TOKENS,
        proxy_url=settings.LLM_HTTP_PROXY_URL or None,
    )
    app.state.tool_llm = tool_llm
    app.state.answer_llm = answer_llm

    from agent.graph.builder import build_graph

    app.state.graph = build_graph(tool_llm, answer_llm, tools)

    from agent.services.cache import LLMCache
    from agent.services.quota import RedisQuotaStore

    app.state.llm_cache = LLMCache(redis_client, ttl=settings.LLM_CACHE_TTL)
    app.state.quota_store = RedisQuotaStore(redis_client, daily_limit=settings.CHAT_QUOTA_DAILY)

    # chat.v1 gRPC listener (CheckQuota)
    grpc_server = None
    if settings.GRPC_ENABLED:
        from agent.grpc_server import start_grpc_server

        grpc_server = await start_grpc_server(app.state.quota_store, settings.GRPC_PORT)

    logger.info(
        "Chatbot agent started on http://%s:%s (model: %s via xAI)",
        settings.HOST, settings.PORT, settings.GROQ_MODEL,
    )
    yield

    if grpc_server is not None:
        await grpc_server.stop(grace=5)
    close_transport = getattr(embedding_transport, "close", None)
    if close_transport:
        await close_transport()
    close_search = getattr(search_client, "close", None)
    if close_search:
        await close_search()
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


from agent.api.routes_health import router as health_router
from agent.api.routes_chat import router as chat_router

app.include_router(health_router)
app.include_router(chat_router, prefix="/api/v1")

from agent import metrics as _metrics
_metrics.setup_metrics(app)
