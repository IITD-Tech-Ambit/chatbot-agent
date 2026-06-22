"""Prometheus metrics for the chatbot-agent service.

Exposes RED metrics plus chatbot-specific domain metrics.
Scraped by Prometheus at chatbot:3003/metrics on the internal Docker network.
nginx blocks the public /chat-api/metrics path.
"""

import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response

# ── HTTP RED metrics ─────────────────────────────────────────────────
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=("method", "route", "status_code"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)

# ── Chatbot domain metrics ────────────────────────────────────────────
CHATBOT_LLM_REQUESTS_TOTAL = Counter(
    "chatbot_llm_requests_total",
    "Total LLM calls by outcome",
    labelnames=("outcome",),  # outcome: success|error|cached
)

CHATBOT_TOOL_CALLS_TOTAL = Counter(
    "chatbot_tool_calls_total",
    "Total tool calls by tool name",
    labelnames=("tool",),
)

CHATBOT_TIME_TO_FIRST_TOKEN_SECONDS = Histogram(
    "chatbot_time_to_first_token_seconds",
    "Time from request start to first token emitted (seconds)",
    buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30),
)

CHATBOT_CHAT_DURATION_SECONDS = Histogram(
    "chatbot_chat_duration_seconds",
    "Total time to complete a chat response (seconds)",
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)


def setup_metrics(app) -> None:
    """Attach the timing middleware and /metrics endpoint to a FastAPI app."""

    @app.middleware("http")
    async def _track_requests(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - start
            route = request.scope.get("route")
            route_label = getattr(route, "path", None) or request.url.path
            HTTP_REQUEST_DURATION.labels(
                request.method, route_label, str(status_code)
            ).observe(elapsed)

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
