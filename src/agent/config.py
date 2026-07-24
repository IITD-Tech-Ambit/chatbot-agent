"""Centralized configuration via pydantic-settings. Reads from .env / environment."""

from __future__ import annotations

import json as _json

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PORT: int = 3003
    HOST: str = "0.0.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "info"

    # Secure-by-default: MUST stay true (or unset) in production. Mirrors the
    # ENABLE_AUTH toggle in api-gateway/auth-service — when the gateway bypasses
    # OAuth it injects the same trusted mock identity (devuser) on every request,
    # so that single identity would otherwise burn through the daily quota during
    # a testing session. Set false only for local/dev testing.
    ENABLE_AUTH: bool = True

    # CORS — comma-separated origins or JSON array; "*" allows all (dev default)
    ALLOWED_ORIGINS: list[str] = ["*"]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def _parse_origins(cls, v: object) -> list[str]:
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                parsed = _json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except _json.JSONDecodeError:
                pass
            return [o.strip() for o in v.split(",") if o.strip()]
        return ["*"]

    MONGODB_URI: str = "mongodb://localhost:27017/research_db"

    OPENSEARCH_NODE: str = "http://localhost:9200"
    OPENSEARCH_USER: str = ""
    OPENSEARCH_PASSWORD: str = ""
    OPENSEARCH_INDEX: str = "research_documents"
    OPENSEARCH_IP_INDEX: str = "ip_documents"
    OPENSEARCH_VERIFY_CERTS: bool = False
    OPENSEARCH_USE_SSL: bool = False

    REDIS_URL: str = "redis://localhost:6379"

    # East-west transport: "grpc" routes embedding + faculty-for-query calls
    # through Envoy (production mesh); "http" keeps direct REST for local dev.
    MESH_TRANSPORT: str = "http"
    ENVOY_GRPC_TARGET: str = "envoy:10000"

    # gRPC listener for chat.v1.CheckQuota (served alongside the HTTP app)
    GRPC_ENABLED: bool = True
    GRPC_PORT: int = 50054

    # Embedding service (HTTP fallback when MESH_TRANSPORT=http)
    EMBEDDING_SERVICE_URL: str = "http://localhost:8000"
    EMBEDDING_TIMEOUT_MS: int = 10_000

    # Search API (HTTP fallback when MESH_TRANSPORT=http)
    SEARCH_API_URL: str = "http://localhost:3001"

    # Main backend (directory). Used at startup to build the department/centre/
    # school reference from the SAME endpoint the Directory page calls, so the
    # bot's lists match that page exactly.
    BACKEND_API_URL: str = "http://localhost:3002"

    # Per-user daily message quota (IST calendar day). Only categories listed
    # here (comma-separated, case-insensitive substring match against the
    # IITD OAuth `category` claim) are subject to the limit — see
    # agent.services.quota.is_quota_exempt. Comma-separated kerberos IDs in
    # CHAT_QUOTA_WHITELIST_KERBEROS bypass the limit regardless of category.
    CHAT_QUOTA_DAILY: int = 5
    CHAT_QUOTA_LIMITED_CATEGORIES: str = "student"
    CHAT_QUOTA_WHITELIST_KERBEROS: str = ""

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "grok-4.3"
    GROQ_EXTRACT_MODEL: str = "grok-3-mini"  # cheap model for query parsing
    MAX_ANSWER_TOKENS: int = 1024

    # Outbound calls to api.x.ai need the campus proxy to reach the internet.
    # Deliberately NOT the generic HTTP_PROXY/HTTPS_PROXY env vars: those are
    # container-wide, so anything else in the container that also auto-honors
    # them (e.g. Python's urllib, used by this image's own HEALTHCHECK to hit
    # its own /health over loopback) would try to route through the proxy
    # too. See src/agent/llm/groq_client.py and src/agent/rag/query_parser.py.
    LLM_HTTP_PROXY_URL: str = ""

    # Two rounds let the agent chain lookup_ipc_classification → search_ips /
    # get_ip_stats (resolve an IPC code, then run the refined patent query).
    # The graph still stops early whenever the LLM emits no further tool calls,
    # so single-round queries are unaffected.
    MAX_TOOL_ROUNDS: int = 2
    CHAT_TOP_K: int = 8
    CHAT_MAX_HISTORY_TURNS: int = 5
    CHAT_MAX_MESSAGE_LENGTH: int = 2000
    HISTORY_TOKEN_BUDGET: int = 800

    LLM_CACHE_TTL: int = 90
    EMBEDDING_CACHE_TTL: int = 86400

    # Per-tool output token caps (approximate char counts; 1 token ≈ 4 chars)
    TOKEN_CAP_SEARCH_PAPERS: int = 8000
    TOKEN_CAP_FACULTY_PROFILE: int = 2000
    TOKEN_CAP_PUBLICATION_STATS: int = 1500
    TOKEN_CAP_DEPARTMENT_PROFILE: int = 2500
    TOKEN_CAP_LIST_DEPARTMENTS: int = 3000
    TOKEN_CAP_FACULTY_EXPERTISE: int = 2000
    TOKEN_CAP_INTERDISCIPLINARY: int = 2000
    TOKEN_CAP_TOP_FACULTY: int = 3000
    TOKEN_CAP_SEARCH_IPS: int = 4000
    TOKEN_CAP_IP_DETAILS: int = 2500
    TOKEN_CAP_IP_STATS: int = 2000
    TOKEN_CAP_IPS_BY_FACULTY: int = 3000
    TOKEN_CAP_IPC_LOOKUP: int = 1500
    # Classification / taxonomy tools (thematic areas + domains)
    TOKEN_CAP_THEMATIC_AREAS: int = 2000
    TOKEN_CAP_RESEARCH_DOMAINS: int = 3000
    TOKEN_CAP_CLASSIFICATION_PAPERS: int = 4000
    TOKEN_CAP_CLASSIFICATION_FACULTY: int = 4500
    TOKEN_CAP_THEME_BREAKDOWN: int = 2500
    TOKEN_CAP_THEME_DISTRIBUTION: int = 2500
    TOKEN_CAP_DEFAULT: int = 1500

    # IPC classification lookup (lookup_ipc_classification tool). On a cache
    # miss the tool fetches the scheme entry from WIPO's published IPC data over
    # HTTP with a short timeout, then caches it in Redis.
    IPC_WIPO_API_URL: str = "https://ipcpub.wipo.int/rest-services/ipc"
    IPC_LOOKUP_TIMEOUT_MS: int = 4000
    IPC_CACHE_TTL: int = 2_592_000  # 30 days

    # Context budget for answer generation. _enforce_context_budget allows
    # CONTEXT_ANSWER_RESERVE * 4 characters across ALL messages — system prompt
    # included. The system prompt now carries the full IIT Delhi structural
    # reference (~14k chars), so this must leave room for that PLUS history,
    # the user message, and every tool result; otherwise tool outputs get
    # replaced with {"truncated": true} and the bot answers "I couldn't find that".
    CONTEXT_ANSWER_RESERVE: int = 12288


settings = Settings()
