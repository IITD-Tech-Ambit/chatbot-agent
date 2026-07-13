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

    # Per-user daily message quota (IST calendar day). Students only — see
    # agent.services.quota.is_quota_exempt. Comma-separated kerberos IDs here
    # bypass the limit regardless of category.
    CHAT_QUOTA_DAILY: int = 5
    CHAT_QUOTA_WHITELIST_KERBEROS: str = ""

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "grok-4.3"
    GROQ_EXTRACT_MODEL: str = "grok-3-mini"  # cheap model for query parsing
    MAX_ANSWER_TOKENS: int = 1024

    MAX_TOOL_ROUNDS: int = 1
    CHAT_TOP_K: int = 8
    CHAT_MAX_HISTORY_TURNS: int = 6
    CHAT_MAX_MESSAGE_LENGTH: int = 2000
    HISTORY_TOKEN_BUDGET: int = 800

    LLM_CACHE_TTL: int = 90
    EMBEDDING_CACHE_TTL: int = 86400

    # Per-tool output token caps (approximate char counts; 1 token ≈ 4 chars)
    TOKEN_CAP_SEARCH_PAPERS: int = 4000
    TOKEN_CAP_FACULTY_PROFILE: int = 2000
    TOKEN_CAP_PUBLICATION_STATS: int = 1500
    TOKEN_CAP_DEPARTMENT_PROFILE: int = 2500
    TOKEN_CAP_LIST_DEPARTMENTS: int = 3000
    TOKEN_CAP_FACULTY_EXPERTISE: int = 2000
    TOKEN_CAP_INTERDISCIPLINARY: int = 2000
    TOKEN_CAP_TOP_FACULTY: int = 3000
    TOKEN_CAP_DEFAULT: int = 1500

    # Context budget: chars reserved for the answer generation
    CONTEXT_ANSWER_RESERVE: int = 4096


settings = Settings()
