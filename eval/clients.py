"""Service connectivity and retriever factory for live evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from agent.config import settings
from agent.data import mongo, opensearch, redis as redis_mod
from agent.rag.embeddings import EmbeddingClient
from agent.rag.query_parser import QueryParser
from agent.rag.retriever import Retriever
from agent.repositories.faculty_repo import FacultyRepository
from agent.repositories.research_repo import ResearchRepository

logger = logging.getLogger(__name__)


@dataclass
class ServiceStatus:
    opensearch: bool = False
    mongodb: bool = False
    redis: bool = False
    embedding: bool = False
    chatbot: bool = False

    @property
    def retrieval_ready(self) -> bool:
        return self.opensearch and self.mongodb and self.embedding

    @property
    def e2e_ready(self) -> bool:
        return self.chatbot and self.retrieval_ready

    def as_dict(self) -> dict[str, bool]:
        return {
            "opensearch": self.opensearch,
            "mongodb": self.mongodb,
            "redis": self.redis,
            "embedding": self.embedding,
            "chatbot": self.chatbot,
        }


async def check_services(chatbot_url: str | None = None) -> ServiceStatus:
    status = ServiceStatus()
    chat_url = chatbot_url or f"http://{settings.HOST}:{settings.PORT}"

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.OPENSEARCH_NODE.rstrip('/')}/")
            status.opensearch = r.is_success
    except Exception:
        pass

    try:
        db = await mongo.connect(settings.MONGODB_URI)
        await db.command("ping")
        status.mongodb = True
        await mongo.close()
    except Exception:
        pass

    try:
        rc = await redis_mod.connect(settings.REDIS_URL)
        status.redis = await rc.ping()
        await redis_mod.close()
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.EMBEDDING_SERVICE_URL.rstrip('/')}/health")
            status.embedding = r.is_success
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{chat_url.rstrip('/')}/health")
            status.chatbot = r.is_success
    except Exception:
        pass

    return status


class RetrieverSession:
    """Async context manager wiring the production retriever stack."""

    def __init__(self, top_k: int | None = None) -> None:
        self._top_k = top_k or settings.CHAT_TOP_K
        self._db = None
        self._os = None
        self._redis = None
        self.retriever: Retriever | None = None

    async def __aenter__(self) -> Retriever:
        self._db = await mongo.connect(settings.MONGODB_URI)
        self._os = await opensearch.connect(
            settings.OPENSEARCH_NODE,
            settings.OPENSEARCH_USER,
            settings.OPENSEARCH_PASSWORD,
            verify_certs=settings.OPENSEARCH_VERIFY_CERTS,
            use_ssl=settings.OPENSEARCH_USE_SSL,
        )
        self._redis = await redis_mod.connect(settings.REDIS_URL)
        research_repo = ResearchRepository(self._db)
        faculty_repo = FacultyRepository(self._db)
        embedding_client = EmbeddingClient(
            base_url=settings.EMBEDDING_SERVICE_URL,
            redis_client=self._redis,
            timeout_ms=settings.EMBEDDING_TIMEOUT_MS,
            cache_ttl=settings.EMBEDDING_CACHE_TTL,
        )
        query_parser = QueryParser(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_EXTRACT_MODEL,
        ) if settings.GROQ_API_KEY else None
        self.retriever = Retriever(
            opensearch=self._os,
            index_name=settings.OPENSEARCH_INDEX,
            research_repo=research_repo,
            embedding_client=embedding_client,
            top_k=self._top_k,
            faculty_repo=faculty_repo,
            query_parser=query_parser,
        )
        return self.retriever

    async def __aexit__(self, *args: Any) -> None:
        await opensearch.close()
        await redis_mod.close()
        await mongo.close()


class MockRetriever:
    """Offline retriever using BM25-like token overlap against corpus docs."""

    def __init__(self, corpus_docs: list[dict], top_k: int = 8) -> None:
        self._docs = corpus_docs
        self._top_k = top_k

    async def retrieve(self, query: str, top_k: int | None = None, **kwargs: Any) -> list[dict]:
        k = top_k or self._top_k
        if not query or not query.strip():
            return []

        q_tokens = set(_mock_tokenize(query))
        scored: list[tuple[float, dict]] = []
        for doc in self._docs:
            text = f"{doc.get('title', '')} {doc.get('abstract', '')}".lower()
            d_tokens = set(_mock_tokenize(text))
            if not q_tokens:
                continue
            overlap = len(q_tokens & d_tokens) / len(q_tokens)
            title_bonus = sum(0.2 for t in q_tokens if t in (doc.get("title") or "").lower())
            score = overlap + title_bonus
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for i, (_, doc) in enumerate(scored[:k]):
            results.append({
                "index": i + 1,
                "id": doc["mongo_id"],
                "title": doc.get("title", ""),
                "abstract": (doc.get("abstract") or "")[:150],
                "authors": [a.get("name", "") for a in (doc.get("authors") or [])[:5]],
                "publication_year": doc.get("publication_year"),
                "document_type": doc.get("document_type"),
                "field_associated": doc.get("field_associated"),
                "citation_count": doc.get("citation_count", 0),
                "kerberos": doc.get("kerberos"),
            })
        return results


def _mock_tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]{3,}", text.lower())
