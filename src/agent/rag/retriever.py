"""RAG retriever: hybrid BM25 + kNN search in OpenSearch, hydrate from MongoDB.

Ports ragService.js with abstract truncation for ctx=2048 budget.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from opensearchpy import AsyncOpenSearch

from agent.rag.embeddings import EmbeddingClient
from agent.repositories.research_repo import ResearchRepository

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        opensearch: AsyncOpenSearch,
        index_name: str,
        research_repo: ResearchRepository,
        embedding_client: EmbeddingClient,
        top_k: int = 8,
    ) -> None:
        self._os = opensearch
        self._index = index_name
        self._repo = research_repo
        self._embed = embedding_client
        self._top_k = top_k

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        abstract_max_chars: int = 150,
    ) -> list[dict[str, Any]]:
        k = top_k or self._top_k
        embedding = await self._embed.embed_query(query)

        body = {
            "size": k,
            "_source": ["mongo_id"],
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "title^5",
                                    
                                    "abstract^2",
                                    "subject_area^2",
                                    "field_associated^2",
                                    "authors.author_name^3",
                                ],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                            }
                        },
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "abstract^1.5"],
                                "type": "phrase",
                                "boost": 3.0,
                            }
                        },
                        {
                            "knn": {
                                "embedding": {
                                    "vector": embedding,
                                    "k": max(k * 5, 50),
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
        }

        response = await self._os.search(index=self._index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            return []

        mongo_ids = [h["_source"]["mongo_id"] for h in hits if h.get("_source", {}).get("mongo_id")]
        docs = await self._repo.find_by_ids(mongo_ids)

        results = []
        for i, doc in enumerate(docs):
            abstract = doc.get("abstract") or ""
            if len(abstract) > abstract_max_chars:
                abstract = abstract[:abstract_max_chars] + "..."

            authors = doc.get("authors") or []
            author_names = [a.get("author_name", "") for a in authors if a.get("author_name")]

            results.append({
                "index": i + 1,
                "id": str(doc.get("_id", "")),
                "title": doc.get("title", ""),
                "abstract": abstract,
                "authors": author_names[:5],
                "publication_year": doc.get("publication_year"),
                "document_type": doc.get("document_type"),
                "field_associated": doc.get("field_associated"),
                "citation_count": doc.get("citation_count", 0),
                "link": doc.get("link"),
                "document_scopus_id": doc.get("document_scopus_id"),
                "document_eid": doc.get("document_eid"),
            })
        return results
