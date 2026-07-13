"""RAG retriever: hybrid BM25 + kNN over OpenSearch, hydrated from MongoDB.

Retrieval architecture
──────────────────────
The only reliable paper → faculty linkage is the `kerberos` keyword field that
the indexer stamps on every paper it attributes to an IIT Delhi faculty member:

    paper.kerberos  ──►  Faculty.email prefix  ──►  Faculty.department

We NEVER search on `field_associated` (Scopus-derived and inconsistent) or
`author_names` (flat copy of all co-authors, not just IIT Delhi ones).

Two execution paths
───────────────────
A. **Kerberos-filtered path** (author or dept resolved):
   Use kerberos as a hard `bool.filter` so only that faculty's / department's
   papers are returned.  Within the filter, rank by topic relevance (BM25 +
   kNN) so multi-hop queries ("papers by Singh about bridge converter") rank
   the most relevant papers first.  Falls back to content path if the filter
   yields zero hits.

B. **Content path** (no kerberos resolved):
   `bool.should` over BM25 (`most_fields` type for better abstract coverage)
   and kNN.  `most_fields` sums scores across title + abstract rather than
   taking only the best-matching field, which is critical for queries where
   the relevant terms appear only in the abstract.

Query parsing
─────────────
Faculty name and department are extracted by a cheap xAI LLM (QueryParser)
rather than hardcoded regex, so any natural-language phrasing is handled
robustly.  QueryParser results are cached in-process so repeated queries are
free.

Reranking
─────────
Both paths feed into a BGE cross-encoder reranker (POST /rerank on the
embedding service) to fix first-stage ranking errors — especially important
for multi-relevant and multi-hop queries.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from opensearchpy import AsyncOpenSearch

from agent.rag.embeddings import EmbeddingClient
from agent.repositories.research_repo import ResearchRepository

if TYPE_CHECKING:
    from agent.rag.query_parser import ParsedQuery, QueryParser
    from agent.repositories.faculty_repo import FacultyRepository

logger = logging.getLogger(__name__)

def _build_kerberos_filtered_body(
    query: str,
    embedding: list[float],
    k: int,
    kerberoses: list[str],
) -> dict[str, Any]:
    """Hard-filter on kerberos; rank by topic content within that faculty/dept set."""
    knn_k = max(k * 10, 100)
    return {
        "size": k,
        "_source": ["mongo_id"],
        "query": {
            "bool": {
                "filter": [{"terms": {"kerberos": kerberoses}}],
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "title^4",
                                "title.standard^3",
                                "abstract^3",
                                "abstract.standard^2",
                            ],
                            "type": "most_fields",
                            "fuzziness": "AUTO",
                        }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": embedding,
                                "k": knn_k,
                            }
                        }
                    },
                ],
            }
        },
    }

def _build_content_body(
    query: str,
    embedding: list[float],
    k: int,
) -> dict[str, Any]:
    """Content-only hybrid BM25 + kNN retrieval.

    Uses `most_fields` so abstract-only terms rank — the scorer sums
    contributions from title AND abstract rather than taking the max.
    Title boost is moderated (4×) so abstract matches are competitive.
    """
    knn_k = max(k * 5, 50)
    return {
        "size": k,
        "_source": ["mongo_id"],
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "title^4",
                                "title.standard^3",
                                "abstract^3",
                                "abstract.standard^2",
                                "abstract.shingles^1.5",
                            ],
                            "type": "most_fields",
                            "fuzziness": "AUTO",
                            "minimum_should_match": "1",
                        }
                    },
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^5", "abstract^3"],
                            "type": "phrase",
                            "boost": 5.0,
                        }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": embedding,
                                "k": knn_k,
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
    }

def _kerberoses_from_faculty(faculty: list[dict]) -> list[str]:
    result: list[str] = []
    for f in faculty:
        email = f.get("email", "")
        if email and "@" in email:
            k = email.split("@")[0].strip().lower()
            if k:
                result.append(k)
    return result

class Retriever:
    def __init__(
        self,
        opensearch: AsyncOpenSearch,
        index_name: str,
        research_repo: ResearchRepository,
        embedding_client: EmbeddingClient,
        top_k: int = 8,
        faculty_repo: FacultyRepository | None = None,
        query_parser: QueryParser | None = None,
    ) -> None:
        self._os = opensearch
        self._index = index_name
        self._repo = research_repo
        self._embed = embedding_client
        self._top_k = top_k
        self._faculty = faculty_repo
        self._parser = query_parser

    async def _resolve_author_kerberoses(self, faculty_name: str | None) -> list[str]:
        if not self._faculty or not faculty_name:
            return []
        tokens = faculty_name.split()
        if not tokens:
            return []
        try:
            faculty = await self._faculty.compound_name_search(tokens, limit=5)
        except Exception as exc:
            logger.debug("Faculty compound_name_search failed: %s", exc)
            return []
        return _kerberoses_from_faculty(faculty)

    async def _resolve_dept_kerberoses(self, departments: tuple[str, ...]) -> list[str]:
        if not self._faculty or not departments:
            return []
        kerberoses: list[str] = []
        for dept_name in departments:
            try:
                dept = await self._faculty.find_department(dept_name.strip())
            except Exception as exc:
                logger.debug("Department lookup failed for %r: %s", dept_name, exc)
                continue
            if not dept:
                logger.debug("No dept matched: %r", dept_name)
                continue
            try:
                faculty_docs = await self._faculty.find_faculty_by_department_id(dept["_id"])
                kerberoses.extend(_kerberoses_from_faculty(faculty_docs))
                logger.debug("Dept %r → %d faculty kerberos", dept.get("name"), len(faculty_docs))
            except Exception as exc:
                logger.debug("Faculty by dept_id failed for %r: %s", dept_name, exc)
        return kerberoses

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        abstract_max_chars: int = 150,
    ) -> list[dict[str, Any]]:
        k = top_k or self._top_k

        async def _null_parse() -> ParsedQuery:
            from agent.rag.query_parser import _NULL
            return _NULL

        embedding, parsed = await asyncio.gather(
            self._embed.embed_query(query),
            self._parser.extract(query) if self._parser else _null_parse(),
        )

        author_kerberoses, dept_kerberoses = await asyncio.gather(
            self._resolve_author_kerberoses(parsed.faculty_name),
            self._resolve_dept_kerberoses(parsed.departments),
        )

        author_set = set(author_kerberoses)
        dept_set = set(dept_kerberoses)

        if author_set and dept_set:
            intersected = author_set & dept_set
            all_kerberoses = list(intersected) if intersected else list(author_set)
        elif author_set:
            all_kerberoses = list(author_set)
        elif dept_set:
            all_kerberoses = list(dept_set)
        else:
            all_kerberoses = []

        if all_kerberoses:
            logger.debug(
                "Kerberos filter: %d ids (author=%s, %d dept, intersection=%s) query=%r",
                len(all_kerberoses),
                parsed.faculty_name,
                len(dept_kerberoses),
                bool(author_set and dept_set),
                query,
            )

        hits: list[dict] = []

        if all_kerberoses:
            body = _build_kerberos_filtered_body(query, embedding, k, all_kerberoses)
            response = await self._os.search(index=self._index, body=body)
            hits = response.get("hits", {}).get("hits", [])

            if not hits:
                logger.debug("Kerberos filter 0 hits; falling back to content search")

        if not hits:
            body = _build_content_body(query, embedding, k)
            response = await self._os.search(index=self._index, body=body)
            hits = response.get("hits", {}).get("hits", [])

        if not hits:
            logger.debug("retrieve: 0 hits for query=%r", query)
            return []

        mongo_ids = [
            h["_source"]["mongo_id"]
            for h in hits
            if h.get("_source", {}).get("mongo_id")
        ]
        docs = await self._repo.find_by_ids(mongo_ids)

        if len(docs) > 1:
            doc_texts = [
                f"{d.get('title', '')} {(d.get('abstract') or '')[:400]}"
                for d in docs
            ]
            ranked_pairs = await self._embed.rerank(query, doc_texts)
            if ranked_pairs:
                docs = [docs[idx] for idx, _ in ranked_pairs]

        results: list[dict[str, Any]] = []
        for i, doc in enumerate(docs):
            abstract = doc.get("abstract") or ""
            if len(abstract) > abstract_max_chars:
                abstract = abstract[:abstract_max_chars] + "..."

            authors = doc.get("authors") or []
            author_names = [
                a.get("author_name", "") for a in authors if a.get("author_name")
            ]

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
                "kerberos": doc.get("kerberos"),
            })
        return results
