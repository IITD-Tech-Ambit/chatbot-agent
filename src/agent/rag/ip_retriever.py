"""Hybrid IP retriever: BM25 + kNN over the `ip_documents` OpenSearch index,
hydrated from MongoDB `ipmetadatas`.

Mirrors agent.rag.retriever (papers) but shaped for patents/IP: title/abstract/
field_of_invention BM25, nested inventor matching, and a kNN arm over the shared
BGE `embedding` field. Structured filters (type_of_ip, year range, country,
department, inventor, classification prefix) are applied as OpenSearch filter
clauses so they never affect relevance scoring.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from opensearchpy import AsyncOpenSearch

from agent.rag.embeddings import EmbeddingClient

if TYPE_CHECKING:
    from agent.repositories.ip_repo import IpRepository

logger = logging.getLogger(__name__)


def _case_variants(value: str) -> list[str]:
    """Keyword fields are case-sensitive; try common casings of a term."""
    v = value.strip()
    if not v:
        return []
    variants = {v, v.lower(), v.upper(), v.title(), v.capitalize()}
    return list(variants)


def _build_filters(
    *,
    type_of_ip: str | None,
    year_from: int | None,
    year_to: int | None,
    country: str | None,
    department: str | None,
    inventor: str | None,
    classification_prefix: str | None,
) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []

    if year_from or year_to:
        rng: dict[str, int] = {}
        if year_from:
            rng["gte"] = year_from
        if year_to:
            rng["lte"] = year_to
        filters.append({"range": {"publication_year": rng}})

    if type_of_ip:
        filters.append({"terms": {"type_of_ip": _case_variants(type_of_ip)}})

    if country:
        filters.append({"terms": {"country": _case_variants(country)}})

    if department:
        filters.append({
            "match": {"department_name": {"query": department, "operator": "and"}}
        })

    if classification_prefix:
        filters.append({"prefix": {"classification": classification_prefix.strip().upper()}})

    if inventor:
        filters.append({
            "bool": {
                "should": [
                    {"nested": {
                        "path": "inventors",
                        "query": {"match": {"inventors.name": inventor}},
                    }},
                    {"match": {"inventor_names": inventor}},
                    {"term": {"inventor_kerberos": inventor.lower().strip()}},
                ],
                "minimum_should_match": 1,
            }
        })

    return filters


def _build_ip_body(
    query: str,
    embedding: list[float],
    k: int,
    filters: list[dict[str, Any]],
) -> dict[str, Any]:
    knn_k = max(k * 5, 50)
    if query and query.strip():
        should = [
            {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "title^4",
                        "title.standard^3",
                        "abstract^2",
                        "abstract.standard^1.5",
                        "field_of_invention^2",
                    ],
                    "type": "most_fields",
                    "fuzziness": "AUTO",
                }
            },
            {
                "multi_match": {
                    "query": query,
                    "fields": ["title^5", "abstract^3"],
                    "type": "phrase",
                    "boost": 4.0,
                }
            },
            {
                "nested": {
                    "path": "inventors",
                    "score_mode": "max",
                    "query": {"match": {"inventors.name": {"query": query, "boost": 2}}},
                }
            },
            {"match": {"inventor_names": {"query": query, "boost": 1.5}}},
            {"knn": {"embedding": {"vector": embedding, "k": knn_k}}},
        ]
        inner: dict[str, Any] = {"bool": {"should": should, "minimum_should_match": 1}}
    else:
        inner = {"match_all": {}}

    return {
        "size": k,
        "_source": ["mongo_id"],
        "query": {"bool": {"must": [inner], "filter": filters}},
    }


def _abstract_snippet(abstract: str | None, max_chars: int) -> str:
    text = abstract or ""
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


class IpRetriever:
    def __init__(
        self,
        opensearch: AsyncOpenSearch,
        index_name: str,
        ip_repo: "IpRepository",
        embedding_client: EmbeddingClient,
        top_k: int = 8,
    ) -> None:
        self._os = opensearch
        self._index = index_name
        self._repo = ip_repo
        self._embed = embedding_client
        self._top_k = top_k

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        type_of_ip: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        country: str | None = None,
        department: str | None = None,
        inventor: str | None = None,
        classification_prefix: str | None = None,
        abstract_max_chars: int = 200,
    ) -> list[dict[str, Any]]:
        k = top_k or self._top_k

        embedding: list[float] = []
        if query and query.strip():
            embedding = await self._embed.embed_query(query)

        filters = _build_filters(
            type_of_ip=type_of_ip,
            year_from=year_from,
            year_to=year_to,
            country=country,
            department=department,
            inventor=inventor,
            classification_prefix=classification_prefix,
        )
        body = _build_ip_body(query, embedding, k, filters)

        response = await self._os.search(index=self._index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            logger.debug("IP retrieve: 0 hits for query=%r", query)
            return []

        mongo_ids = [
            h["_source"]["mongo_id"]
            for h in hits
            if h.get("_source", {}).get("mongo_id")
        ]
        docs = await self._repo.find_by_ids(mongo_ids)

        if query and len(docs) > 1:
            doc_texts = [
                f"{d.get('title', '')} {(d.get('abstract') or '')[:400]}"
                for d in docs
            ]
            ranked_pairs = await self._embed.rerank(query, doc_texts)
            if ranked_pairs:
                docs = [docs[idx] for idx, _ in ranked_pairs]

        dept_ids = {d.get("department") for d in docs if d.get("department")}
        dept_name_map = await self._repo.resolve_department_names(dept_ids)

        results: list[dict[str, Any]] = []
        for i, doc in enumerate(docs):
            inventors = [
                {
                    "name": inv.get("name"),
                    "is_faculty": bool(inv.get("is_faculty")),
                    "kerberos": inv.get("kerberos"),
                }
                for inv in (doc.get("inventors") or [])
                if inv.get("name")
            ][:8]

            dept_id = doc.get("department")
            results.append({
                "index": i + 1,
                "id": str(doc.get("_id", "")),
                "application_number": doc.get("application_number"),
                "title": doc.get("title", ""),
                "abstract": _abstract_snippet(doc.get("abstract"), abstract_max_chars),
                "type_of_ip": doc.get("type_of_ip"),
                "field_of_invention": doc.get("field_of_invention"),
                "classification": doc.get("classification") or [],
                "inventors": inventors,
                "country": doc.get("country"),
                "filing_date": doc.get("filing_date"),
                "publication_date": doc.get("publication_date"),
                "publication_year": doc.get("publication_year"),
                "department": dept_name_map.get(dept_id) if dept_id else None,
            })
        return results
