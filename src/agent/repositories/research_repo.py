"""Research document repository — async methods over MongoDB ResearchMetaDataScopus.

Mock seam for tests: tools depend on this layer, not the raw Motor client.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


class ResearchRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._coll = db["researchmetadatascopus"]

    # ── Hydrate from OpenSearch mongo_ids ──

    async def find_by_ids(self, ids: list[str], fields: dict[str, int] | None = None) -> list[dict[str, Any]]:
        from bson import ObjectId

        oids = [ObjectId(i) for i in ids if ObjectId.is_valid(i)]
        if not oids:
            return []
        projection = fields or {
            "title": 1, "abstract": 1, "authors": 1,
            "publication_year": 1, "document_type": 1,
            "field_associated": 1, "citation_count": 1, "link": 1,
            "document_scopus_id": 1, "document_eid": 1,
        }
        cursor = self._coll.find({"_id": {"$in": oids}}, projection)
        docs = await cursor.to_list(length=len(oids))
        doc_map = {str(d["_id"]): d for d in docs}
        return [doc_map[i] for i in [str(o) for o in oids] if i in doc_map]

    # ── Publication stats ──

    async def count_documents(self, match: dict) -> int:
        return await self._coll.count_documents(match)

    async def aggregate(self, pipeline: list[dict]) -> list[dict[str, Any]]:
        cursor = self._coll.aggregate(pipeline)
        return await cursor.to_list(length=100)

    async def find_top_cited(self, match: dict, limit: int = 5) -> list[dict[str, Any]]:
        cursor = (
            self._coll.find(match, {"title": 1, "publication_year": 1, "citation_count": 1})
            .sort("citation_count", -1)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    # ── Faculty profile: parallel aggregations ──

    async def faculty_publication_stats(self, match: dict) -> dict[str, Any] | None:
        import asyncio

        total, by_year, top_subjects, top_fields, top_papers = await asyncio.gather(
            self.count_documents(match),
            self.aggregate([
                {"$match": match},
                {"$group": {"_id": "$publication_year", "count": {"$sum": 1}}},
                {"$sort": {"_id": -1}},
                {"$limit": 8},
            ]),
            self.aggregate([
                {"$match": match},
                {"$unwind": "$subject_area"},
                {"$group": {"_id": "$subject_area", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 8},
            ]),
            self.aggregate([
                {"$match": match},
                {"$group": {"_id": "$field_associated", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 5},
            ]),
            self.find_top_cited(match, limit=5),
        )

        if total == 0:
            return None

        return {
            "total_papers": total,
            "papers_by_recent_year": [{"year": b["_id"], "count": b["count"]} for b in by_year],
            "top_subject_areas": [
                {"subject": b["_id"], "papers": b["count"]}
                for b in top_subjects if b["_id"]
            ],
            "top_fields": [
                {"field": b["_id"], "papers": b["count"]}
                for b in top_fields if b["_id"]
            ],
            "most_cited_papers": [
                {"title": p.get("title"), "year": p.get("publication_year"), "citations": p.get("citation_count")}
                for p in top_papers
            ],
        }

    # ── Department stats ──

    async def department_stats(self, match: dict) -> dict[str, Any]:
        import asyncio

        total, by_year, by_type = await asyncio.gather(
            self.count_documents(match),
            self.aggregate([
                {"$match": match},
                {"$group": {"_id": "$publication_year", "count": {"$sum": 1}}},
                {"$sort": {"_id": -1}},
                {"$limit": 10},
            ]),
            self.aggregate([
                {"$match": match},
                {"$group": {"_id": "$document_type", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 8},
            ]),
        )
        return {
            "total_papers": total,
            "papers_by_recent_year": [{"year": b["_id"], "count": b["count"]} for b in by_year],
            "papers_by_type": [
                {"type": b["_id"], "count": b["count"]}
                for b in by_type if b["_id"]
            ],
        }

    # ── Papers grouped by kerberos (for department attribution) ──

    async def papers_by_kerberos(self, base_match: dict) -> list[dict[str, Any]]:
        """Returns [{_id: kerberos_prefix, count: int}] for papers that have a kerberos field."""
        match = {
            **base_match,
            "kerberos": {"$exists": True, "$nin": [None, ""]},
        }
        return await self.aggregate([
            {"$match": match},
            {"$group": {"_id": "$kerberos", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 50000},
        ])

    # ── Global publication stats ──

    async def global_stats(self, base_match: dict, dimension: str, sort_field: str, limit: int = 25) -> tuple[int, list[dict]]:
        import asyncio

        total, buckets = await asyncio.gather(
            self.count_documents(base_match),
            self.aggregate([
                {"$match": base_match},
                {"$group": {"_id": dimension, "count": {"$sum": 1}}},
                {"$sort": {sort_field: -1}},
                {"$limit": limit},
            ]),
        )
        return total, buckets

    # ── Research trends (ID-based — caller uses retriever to find IDs) ──

    async def trend_by_ids(
        self, paper_ids: list[str], year_from: int | None, year_to: int | None
    ) -> list[dict[str, Any]]:
        """Aggregate paper counts per year for a given set of MongoDB IDs."""
        from bson import ObjectId

        oids = [ObjectId(i) for i in paper_ids if ObjectId.is_valid(str(i))]
        if not oids:
            return []
        match: dict[str, Any] = {"_id": {"$in": oids}}
        if year_from or year_to:
            yr: dict[str, int] = {}
            if year_from:
                yr["$gte"] = year_from
            if year_to:
                yr["$lte"] = year_to
            match["publication_year"] = yr
        return await self.aggregate([
            {"$match": match},
            {"$group": {"_id": "$publication_year", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ])

    async def kerberos_counts_for_ids(
        self, paper_ids: list[str], base_match: dict
    ) -> list[dict[str, Any]]:
        """Group papers by kerberos, restricted to a specific set of paper IDs.
        Used for topic-filtered department publication stats.
        """
        from bson import ObjectId

        oids = [ObjectId(i) for i in paper_ids if ObjectId.is_valid(str(i))]
        if not oids:
            return []
        match = {
            **base_match,
            "_id": {"$in": oids},
            "kerberos": {"$exists": True, "$nin": [None, ""]},
        }
        return await self.aggregate([
            {"$match": match},
            {"$group": {"_id": "$kerberos", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 50000},
        ])

    # ── Interdisciplinary papers ──

    async def find_interdisciplinary_papers(
        self, fields: list[str], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Stub fallback — the tool uses the OpenSearch retriever instead of this method.
        Kept for interface compatibility; returns empty so the tool falls back gracefully.
        """
        return []
