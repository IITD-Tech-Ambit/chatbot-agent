"""IP/patent repository — async methods over MongoDB `ipmetadatas` + `departments`.

Mirrors research_repo.py: tools depend on this layer (the mock seam), never the
raw Motor client. Covers direct lookups, inventor lookups, grouped analytics
(department/year/type/country/classification/inventor), and department
ObjectId → name resolution.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

# Non-array grouping dimensions → the Mongo field they map to. `classification`
# and `inventor` are array fields handled separately (they need an $unwind).
_SCALAR_DIMENSIONS: dict[str, str] = {
    "year": "publication_year",
    "publication_year": "publication_year",
    "type": "type_of_ip",
    "type_of_ip": "type_of_ip",
    "country": "country",
    "department": "department",
    "field_of_invention": "field_of_invention",
}

_OUTPUT_KEY = {
    "publication_year": "year",
    "type_of_ip": "type",
    "department": "department",
    "country": "country",
    "field_of_invention": "field_of_invention",
}


class IpRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._coll = db["ipmetadatas"]
        self._departments = db["departments"]

    async def get_by_application_number(self, application_number: str) -> dict[str, Any] | None:
        return await self._coll.find_one({"application_number": application_number})

    async def get_by_id(self, ip_id: str) -> dict[str, Any] | None:
        if not ObjectId.is_valid(str(ip_id)):
            return None
        return await self._coll.find_one({"_id": ObjectId(str(ip_id))})

    async def find_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        oids = [ObjectId(i) for i in ids if ObjectId.is_valid(str(i))]
        if not oids:
            return []
        cursor = self._coll.find({"_id": {"$in": oids}})
        docs = await cursor.to_list(length=len(oids))
        doc_map = {str(d["_id"]): d for d in docs}
        return [doc_map[str(o)] for o in oids if str(o) in doc_map]

    async def search_by_title(self, title: str, limit: int = 5) -> list[dict[str, Any]]:
        """Best-effort title lookup: Mongo text index first, regex fallback."""
        try:
            cursor = (
                self._coll.find(
                    {"$text": {"$search": title}},
                    {"score": {"$meta": "textScore"}},
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(limit)
            )
            docs = await cursor.to_list(length=limit)
            if docs:
                return docs
        except Exception as exc:
            logger.debug("IP title text search failed, falling back to regex: %s", exc)
        regex = re.compile(re.escape(title), re.IGNORECASE)
        cursor = self._coll.find({"title": regex}).limit(limit)
        return await cursor.to_list(length=limit)

    async def find_by_inventor(
        self,
        kerberos: str | None = None,
        name: str | None = None,
        faculty_ref: str | ObjectId | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        or_clauses: list[dict[str, Any]] = []
        if kerberos:
            or_clauses.append({"inventors.kerberos": kerberos.lower().strip()})
        if faculty_ref and ObjectId.is_valid(str(faculty_ref)):
            or_clauses.append({"inventors.faculty_ref": ObjectId(str(faculty_ref))})
        if name:
            regex = re.compile(re.escape(name.strip()), re.IGNORECASE)
            or_clauses.append({"inventors.name": regex})
            or_clauses.append({"inventors.raw_name": regex})
        if not or_clauses:
            return []
        cursor = (
            self._coll.find({"$or": or_clauses})
            .sort("publication_year", -1)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    async def count_documents(self, match: dict) -> int:
        return await self._coll.count_documents(match)

    async def aggregate(self, pipeline: list[dict], length: int = 500) -> list[dict[str, Any]]:
        cursor = self._coll.aggregate(pipeline)
        return await cursor.to_list(length=length)

    async def resolve_department_names(self, dept_ids) -> dict[ObjectId, str]:
        oids: list[ObjectId] = []
        for d in dept_ids:
            if isinstance(d, ObjectId):
                oids.append(d)
            elif isinstance(d, str) and ObjectId.is_valid(d):
                oids.append(ObjectId(d))
        if not oids:
            return {}
        cursor = self._departments.find({"_id": {"$in": oids}}, {"name": 1})
        docs = await cursor.to_list(length=len(oids))
        return {d["_id"]: d.get("name", "") for d in docs}

    async def grouped_counts(
        self, match: dict, dimensions: list[str], limit: int = 300
    ) -> list[dict[str, Any]]:
        """Count IP filings grouped by one or more dimensions.

        Supported dimension keys: year, type, country, department,
        field_of_invention, classification, inventor. Array dimensions
        (classification, inventor) trigger an $unwind. Department ObjectIds are
        resolved to names in the returned rows.
        """
        want_classification = "classification" in dimensions
        want_inventor = "inventor" in dimensions or "faculty" in dimensions

        pipeline: list[dict] = [{"$match": match}]
        if want_classification:
            pipeline.append({"$unwind": "$classification"})
        if want_inventor:
            pipeline.append({"$unwind": "$inventors"})

        group_id: dict[str, Any] = {}
        for dim in dimensions:
            if dim in ("inventor", "faculty"):
                group_id["inventor"] = "$inventors.name"
                group_id["kerberos"] = "$inventors.kerberos"
                group_id["is_faculty"] = "$inventors.is_faculty"
            elif dim == "classification":
                group_id["classification"] = "$classification"
            else:
                field = _SCALAR_DIMENSIONS.get(dim)
                if not field:
                    continue
                group_id[_OUTPUT_KEY[field]] = f"${field}"

        if not group_id:
            return []

        pipeline += [
            {"$group": {"_id": group_id, "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]

        buckets = await self.aggregate(pipeline, length=limit)

        dept_name_map: dict[ObjectId, str] = {}
        if "department" in group_id:
            dept_ids = {
                (b.get("_id") or {}).get("department")
                for b in buckets
                if (b.get("_id") or {}).get("department")
            }
            dept_name_map = await self.resolve_department_names(dept_ids)

        rows: list[dict[str, Any]] = []
        for b in buckets:
            gid = b.get("_id") or {}
            row: dict[str, Any] = {"count": b.get("count", 0)}
            for key, value in gid.items():
                if key == "department":
                    row["department"] = dept_name_map.get(value, "Unknown")
                else:
                    row[key] = value
            rows.append(row)
        return rows
