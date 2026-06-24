"""Faculty repository — async methods over MongoDB Faculty + Department collections.

This is the mock seam for tests: tools depend on this layer, not the raw Motor client.
"""

from __future__ import annotations

import re
import logging
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


class FacultyRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._faculty = db["faculties"]
        self._departments = db["departments"]

    # ── Faculty text search ──

    async def text_search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        cursor = (
            self._faculty.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}},
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(limit)
        )
        results = await cursor.to_list(length=limit)
        return await self._populate_department(results)

    async def regex_search(self, tokens: list[str], limit: int = 5) -> list[dict[str, Any]]:
        regexes = [re.compile(re.escape(t), re.IGNORECASE) for t in tokens]
        cursor = self._faculty.find(
            {"$or": [{"firstName": {"$in": regexes}}, {"lastName": {"$in": regexes}}]}
        ).limit(limit)
        results = await cursor.to_list(length=limit)
        return await self._populate_department(results)

    # ── Faculty by expert_id list ──

    async def find_by_expert_ids(self, expert_ids: list[str]) -> list[dict[str, Any]]:
        cursor = self._faculty.find(
            {"expert_id": {"$in": expert_ids}},
            {
                "expert_id": 1, "title": 1, "firstName": 1, "lastName": 1,
                "email": 1, "designation": 1, "department": 1,
                "expertise": 1, "brief_expertise": 1,
                "h_index": 1, "citation_count": 1,
            },
        )
        results = await cursor.to_list(length=len(expert_ids))
        return await self._populate_department(results)

    # ── Department lookup ──

    async def find_department(self, name: str) -> dict[str, Any] | None:
        exact = re.compile(r"^" + re.escape(name) + r"$", re.IGNORECASE)
        doc = await self._departments.find_one(
            {"$or": [{"name": exact}, {"code": exact}]}
        )
        if doc:
            return doc
        regex = re.compile(re.escape(name), re.IGNORECASE)
        return await self._departments.find_one(
            {"$or": [{"name": regex}, {"code": regex}]}
        )

    async def find_faculty_by_department_id(self, dept_id: ObjectId) -> list[dict[str, Any]]:
        cursor = self._faculty.find(
            {"department": dept_id},
            {"email": 1, "scopus_id": 1},
        )
        return await cursor.to_list(length=500)

    async def find_top_faculty_by_department(
        self, department_name: str, limit: int = 10,
    ) -> list[dict[str, Any]]:
        dept = await self.find_department(department_name)
        if not dept:
            return []
        cursor = (
            self._faculty.find(
                {"department": dept["_id"]},
                {
                    "title": 1, "firstName": 1, "lastName": 1,
                    "email": 1, "designation": 1, "department": 1,
                    "expertise": 1, "brief_expertise": 1,
                    "h_index": 1, "citation_count": 1,
                },
            )
            .sort("h_index", -1)
            .limit(limit)
        )
        results = await cursor.to_list(length=limit)
        return await self._populate_department(results)

    # ── Top faculty globally or per-department ──

    async def find_top_faculty_global(
        self,
        sort_by: str = "h_index",
        limit: int = 10,
        department_name: str | None = None,
    ) -> list[dict[str, Any]]:
        allowed_sort_fields = {"h_index", "citation_count"}
        field = sort_by if sort_by in allowed_sort_fields else "h_index"
        match: dict[str, Any] = {field: {"$gt": 0}}
        if department_name:
            dept = await self.find_department(department_name)
            if not dept:
                return []
            match["department"] = dept["_id"]
        cursor = (
            self._faculty.find(
                match,
                {
                    "title": 1, "firstName": 1, "lastName": 1,
                    "email": 1, "designation": 1, "department": 1,
                    "expertise": 1, "h_index": 1, "citation_count": 1,
                },
            )
            .sort(field, -1)
            .limit(limit)
        )
        results = await cursor.to_list(length=limit)
        return await self._populate_department(results)

    async def count_all_faculty(self, department_name: str | None = None) -> int:
        match: dict[str, Any] = {}
        if department_name:
            dept = await self.find_department(department_name)
            if dept:
                match["department"] = dept["_id"]
        return await self._faculty.count_documents(match)

    # ── Kerberos → department name map (for paper attribution) ──

    async def get_kerberos_to_faculty_map(self, kerberoses: list[str]) -> dict[str, dict[str, str]]:
        """Batch-fetch {kerberos: {name, department}} for a list of kerberos IDs."""
        if not kerberoses:
            return {}
        unique = list({k for k in kerberoses if k})
        or_clauses = [{"email": re.compile(f"^{re.escape(k)}@", re.IGNORECASE)} for k in unique]
        cursor = self._faculty.find(
            {"$or": or_clauses},
            {"email": 1, "firstName": 1, "lastName": 1, "title": 1, "department": 1},
        )
        docs = await cursor.to_list(length=len(unique) * 2)

        dept_ids = {d["department"] for d in docs if isinstance(d.get("department"), ObjectId)}
        dept_name_map: dict[ObjectId, str] = {}
        if dept_ids:
            cursor2 = self._departments.find({"_id": {"$in": list(dept_ids)}}, {"name": 1})
            dept_docs = await cursor2.to_list(length=500)
            dept_name_map = {d["_id"]: d.get("name", "") for d in dept_docs}

        result: dict[str, dict[str, str]] = {}
        for doc in docs:
            k = (doc.get("email") or "").split("@")[0].lower().strip()
            if not k or k not in unique:
                continue
            name_parts = [doc.get("title", ""), doc.get("firstName", ""), doc.get("lastName", "")]
            name = " ".join(p for p in name_parts if p).strip()
            dept_id = doc.get("department")
            dept = dept_name_map.get(dept_id, "") if isinstance(dept_id, ObjectId) else ""
            result[k] = {"name": name, "department": dept}
        return result

    async def get_kerberos_to_dept_map(self) -> dict[str, str]:
        """Build {kerberos_prefix: department_name} for all faculty.

        Papers link to faculty via paper.kerberos == email.split('@')[0].lower().
        Faculty link to departments via faculty.department (ObjectId).
        """
        cursor = self._faculty.find({}, {"email": 1, "department": 1})
        docs = await cursor.to_list(length=5000)

        dept_ids = {
            d["department"] for d in docs
            if isinstance(d.get("department"), ObjectId)
        }
        dept_name_map: dict[ObjectId, str] = {}
        if dept_ids:
            cursor2 = self._departments.find({"_id": {"$in": list(dept_ids)}}, {"name": 1})
            dept_docs = await cursor2.to_list(length=500)
            dept_name_map = {d["_id"]: d.get("name", "") for d in dept_docs}

        result: dict[str, str] = {}
        for doc in docs:
            kerberos = (doc.get("email") or "").split("@")[0].lower().strip()
            dept_id = doc.get("department")
            dept_name = dept_name_map.get(dept_id, "") if isinstance(dept_id, ObjectId) else ""
            if kerberos and dept_name:
                result[kerberos] = dept_name
        return result

    # ── List all departments ──

    async def list_all_departments(self, category: str | None = None) -> list[dict[str, Any]]:
        match: dict[str, Any] = {}
        if category:
            match["category"] = re.compile(re.escape(category), re.IGNORECASE)
        cursor = self._departments.find(
            match, {"name": 1, "code": 1, "category": 1}
        ).sort("name", 1)
        return await cursor.to_list(length=500)

    # ── Faculty by expertise ──

    async def find_faculty_by_expertise(
        self, expertise_terms: list[str], limit: int = 15
    ) -> list[dict[str, Any]]:
        or_clauses: list[dict[str, Any]] = []
        for term in expertise_terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            or_clauses.append({"expertise": {"$elemMatch": {"$regex": pattern.pattern, "$options": "i"}}})
            or_clauses.append({"brief_expertise": {"$elemMatch": {"$regex": pattern.pattern, "$options": "i"}}})
        if not or_clauses:
            return []
        cursor = (
            self._faculty.find(
                {"$or": or_clauses},
                {
                    "title": 1, "firstName": 1, "lastName": 1, "email": 1,
                    "designation": 1, "department": 1,
                    "expertise": 1, "brief_expertise": 1,
                    "h_index": 1, "citation_count": 1,
                },
            )
            .sort("h_index", -1)
            .limit(limit)
        )
        results = await cursor.to_list(length=limit)
        return await self._populate_department(results)

    # ── Private helpers ──

    async def _populate_department(self, docs: list[dict]) -> list[dict]:
        dept_ids = {
            d["department"] for d in docs
            if "department" in d and isinstance(d.get("department"), ObjectId)
        }
        if not dept_ids:
            return docs
        cursor = self._departments.find({"_id": {"$in": list(dept_ids)}}, {"name": 1})
        dept_map = {d["_id"]: d for d in await cursor.to_list(length=len(dept_ids))}
        for doc in docs:
            did = doc.get("department")
            if isinstance(did, ObjectId) and did in dept_map:
                doc["department"] = dept_map[did]
        return docs
