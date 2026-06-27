"""get_department_profile tool — full overview of an IIT Delhi department."""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.tools._registry import get_faculty_repo, get_research_repo

logger = logging.getLogger(__name__)


class DepartmentProfileArgs(BaseModel):
    department: str = Field(
        description="Department, school, or centre name (e.g. 'Computer Science', 'Electrical Engineering', 'School of Biological Sciences')",
        min_length=2,
        max_length=200,
    )


@tool("get_department_profile", args_schema=DepartmentProfileArgs)
async def get_department_profile(department: str) -> str:
    """Get a full overview of an IIT Delhi department including faculty count, top faculty by h-index, and publication statistics."""
    faculty_repo = get_faculty_repo()
    research_repo = get_research_repo()

    dept_doc = await faculty_repo.find_department(department)
    if not dept_doc:
        return json.dumps({"error": f'No department matching "{department}" found at IIT Delhi.'})

    dept_id = dept_doc["_id"]
    dept_name = dept_doc.get("name", department)

    faculty_docs = await faculty_repo.find_faculty_by_department_id(dept_id)
    faculty_count = len(faculty_docs)

    top_faculty = await faculty_repo.find_top_faculty_by_department(dept_name, limit=5)
    top_faculty_list = [
        {
            "name": f"{d.get('title', '')} {d.get('firstName', '')} {d.get('lastName', '')}".strip(),
            "email": d.get("email", ""),
            "designation": d.get("designation", ""),
            "h_index": d.get("h_index"),
            "citation_count": d.get("citation_count"),
            "expertise": (d.get("expertise") or [])[:4],
        }
        for d in top_faculty
    ]

    scopus_ids = [str(s) for doc in faculty_docs for s in (doc.get("scopus_id") or [])]
    kerberos_list = [
        (doc.get("email") or "").split("@")[0].lower()
        for doc in faculty_docs
        if doc.get("email")
    ]
    or_clauses: list[dict] = []
    if kerberos_list:
        or_clauses.append({"kerberos": {"$in": kerberos_list}})
    if scopus_ids:
        or_clauses.append({"authors.author_id": {"$in": scopus_ids}})

    pub_stats: dict = {}
    if or_clauses:
        pub_stats = await research_repo.department_stats({"$or": or_clauses})

    result = {
        "department": {
            "name": dept_name,
            "code": dept_doc.get("code", ""),
            "category": dept_doc.get("category", ""),
        },
        "faculty_count": faculty_count,
        "top_faculty": top_faculty_list,
        "publication_stats": pub_stats,
    }

    return json.dumps(result, default=str)
