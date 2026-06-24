"""find_faculty_for_topic tool — HTTP call to search-api, then enrich from MongoDB.

When a department is specified but the search API returns no faculty from that
department, falls back to a direct MongoDB lookup of top faculty by h-index.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class FindFacultyArgs(BaseModel):
    topic: str = Field(description="Research topic to find faculty for")
    department: Optional[str] = Field(
        default=None,
        description="Optional department name to filter results (e.g. 'Chemical Engineering', 'Computer Science'). "
                    "Use when the user asks for faculty from a specific department.",
    )


def _serialize_faculty_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Convert a MongoDB faculty document into a consistent output dict."""
    resolved_dept = None
    if isinstance(doc.get("department"), dict):
        resolved_dept = doc["department"].get("name")

    return {
        "name": f"{doc.get('title', '')} {doc.get('firstName', '')} {doc.get('lastName', '')}".strip(),
        "department": resolved_dept,
        "designation": doc.get("designation"),
        "email": doc.get("email"),
        "expertise": (doc.get("brief_expertise") or doc.get("expertise") or [])[:6],
        "h_index": doc.get("h_index"),
        "citation_count": doc.get("citation_count"),
    }


@tool(args_schema=FindFacultyArgs)
async def find_faculty_for_topic(topic: str, department: str | None = None) -> str:
    """Find IIT Delhi professors who work on a given research topic, with department, email, and paper count.
    Pass the department parameter when the user asks about a specific department."""
    from agent.tools._registry import get_config, get_faculty_repo

    cfg = get_config()
    faculty_repo = get_faculty_repo()

    # Try search API first
    search_api_ok = True
    agg: dict[str, Any] = {"departments": [], "total_matching_papers": 0}

    url = f"{cfg.SEARCH_API_URL.rstrip('/')}/api/v1/search/faculty-for-query"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url, params={"query": topic})
            resp.raise_for_status()
            agg = resp.json()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        search_api_ok = False

    # Flatten search API results
    flat: list[dict[str, Any]] = []
    for dept in agg.get("departments", []):
        for f in dept.get("faculty", []):
            flat.append({**f, "department": dept["name"]})
    flat.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    # Check if any faculty from the requested department exist in search results
    dept_in_search = False
    if department:
        dept_lower = department.lower().strip()
        dept_matches = [f for f in flat if dept_lower in f.get("department", "").lower()]
        dept_in_search = len(dept_matches) > 0

        if dept_in_search:
            other = [f for f in flat if dept_lower not in f.get("department", "").lower()]
            flat = dept_matches + other

    # If department was requested but NOT found in search API results,
    # fall back to direct MongoDB lookup for that department's faculty
    if department and not dept_in_search:
        dept_faculty = await faculty_repo.find_top_faculty_by_department(department, limit=10)
        if dept_faculty:
            faculty_list = [_serialize_faculty_doc(doc) for doc in dept_faculty]
            result = {
                "topic": topic,
                "department_filter": department,
                "total_matching_papers": agg.get("total_matching_papers", 0),
                "faculty": faculty_list,
                "source": "department_directory",
                "note": (
                    f"These are the top faculty from the {department} department ranked by h-index. "
                    f"They may not all have papers specifically on '{topic}' in the indexed corpus."
                ),
            }
            return json.dumps(result, default=str)
        else:
            return json.dumps({
                "topic": topic,
                "department_filter": department,
                "faculty": [],
                "error": f"No department matching '{department}' found in the database.",
            })

    # Standard path: enrich search API results from MongoDB
    top = flat[:10]
    if not top:
        if not search_api_ok:
            return json.dumps({"topic": topic, "faculty": [], "error": "Search API unavailable"})
        return json.dumps({"topic": topic, "faculty": [], "total_matching_papers": 0})

    expert_ids = [f["author_id"] for f in top if f.get("author_id")]
    faculty_docs = await faculty_repo.find_by_expert_ids(expert_ids)
    by_expert_id = {d["expert_id"]: d for d in faculty_docs}

    faculty_list = []
    for f in top:
        doc = by_expert_id.get(f.get("author_id"))
        expertise = []
        if doc:
            expertise = (doc.get("brief_expertise") or doc.get("expertise") or [])[:6]

        resolved_dept = None
        if doc and isinstance(doc.get("department"), dict):
            resolved_dept = doc["department"].get("name")
        if not resolved_dept:
            resolved_dept = f.get("department")

        email = doc.get("email") if doc else None
        kerberos = (email or "").split("@")[0].lower() or None

        faculty_list.append({
            "name": (
                f"{doc.get('title', '')} {doc.get('firstName', '')} {doc.get('lastName', '')}".strip()
                if doc else f.get("name", "")
            ),
            "department": resolved_dept,
            "designation": doc.get("designation") if doc else None,
            "email": email,
            "kerberos": kerberos,
            "profile_url": f"/faculty/{kerberos}" if kerberos else None,
            "expertise": expertise,
            "relevant_paper_count": f.get("paper_count", 0),
            "h_index": doc.get("h_index") if doc else None,
        })

    result = {
        "topic": topic,
        "total_matching_papers": agg.get("total_matching_papers", 0),
        "faculty": faculty_list,
    }
    if department:
        result["department_filter"] = department

    return json.dumps(result, default=str)
