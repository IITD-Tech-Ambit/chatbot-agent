"""find_faculty_for_topic tool — faculty-for-query via the mesh search client
(gRPC through Envoy in production, HTTP in dev), then enrich from MongoDB.

When a department is specified but the search API returns no faculty from that
department, falls back to a direct MongoDB lookup of top faculty by h-index.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent.tools.meta import annotate_tool
from agent.tools.deps import ToolDeps

logger = logging.getLogger(__name__)

_QUERY_INDICATORS = frozenset({
    "a", "an", "the",
    "who", "what", "where", "which", "tell", "is", "are",
    "working", "work", "research", "professor", "faculty",
    "iit", "delhi", "find", "show",
})

# Department names at IIT Delhi are short proper nouns (2–5 words at most).
# Anything longer, or containing question-mark / sentence words, is a user
# query that the LLM accidentally forwarded — treat it as no filter.
_MAX_DEPT_WORDS = 6


def _sanitize_department(raw: str | None) -> str | None:
    """Return None if `raw` looks like a query rather than a department name."""
    if not raw:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    words = stripped.split()
    if len(words) > _MAX_DEPT_WORDS:
        return None
    if "?" in stripped or "." in stripped:
        return None
    # Reject if majority of words are query-like keywords
    indicator_hits = sum(1 for w in words if w.lower() in _QUERY_INDICATORS)
    if indicator_hits >= 2:
        return None
    return stripped


class FindFacultyArgs(BaseModel):
    topic: str = Field(description="Research topic or question to find faculty for")
    department: Optional[str] = Field(
        default=None,
        description=(
            "Set ONLY when the user explicitly names a specific IIT Delhi department, centre, or school — "
            "e.g. 'Computer Science and Engineering', 'Electrical Engineering', 'Chemical Engineering', "
            "'Centre for Applied Research in Electronics'. "
            "NEVER pass the user's query, a research topic, or any sentence here. "
            "Leave null for general or cross-department topic searches."
        ),
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


def build_tool(deps: ToolDeps) -> BaseTool:
    faculty_repo = deps.faculty_repo
    search_client = deps.search_client
    if search_client is None:
        from agent.transports.faculty_search import HttpFacultySearchClient
        search_client = HttpFacultySearchClient(deps.config.SEARCH_API_URL)

    @tool(args_schema=FindFacultyArgs)
    async def find_faculty_for_topic(topic: str, department: str | None = None) -> str:
        """Find IIT Delhi professors who work on a given research topic, with department, email, and paper count.
        Set department only when the user names a specific IIT Delhi department — never pass the research topic or user query as department."""
        department = _sanitize_department(department)

        search_api_ok = True
        agg: dict[str, Any] = {"departments": [], "total_matching_papers": 0}

        try:
            agg = await search_client.faculty_for_query(topic)
        except Exception as exc:
            logger.warning("faculty-for-query call failed: %s", exc)
            search_api_ok = False

        flat: list[dict[str, Any]] = []
        for dept in agg.get("departments", []):
            for f in dept.get("faculty", []):
                flat.append({**f, "department": dept["name"]})
        flat.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        dept_in_search = False
        if department:
            dept_lower = department.lower().strip()
            dept_matches = [f for f in flat if dept_lower in f.get("department", "").lower()]
            dept_in_search = len(dept_matches) > 0

            if dept_in_search:
                other = [f for f in flat if dept_lower not in f.get("department", "").lower()]
                flat = dept_matches + other

        # If department was requested but not found in search API results,
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
            return json.dumps({
                "topic": topic,
                "department_filter": department,
                "faculty": [],
                "error": f"No department matching '{department}' found in the database.",
            })

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

    return annotate_tool(
        find_faculty_for_topic,
        thinking_label="Identifying relevant researchers",
        token_cap=deps.config.TOKEN_CAP_FACULTY_EXPERTISE,
    )
