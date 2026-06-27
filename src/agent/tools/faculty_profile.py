"""get_faculty_profile tool — complex dual kerberos+scopus_id resolution.

Ported from toolsService.js _getFacultyProfile with identical dual-lookup logic.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class FacultyProfileArgs(BaseModel):
    name: str = Field(description="The professor's name (with or without titles like Prof./Dr.)")


@tool(args_schema=FacultyProfileArgs)
async def get_faculty_profile(name: str) -> str:
    """Look up a specific IIT Delhi professor BY NAME: email, department, expertise, and publication stats. Only use when the user names an actual person."""
    from agent.guardrails.guardrails import name_tokens, faculty_name_matches
    from agent.tools._registry import get_faculty_repo, get_research_repo

    faculty_repo = get_faculty_repo()
    research_repo = get_research_repo()

    tokens = name_tokens(name)
    if not tokens:
        return json.dumps({"error": "No valid professor name provided. Name a specific IIT Delhi professor."})

    cleaned = " ".join(tokens)

    # Primary: text index search
    matches = await faculty_repo.text_search(cleaned, limit=5)

    # Fallback: regex on firstName/lastName
    if not matches:
        matches = await faculty_repo.regex_search(tokens, limit=5)

    # Validate name overlap
    validated = [m for m in matches if faculty_name_matches(name, m.get("firstName", ""), m.get("lastName", ""))]

    if not validated:
        return json.dumps({"error": f'No IIT Delhi faculty named "{name}" found. Check the spelling.'})

    f = validated[0]
    kerberos = (f.get("email") or "").split("@")[0].lower()
    scopus_ids = [str(s) for s in (f.get("scopus_id") or [])]

    or_clauses: list[dict[str, Any]] = []
    if kerberos:
        or_clauses.append({"kerberos": kerberos})
    if scopus_ids:
        or_clauses.append({"authors.author_id": {"$in": scopus_ids}})

    stats = None
    if or_clauses:
        match_filter = {"$or": or_clauses}
        stats = await research_repo.faculty_publication_stats(match_filter)

    dept = f.get("department")
    dept_name = dept.get("name") if isinstance(dept, dict) else None

    expertise = (f.get("brief_expertise") or f.get("expertise") or [])[:10]
    subjects = (f.get("subjects") or [])[:10]

    result = {
        "profile": {
            "name": f"{f.get('title', '')} {f.get('firstName', '')} {f.get('lastName', '')}".strip(),
            "email": f.get("email"),
            "kerberos": kerberos,
            "profile_url": f"/faculty/{kerberos}" if kerberos else None,
            "department": dept_name,
            "designation": f.get("designation"),
            "expertise": expertise,
            "subjects": subjects,
            "h_index": f.get("h_index"),
            "total_citations": f.get("citation_count"),
        },
        "publication_stats": stats,
        "other_possible_matches": [
            {
                "name": f"{m.get('title', '')} {m.get('firstName', '')} {m.get('lastName', '')}".strip(),
                "department": m.get("department", {}).get("name") if isinstance(m.get("department"), dict) else None,
            }
            for m in validated[1:3]
        ],
    }

    return json.dumps(result, default=str)
