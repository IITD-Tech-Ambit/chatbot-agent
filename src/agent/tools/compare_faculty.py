"""compare_faculty tool — side-by-side metrics for two professors."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class CompareFacultyArgs(BaseModel):
    name_a: str = Field(description="First professor's name")
    name_b: str = Field(description="Second professor's name")


@tool(args_schema=CompareFacultyArgs)
async def compare_faculty(name_a: str, name_b: str) -> str:
    """Compare two IIT Delhi professors side-by-side: h-index, citations, paper count, expertise."""
    from agent.guardrails.guardrails import name_tokens, faculty_name_matches
    from agent.tools._registry import get_faculty_repo, get_research_repo, get_config

    faculty_repo = get_faculty_repo()
    research_repo = get_research_repo()
    cfg = get_config()

    async def _resolve(name: str) -> dict[str, Any] | None:
        tokens = name_tokens(name)
        if not tokens:
            return None
        matches = await faculty_repo.text_search(" ".join(tokens), limit=3)
        if not matches:
            matches = await faculty_repo.regex_search(tokens, limit=3)
        for m in matches:
            if faculty_name_matches(name, m.get("firstName", ""), m.get("lastName", "")):
                kerberos = (m.get("email") or "").split("@")[0].lower()
                scopus_ids = [str(s) for s in (m.get("scopus_id") or [])]
                or_clauses: list[dict] = []
                if kerberos:
                    or_clauses.append({"kerberos": kerberos})
                if scopus_ids:
                    or_clauses.append({"authors.author_id": {"$in": scopus_ids}})
                total = 0
                if or_clauses:
                    total = await research_repo.count_documents({"$or": or_clauses})
                dept = m.get("department")
                return {
                    "name": f"{m.get('title', '')} {m.get('firstName', '')} {m.get('lastName', '')}".strip(),
                    "department": dept.get("name") if isinstance(dept, dict) else None,
                    "h_index": m.get("h_index"),
                    "total_citations": m.get("citation_count"),
                    "total_papers": total,
                    "expertise": (m.get("brief_expertise") or m.get("expertise") or [])[:6],
                }
        return None

    a = await _resolve(name_a)
    b = await _resolve(name_b)

    if not a and not b:
        return json.dumps({"error": f'Neither "{name_a}" nor "{name_b}" found.'})
    if not a:
        return json.dumps({"error": f'Faculty "{name_a}" not found.', "found": b})
    if not b:
        return json.dumps({"error": f'Faculty "{name_b}" not found.', "found": a})

    output = json.dumps({"comparison": [a, b]}, default=str)
    cap = cfg.TOKEN_CAP_DEFAULT
    if len(output) > cap:
        output = output[:cap] + '..."}'
    return output
