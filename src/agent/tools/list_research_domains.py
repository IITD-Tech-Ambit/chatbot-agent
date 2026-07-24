"""list_research_domains tool — research domains (discipline axis) + counts."""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import BaseTool, tool

from agent.tools.deps import ToolDeps
from agent.tools.meta import annotate_tool


def build_tool(deps: ToolDeps) -> BaseTool:
    taxonomy_repo = deps.taxonomy_repo
    faculty_repo = deps.faculty_repo
    cap = deps.config.TOKEN_CAP_RESEARCH_DOMAINS

    @tool
    async def list_research_domains(
        theme: Optional[str] = None,
        department: Optional[str] = None,
        sort_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        """List research domains (the discipline axis, e.g. "Power Electronics",
        "Machine Learning", "Photonics") with their paper and faculty counts.
        Domains are an INDEPENDENT axis from thematic areas — a paper has one
        theme AND one domain. Pass `theme` to list only the domains that carry
        papers within a given thematic area; pass `department` to scope counts
        to a department. Use for "what research domains / fields exist", "which
        domains fall under theme X", or "what fields does department Y work in".

        Knobs:
          - `sort_by`: "paper_count" (default), "faculty_count", or "name".
          - `limit`: return only the top N domains (e.g. "top 10 domains" → 10)."""
        if taxonomy_repo is None:
            return json.dumps({"domains": [], "error": "Classification data is not available"})

        theme_id = None
        theme_name = None
        if theme:
            theme_doc = await taxonomy_repo.resolve_theme(theme)
            if not theme_doc:
                return json.dumps({"domains": [], "error": f'No thematic area matching "{theme}" was found.'})
            theme_id = theme_doc["_id"]
            theme_name = theme_doc.get("name")

        dept_id = None
        if department:
            dept = await faculty_repo.find_department(department)
            if not dept:
                return json.dumps({"domains": [], "error": f'No department matching "{department}" was found.'})
            dept_id = dept["_id"]

        try:
            if theme_id is not None or dept_id is not None:
                rows = await taxonomy_repo.domain_counts(theme_id=theme_id, department_id=dept_id)
                _, domain_names = await taxonomy_repo.name_maps()
                out = [{
                    "name": domain_names.get(str(r.get("domain_id")), "Unknown"),
                    "paper_count": r.get("paper_count", 0),
                    "faculty_count": r.get("faculty_count", 0),
                } for r in rows if r.get("domain_id") is not None]
            else:
                domains = await taxonomy_repo.all_domains()
                out = [{
                    "name": d.get("name"),
                    "paper_count": (d.get("stats") or {}).get("paper_count", 0),
                    "faculty_count": (d.get("stats") or {}).get("faculty_count", 0),
                } for d in domains]
        except Exception as exc:
            return json.dumps({"domains": [], "error": f"Lookup failed: {type(exc).__name__}"})

        out = [d for d in out if d.get("paper_count", 0) > 0]
        mode = (sort_by or "").strip().lower()
        if mode in ("faculty_count", "faculty"):
            out.sort(key=lambda x: x.get("faculty_count", 0), reverse=True)
        elif mode == "name":
            out.sort(key=lambda x: (x.get("name") or "").lower())
        else:  # paper_count (default)
            out.sort(key=lambda x: x.get("paper_count", 0), reverse=True)
        if limit and limit >= 1:
            out = out[: int(limit)]
        result = {
            "theme": theme_name, "department": department,
            "count": len(out), "domains": out,
        }
        output = json.dumps(result, default=str)
        while len(output) > cap and result["domains"]:
            result["domains"].pop()
            output = json.dumps(result, default=str)
        return output

    return annotate_tool(
        list_research_domains,
        thinking_label="Loading research domains",
        token_cap=cap,
    )
