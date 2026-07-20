"""theme_distribution tool — research profile across thematic areas."""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import BaseTool, tool

from agent.tools.deps import ToolDeps
from agent.tools.meta import annotate_tool


def build_tool(deps: ToolDeps) -> BaseTool:
    taxonomy_repo = deps.taxonomy_repo
    faculty_repo = deps.faculty_repo
    cap = deps.config.TOKEN_CAP_THEME_DISTRIBUTION

    @tool
    async def theme_distribution(department: Optional[str] = None) -> str:
        """Show how IIT Delhi's classified papers are distributed across the
        research thematic areas — the research profile / landscape. Use for
        "what is IIT Delhi's research profile", "breakdown of research by theme",
        "which areas does IIT Delhi focus on", or "plot research areas". Pass
        `department` to get that department's profile across themes instead of
        the whole institute. The frontend renders this as a chart automatically."""
        if taxonomy_repo is None:
            return json.dumps({"distribution": [], "error": "Classification data is not available"})

        dept_ref = None
        if department:
            dept = await faculty_repo.find_department(department)
            if not dept:
                return json.dumps({"distribution": [], "error": f'No department matching "{department}" was found.'})
            dept_ref = dept["_id"]

        try:
            rows = await taxonomy_repo.theme_distribution(department_ref=dept_ref)
            theme_names, _ = await taxonomy_repo.name_maps()
        except Exception as exc:
            return json.dumps({"distribution": [], "error": f"Aggregation failed: {type(exc).__name__}"})

        distribution = [{
            "theme": theme_names.get(str(r.get("_id")), "Unclassified"),
            "paper_count": r.get("count", 0),
        } for r in rows]
        total = sum(d["paper_count"] for d in distribution)

        result = {
            "scope": department or "IIT Delhi (all)",
            "total_classified_papers": total,
            "distribution": distribution,
        }
        output = json.dumps(result, default=str)
        while len(output) > cap and result["distribution"]:
            result["distribution"].pop()
            output = json.dumps(result, default=str)
        return output

    return annotate_tool(
        theme_distribution,
        thinking_label="Analyzing research profile by theme",
        token_cap=cap,
    )
