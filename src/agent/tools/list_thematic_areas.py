"""list_thematic_areas tool — IIT Delhi's top-level research thematic areas."""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import BaseTool, tool

from agent.tools.deps import ToolDeps
from agent.tools.meta import annotate_tool


def build_tool(deps: ToolDeps) -> BaseTool:
    taxonomy_repo = deps.taxonomy_repo
    faculty_repo = deps.faculty_repo
    cap = deps.config.TOKEN_CAP_THEMATIC_AREAS

    @tool
    async def list_thematic_areas(department: Optional[str] = None) -> str:
        """List IIT Delhi's top-level research thematic areas (the strategic
        themes papers are classified into), each with its paper count and
        distinct faculty count. Use for questions like "what research themes /
        areas does IIT Delhi work on", "how many papers are in each theme", or
        the overall research landscape. Pass `department` to scope the counts to
        a single department."""
        if taxonomy_repo is None:
            return json.dumps({"themes": [], "error": "Classification data is not available"})

        dept_id = None
        if department:
            dept = await faculty_repo.find_department(department)
            if not dept:
                return json.dumps({"themes": [], "error": f'No department matching "{department}" was found.'})
            dept_id = dept["_id"]

        try:
            themes = await taxonomy_repo.all_themes()
            if dept_id is not None:
                rows = await taxonomy_repo.theme_counts_for_department(dept_id)
                by_id = {str(r.get("thematic_area_id")): r for r in rows}
                out = []
                for t in themes:
                    r = by_id.get(str(t["_id"]))
                    if not r:
                        continue  # no papers for this theme in this department
                    out.append({
                        "name": t.get("name"), "slug": t.get("slug"),
                        "paper_count": r.get("paper_count", 0),
                        "faculty_count": r.get("faculty_count", 0),
                    })
            else:
                out = [{
                    "name": t.get("name"), "slug": t.get("slug"),
                    "paper_count": (t.get("stats") or {}).get("paper_count", 0),
                    "faculty_count": (t.get("stats") or {}).get("faculty_count", 0),
                } for t in themes]
        except Exception as exc:
            return json.dumps({"themes": [], "error": f"Lookup failed: {type(exc).__name__}"})

        out.sort(key=lambda x: x.get("paper_count", 0), reverse=True)
        result = {"department": department, "count": len(out), "themes": out}
        output = json.dumps(result, default=str)
        while len(output) > cap and result["themes"]:
            result["themes"].pop()
            output = json.dumps(result, default=str)
        return output

    return annotate_tool(
        list_thematic_areas,
        thinking_label="Loading research thematic areas",
        token_cap=cap,
    )
