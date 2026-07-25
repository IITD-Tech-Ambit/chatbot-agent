"""list_department_faculty — the Directory page's "faculty in a unit" view.

Given a department / centre / school name (the bot knows which kind it is from
the structural reference in the system prompt), returns the top faculty in that
unit — in the SAME order the Directory page shows them (h-index desc) — plus a
button to open that unit's expanded list on the Directory page.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent.tools.deps import ToolDeps
from agent.tools.meta import annotate_tool
from agent.tools._taxonomy_resolve import resolve_node
from agent.transports.directory import DirectoryClient

logger = logging.getLogger(__name__)

_RETURN = 10

# User-facing category -> Directory API category (+ common aliases).
_CATEGORY = {
    "department": "departments", "departments": "departments", "dept": "departments",
    "centre": "centres", "center": "centres", "centres": "centres", "centers": "centres",
    "school": "schools", "schools": "schools",
}


class ListDepartmentFacultyArgs(BaseModel):
    unit: str = Field(description="The department / centre / school name, e.g. 'Chemical Engineering' or 'Centre for Energy Studies'.")
    category: str = Field(description="Which kind of unit this is — 'department', 'centre', or 'school'. Determine it from the structural reference (the unit is listed under Departments, Centres, or Schools there).")


def build_tool(deps: ToolDeps) -> BaseTool:
    client = DirectoryClient(deps.config.BACKEND_API_URL)
    cap = deps.config.TOKEN_CAP_LIST_DEPARTMENTS

    @tool(args_schema=ListDepartmentFacultyArgs)
    async def list_department_faculty(unit: str, category: str) -> str:
        """List the faculty of a specific IIT Delhi department, centre, or school
        — the Directory page's per-unit view. Use for "who are the faculty in the
        Chemical Engineering department", "professors in the Centre for Energy
        Studies", "faculty of the School of AI".

        `category` must be 'department', 'centre', or 'school' — read it off the
        structural reference (each unit is listed under one of those). Returns the
        top 10 faculty (name, h-index, profile link) in the Directory page's order
        (h-index, highest first), the unit's total faculty count, and a button to
        open that unit's full list on the Directory page."""
        cat = _CATEGORY.get((category or "").strip().lower())
        if not cat:
            return json.dumps({"faculty": [], "error": "category must be department, centre, or school."})

        try:
            units = await client.unit_groups(cat)
        except Exception as exc:
            logger.warning("directory unit lookup failed: %s", exc)
            return json.dumps({"faculty": [], "error": f"Directory lookup failed: {type(exc).__name__}"})

        items = [
            {"name": (u.get("department") or {}).get("name"), "id": (u.get("department") or {}).get("_id")}
            for u in units
            if (u.get("department") or {}).get("name") and (u.get("department") or {}).get("_id")
        ]
        match = resolve_node(items, unit, name_key="name", slug_key="name")
        if not match:
            return json.dumps({
                "faculty": [],
                "error": f'No {cat[:-1]} matching "{unit}" was found.',
                "available": [i["name"] for i in items],
            })

        try:
            faculty = await client.unit_faculty(match["id"], cat)
        except Exception as exc:
            logger.warning("directory unit-faculty fetch failed: %s", exc)
            return json.dumps({"faculty": [], "error": f"Directory lookup failed: {type(exc).__name__}"})

        total = len(faculty)
        top = []
        for f in faculty[:_RETURN]:
            kerb = (f.get("email") or "").split("@")[0].lower() or None
            top.append({
                "name": f.get("name"),
                "h_index": f.get("hIndex"),
                "designation": f.get("designation"),
                "kerberos": kerb,
                "profile_url": f"/faculty/{kerb}" if kerb else None,
            })

        result = {
            "unit": match["name"],
            "category": category.strip().lower(),
            "total_faculty": total,
            "showing": len(top),
            "faculty": top,
            "explore_link": {
                "kind": "directory",
                "category": cat,
                "unit": match["name"],
                "label": f"View all {total} faculty in {match['name']}",
            },
        }
        output = json.dumps(result, default=str)
        while len(output) > cap and len(result["faculty"]) > 3:
            result["faculty"].pop()
            result["showing"] = len(result["faculty"])
            output = json.dumps(result, default=str)
        return output

    return annotate_tool(
        list_department_faculty,
        thinking_label="Looking up department faculty",
        token_cap=cap,
    )
