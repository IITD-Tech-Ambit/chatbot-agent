"""find_faculty_by_expertise tool — find faculty by specific expertise keywords."""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent.tools.meta import annotate_tool
from agent.tools.deps import ToolDeps


class FacultyByExpertiseArgs(BaseModel):
    expertise: str = Field(
        description="Specific expertise area or skill, e.g. 'computer vision', 'VLSI design', 'solid mechanics', 'nanomaterials'",
        min_length=2,
        max_length=200,
    )
    department: Optional[str] = Field(
        default=None,
        description="Optionally filter by department name",
        max_length=200,
    )
    limit: int = Field(default=10, ge=1, le=20)


def build_tool(deps: ToolDeps) -> BaseTool:
    faculty_repo = deps.faculty_repo

    @tool("find_faculty_by_expertise", args_schema=FacultyByExpertiseArgs)
    async def find_faculty_by_expertise(
        expertise: str,
        department: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Find IIT Delhi faculty who have a specific expertise or skill listed in their profile. Use when the user asks about a precise technical skill or research area keyword."""
        terms = [t.strip() for t in expertise.replace(",", " ").split() if len(t.strip()) >= 2]
        if not terms:
            terms = [expertise.strip()]

        docs = await faculty_repo.find_faculty_by_expertise(terms, limit=limit * 2)

        if department:
            dept_doc = await faculty_repo.find_department(department)
            if dept_doc:
                dept_id = str(dept_doc["_id"])
                docs = [
                    d for d in docs
                    if str((d.get("department") or {}).get("_id", "")) == dept_id
                ]

        docs = docs[:limit]

        if not docs:
            return json.dumps({
                "expertise": expertise,
                "count": 0,
                "faculty": [],
                "message": f'No faculty found with expertise in "{expertise}".',
            })

        faculty_list = [
            {
                "name": f"{d.get('title', '')} {d.get('firstName', '')} {d.get('lastName', '')}".strip(),
                "email": d.get("email", ""),
                "designation": d.get("designation", ""),
                "department": (d.get("department") or {}).get("name", ""),
                "expertise": (d.get("expertise") or [])[:5],
                "h_index": d.get("h_index"),
                "citation_count": d.get("citation_count"),
            }
            for d in docs
        ]

        result = {
            "expertise": expertise,
            "department_filter": department,
            "count": len(faculty_list),
            "faculty": faculty_list,
        }
        return json.dumps(result, default=str)

    return annotate_tool(
        find_faculty_by_expertise,
        thinking_label="Scanning faculty expertise profiles",
        token_cap=deps.config.TOKEN_CAP_FACULTY_EXPERTISE,
    )
