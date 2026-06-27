"""get_top_faculty tool — rank IIT Delhi faculty by H-index or citation count."""

from __future__ import annotations

import json
import logging
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.tools._registry import get_faculty_repo

logger = logging.getLogger(__name__)


class TopFacultyArgs(BaseModel):
    sort_by: Literal["h_index", "citation_count"] = Field(
        default="h_index",
        description="Rank faculty by 'h_index' (default) or 'citation_count'.",
    )
    limit: int = Field(default=10, ge=1, le=25, description="Number of results (max 25)")
    department: Optional[str] = Field(
        default=None,
        description="Restrict to a specific department, e.g. 'Computer Science and Engineering'",
        max_length=200,
    )


@tool("get_top_faculty", args_schema=TopFacultyArgs)
async def get_top_faculty(
    sort_by: Literal["h_index", "citation_count"] = "h_index",
    limit: int = 10,
    department: Optional[str] = None,
) -> str:
    """Return a ranked list of IIT Delhi faculty sorted by H-index or total citation count.
    Use this when the user asks for the 'top', 'best', 'highest', or 'most cited' professors
    at IIT Delhi, or within a specific department."""
    faculty_repo = get_faculty_repo()

    docs = await faculty_repo.find_top_faculty_global(
        sort_by=sort_by,
        limit=limit,
        department_name=department,
    )

    if not docs:
        scope = f" in {department}" if department else ""
        return json.dumps({"error": f"No faculty data found{scope}."})

    label = "H-Index" if sort_by == "h_index" else "Total Citations"
    faculty_list = []
    for rank, d in enumerate(docs, 1):
        dept_info = d.get("department") or {}
        dept_name = dept_info.get("name", "") if isinstance(dept_info, dict) else ""
        faculty_list.append({
            "rank": rank,
            "name": f"{d.get('title', '')} {d.get('firstName', '')} {d.get('lastName', '')}".strip(),
            "email": d.get("email", ""),
            "designation": d.get("designation", ""),
            "department": dept_name,
            "h_index": d.get("h_index"),
            "citation_count": d.get("citation_count"),
        })

    result = {
        "ranked_by": label,
        "department_filter": department,
        "count": len(faculty_list),
        "faculty": faculty_list,
    }

    return json.dumps(result, default=str)
