"""list_departments tool — list all IIT Delhi departments, schools, and centres."""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.tools._registry import get_faculty_repo, get_config

logger = logging.getLogger(__name__)

_VALID_CATEGORIES = {"Department", "School", "Centre", "Research Lab or Facility", "Other"}


class ListDepartmentsArgs(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="Filter by category: 'Department', 'School', 'Centre', 'Research Lab or Facility', or 'Other'. Leave empty for all.",
        max_length=50,
    )


@tool("list_departments", args_schema=ListDepartmentsArgs)
async def list_departments(category: Optional[str] = None) -> str:
    """List all IIT Delhi departments, schools, centres, and research labs grouped by category."""
    faculty_repo = get_faculty_repo()
    cfg = get_config()

    all_depts = await faculty_repo.list_all_departments(category=category)

    grouped: dict[str, list[dict]] = {}
    for dept in all_depts:
        cat = dept.get("category", "Other")
        grouped.setdefault(cat, []).append({
            "name": dept.get("name", ""),
            "code": dept.get("code", ""),
        })

    result = {
        "total": len(all_depts),
        "filter_category": category,
        "departments": grouped,
    }

    output = json.dumps(result, default=str)
    cap = cfg.TOKEN_CAP_LIST_DEPARTMENTS
    if len(output) > cap:
        output = output[:cap] + '..."}'
    return output
