"""get_ip_stats tool — patent/IP analytics via MongoDB aggregations."""

from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent.tools.deps import ToolDeps
from agent.tools.meta import annotate_tool

_DIMENSION_ALIASES = {
    "department": "department",
    "dept": "department",
    "year": "year",
    "type": "type",
    "ip": "type",
    "country": "country",
    "classification": "classification",
    "ipc": "classification",
    "inventor": "inventor",
    "faculty": "inventor",
}


def _parse_dimensions(group_by: str | None) -> list[str]:
    if not group_by:
        return ["year"]
    dims: list[str] = []
    for token in re.split(r"[^a-z]+", group_by.lower()):
        mapped = _DIMENSION_ALIASES.get(token)
        if mapped and mapped not in dims:
            dims.append(mapped)
    return dims or ["year"]


def build_tool(deps: ToolDeps) -> BaseTool:
    ip_repo = deps.ip_repo
    faculty_repo = deps.faculty_repo
    ipc_service = deps.ipc_service
    cap = deps.config.TOKEN_CAP_IP_STATS

    @tool
    async def get_ip_stats(
        group_by: Optional[str] = None,
        type_of_ip: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        country: Optional[str] = None,
        department: Optional[str] = None,
        inventor: Optional[str] = None,
        classification_prefix: Optional[str] = None,
    ) -> str:
        """Patent/IP statistics for IIT Delhi. group_by supports combinations like
        "department", "year", "type", "country", "classification", "department x year",
        "type x year", or "inventor x year" (e.g. "which department filed how many patents in 2023")."""
        if ip_repo is None:
            return json.dumps({"groups": [], "error": "IP statistics are not available"})

        dimensions = _parse_dimensions(group_by)

        match: dict = {}
        if year_from or year_to:
            yr: dict = {}
            if year_from:
                yr["$gte"] = year_from
            if year_to:
                yr["$lte"] = year_to
            match["publication_year"] = yr
        if type_of_ip:
            match["type_of_ip"] = {"$regex": f"^{re.escape(type_of_ip.strip())}$", "$options": "i"}
        if country:
            match["country"] = {"$regex": f"^{re.escape(country.strip())}$", "$options": "i"}
        if classification_prefix:
            match["classification"] = {"$regex": f"^{re.escape(classification_prefix.strip())}", "$options": "i"}
        if inventor:
            match["$or"] = [
                {"inventors.kerberos": inventor.lower().strip()},
                {"inventors.name": {"$regex": re.escape(inventor.strip()), "$options": "i"}},
            ]
        if department:
            dept = await faculty_repo.find_department(department)
            if not dept:
                return json.dumps({"groups": [], "error": f'No department matching "{department}" was found.'})
            match["department"] = dept["_id"]

        try:
            groups, total = (
                await ip_repo.grouped_counts(match, dimensions, limit=200),
                await ip_repo.count_documents(match),
            )
        except Exception as exc:
            return json.dumps({"groups": [], "error": f"Aggregation failed: {type(exc).__name__}"})

        if "classification" in dimensions and ipc_service is not None:
            for g in groups:
                code = g.get("classification")
                if code:
                    g["classification_code"] = code
                    g["classification"] = ipc_service.format_label(code)

        result = {
            "grouped_by": " x ".join(dimensions),
            "dimensions": dimensions,
            "total": total,
            "year_from": year_from,
            "year_to": year_to,
            "groups": groups,
        }

        output = json.dumps(result, default=str)
        while len(output) > cap and result["groups"]:
            result["groups"].pop()
            output = json.dumps(result, default=str)
        return output

    return annotate_tool(
        get_ip_stats,
        thinking_label="Computing patent statistics",
        token_cap=cap,
    )
