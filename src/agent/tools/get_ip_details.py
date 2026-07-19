"""get_ip_details tool — full record for a single patent/IP filing."""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent.tools.deps import ToolDeps
from agent.tools.meta import annotate_tool


class GetIpDetailsArgs(BaseModel):
    application_number: Optional[str] = Field(default=None, description="Exact application number of the IP filing")
    title: Optional[str] = Field(default=None, description="Title of the patent/IP when the application number is unknown")


def _serialize(doc: dict[str, Any], department_name: str | None) -> dict[str, Any]:
    return {
        "application_number": doc.get("application_number"),
        "title": doc.get("title"),
        "abstract": doc.get("abstract"),
        "type_of_ip": doc.get("type_of_ip"),
        "field_of_invention": doc.get("field_of_invention"),
        "classification": doc.get("classification") or [],
        "department": department_name,
        "country": doc.get("country"),
        "filing_date": doc.get("filing_date"),
        "publication_date": doc.get("publication_date"),
        "publication_year": doc.get("publication_year"),
        "inventors": [
            {
                "name": inv.get("name"),
                "is_faculty": bool(inv.get("is_faculty")),
                "kerberos": inv.get("kerberos"),
            }
            for inv in (doc.get("inventors") or [])
            if inv.get("name")
        ],
        "applicants": doc.get("applicants") or [],
    }


def build_tool(deps: ToolDeps) -> BaseTool:
    ip_repo = deps.ip_repo
    cap = deps.config.TOKEN_CAP_IP_DETAILS

    @tool(args_schema=GetIpDetailsArgs)
    async def get_ip_details(application_number: str | None = None, title: str | None = None) -> str:
        """Get the full record of a specific IIT Delhi patent/IP filing by application number, or by best title match."""
        if ip_repo is None:
            return json.dumps({"error": "IP lookup is not available"})
        if not application_number and not title:
            return json.dumps({"error": "Provide an application number or a title."})

        doc = None
        if application_number:
            doc = await ip_repo.get_by_application_number(application_number.strip())
        if doc is None and title:
            matches = await ip_repo.search_by_title(title.strip(), limit=1)
            doc = matches[0] if matches else None

        if doc is None:
            ref = application_number or title
            return json.dumps({"error": f'No IP filing found for "{ref}".'})

        dept_name = None
        dept_id = doc.get("department")
        if dept_id:
            name_map = await ip_repo.resolve_department_names([dept_id])
            dept_name = name_map.get(dept_id)

        return json.dumps({"ip": _serialize(doc, dept_name)}, default=str)

    return annotate_tool(
        get_ip_details,
        thinking_label="Loading patent details",
        token_cap=cap,
    )
