"""search_ips tool — hybrid semantic/keyword search over IIT Delhi patents & IP."""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent.tools.deps import ToolDeps
from agent.tools.meta import annotate_tool


class SearchIpsArgs(BaseModel):
    query: str = Field(description="Topic, title, or invention to search patents/IP for")
    type_of_ip: Optional[str] = Field(default=None, description='Filter by IP type, e.g. "Patent", "Copyright", "Design"')
    year_from: Optional[int] = Field(default=None, description="Only IP with publication year >= this")
    year_to: Optional[int] = Field(default=None, description="Only IP with publication year <= this")
    country: Optional[str] = Field(default=None, description='Filing jurisdiction, e.g. "IN"')
    department: Optional[str] = Field(default=None, description="IIT Delhi department name")
    inventor: Optional[str] = Field(default=None, description="Inventor name or kerberos")
    classification_prefix: Optional[str] = Field(
        default=None,
        description='IPC code prefix to restrict results, e.g. "A61K" for pharmaceuticals',
    )


def build_tool(deps: ToolDeps) -> BaseTool:
    ip_retriever = deps.ip_retriever
    cap = deps.config.TOKEN_CAP_SEARCH_IPS

    @tool(args_schema=SearchIpsArgs)
    async def search_ips(
        query: str,
        type_of_ip: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        country: str | None = None,
        department: str | None = None,
        inventor: str | None = None,
        classification_prefix: str | None = None,
    ) -> str:
        """Search IIT Delhi patents and other IP (copyrights, designs) by topic, title, or invention.
        Use for questions about what has been patented or filed, optionally filtered by type, year, country, department, inventor, or IPC classification prefix."""
        if ip_retriever is None:
            return json.dumps({"ips": [], "error": "IP search is not available"})
        try:
            hits = await ip_retriever.retrieve(
                query,
                type_of_ip=type_of_ip,
                year_from=year_from,
                year_to=year_to,
                country=country,
                department=department,
                inventor=inventor,
                classification_prefix=classification_prefix,
            )
        except Exception as exc:
            return json.dumps({"ips": [], "error": f"Retrieval failed: {type(exc).__name__}"})

        result = {
            "ips": [
                {
                    "citation_index": i + 1,
                    "id": h.get("id", ""),
                    "application_number": h.get("application_number"),
                    "title": h.get("title"),
                    "type_of_ip": h.get("type_of_ip"),
                    "inventors": [inv.get("name") for inv in h.get("inventors", [])][:5],
                    "department": h.get("department"),
                    "classification": h.get("classification", [])[:6],
                    "filing_date": h.get("filing_date"),
                    "publication_date": h.get("publication_date"),
                    "publication_year": h.get("publication_year"),
                    "country": h.get("country"),
                    "abstract": h.get("abstract"),
                }
                for i, h in enumerate(hits)
            ]
        }

        output = json.dumps(result, default=str)
        while len(output) > cap and result["ips"]:
            result["ips"].pop()
            output = json.dumps(result, default=str)
        return output

    return annotate_tool(
        search_ips,
        thinking_label="Searching patents & IP filings",
        token_cap=cap,
    )
