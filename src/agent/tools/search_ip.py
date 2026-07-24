"""search_ip — the ExploreIP page's advanced patent/IP search, as a bot tool.

Wraps search-api `POST /api/v1/ip/search`: hybrid BM25 + semantic search over
IIT Delhi patents / copyrights / designs with facet filters, returning matching
filings, related inventors (faculty), and facet distributions.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent.tools.deps import ToolDeps
from agent.tools.meta import annotate_tool
from agent.transports.research_search import ResearchSearchClient

logger = logging.getLogger(__name__)

_PER_PAGE = 10
_SORTS = {"relevance", "date", "normalized"}
_SEARCH_IN = {"title", "abstract", "inventor", "field_of_invention", "classification"}


class SearchIpArgs(BaseModel):
    query: str = Field(description="The invention topic, keywords, or an inventor name to search for.")
    year_from: Optional[int] = Field(default=None, description="Only filings published in or after this year.")
    year_to: Optional[int] = Field(default=None, description="Only filings published in or before this year.")
    type_of_ip: Optional[list[str]] = Field(default=None, description="Restrict to IP types, e.g. ['Patent'], ['Copyright'], ['Design'].")
    field_of_invention: Optional[str] = Field(default=None, description="Restrict to a field of invention (exact keyword).")
    country: Optional[str] = Field(default=None, description="Filing jurisdiction, e.g. 'IN'.")
    sort: Optional[str] = Field(default=None, description="One of: relevance (default), date (newest).")
    search_in: Optional[list[str]] = Field(default=None, description="Restrict matching to these fields only: title, abstract, inventor, field_of_invention, classification. Use ['inventor'] to search by a person's name.")
    primary_inventor_only: Optional[bool] = Field(default=None, description="Only filings where the matched person is the primary inventor.")


def _inventor_names(inventors: list, limit: int = 4) -> list[str]:
    names: list[str] = []
    for inv in inventors or []:
        if isinstance(inv, dict):
            n = inv.get("name") or inv.get("raw_name")
        else:
            n = inv
        if n:
            names.append(n)
        if len(names) >= limit:
            break
    return names


def _facet_summary(facets: dict) -> dict:
    if not isinstance(facets, dict):
        return {}
    out: dict = {}
    for key in ("type_of_ip", "field_of_invention", "country", "classification"):
        buckets = facets.get(key)
        if isinstance(buckets, list) and buckets:
            out[key] = [
                {"value": b.get("key"), "count": b.get("doc_count", b.get("count"))}
                for b in buckets[:6]
                if isinstance(b, dict)
            ]
    return out


def build_tool(deps: ToolDeps) -> BaseTool:
    client = ResearchSearchClient(deps.config.SEARCH_API_URL)
    cap = deps.config.TOKEN_CAP_SEARCH_IPS

    @tool(args_schema=SearchIpArgs)
    async def search_ip(
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        type_of_ip: Optional[list[str]] = None,
        field_of_invention: Optional[str] = None,
        country: Optional[str] = None,
        sort: Optional[str] = None,
        search_in: Optional[list[str]] = None,
        primary_inventor_only: Optional[bool] = None,
    ) -> str:
        """Search IIT Delhi PATENTS and intellectual property (patents,
        copyrights, designs) — the primary tool for any question about patents,
        IP, inventions, or "what has Prof X patented". Same hybrid (keyword +
        semantic) engine as the ExploreIP page.

        Use it for: "patents on lithium batteries", "IIT Delhi copyrights on
        educational software", "designs filed since 2021", "who has patented
        work on drug delivery" (the response includes related inventors).

        Honor the user's constraints with the knobs:
          - year_from / year_to for date ranges; sort='date' for "latest".
          - type_of_ip: ['Patent'] / ['Copyright'] / ['Design'].
          - field_of_invention, country ('IN', ...).
          - search_in=['inventor'] to search specifically by a person's name.
          - primary_inventor_only.

        Returns matching filings (title, application number, inventors, type,
        year) plus related inventors. Use search_research for research PAPERS,
        not this."""
        filters: dict = {}
        if year_from:
            filters["year_from"] = year_from
        if year_to:
            filters["year_to"] = year_to
        if type_of_ip:
            filters["type_of_ip_list"] = type_of_ip
        if field_of_invention:
            filters["field_of_invention"] = field_of_invention
        if country:
            filters["country"] = country
        if primary_inventor_only:
            filters["primary_inventor_only"] = True

        body: dict = {"query": query, "mode": "advanced", "per_page": _PER_PAGE}
        body["sort"] = sort if sort in _SORTS else "relevance"
        if filters:
            body["filters"] = filters
        if search_in:
            valid = [f for f in search_in if f in _SEARCH_IN]
            if valid:
                body["search_in"] = valid

        try:
            data = await client.ip_search(body)
        except Exception as exc:
            logger.warning("search_ip call failed: %s", exc)
            return json.dumps({"ips": [], "error": f"IP search failed: {type(exc).__name__}"})

        results = data.get("results", []) or []
        ips = []
        for i, p in enumerate(results):
            dept = p.get("department") or {}
            ips.append({
                "citation_index": i + 1,
                "id": p.get("_id") or p.get("open_search_id", ""),
                "application_number": p.get("application_number"),
                "title": p.get("title", ""),
                "type_of_ip": p.get("type_of_ip"),
                "field_of_invention": p.get("field_of_invention"),
                "department": dept.get("name") if isinstance(dept, dict) else None,
                "country": p.get("country"),
                "publication_year": p.get("publication_year"),
                "inventors": _inventor_names(p.get("inventors", [])),
            })

        related_faculty = []
        for f in (data.get("related_faculty", []) or [])[:10]:
            dept = f.get("department") or {}
            kerb = f.get("kerberos") or ((f.get("email") or "").split("@")[0].lower() or None)
            related_faculty.append({
                "name": f.get("name"),
                "department": dept.get("name") if isinstance(dept, dict) else None,
                "kerberos": kerb,
                "profile_url": f"/faculty/{kerb}" if kerb else None,
                "ip_count": f.get("ipCount"),
            })

        pagination = data.get("pagination", {}) or {}
        result = {
            "query": query,
            "sorted_by": body["sort"],
            "total_matching_ips": pagination.get("total", len(ips)),
            "showing": len(ips),
            "ips": ips,
            "related_faculty": related_faculty,
            "facets": _facet_summary(data.get("facets", {})),
            "suggestions": data.get("suggestions", [])[:5],
        }
        if not ips and data.get("message"):
            result["message"] = data["message"]

        output = json.dumps(result, default=str)
        while len(output) > cap and result["ips"]:
            result["ips"].pop()
            output = json.dumps(result, default=str)
        return output

    return annotate_tool(
        search_ip,
        thinking_label="Searching IIT Delhi patents & IP",
        token_cap=cap,
    )
