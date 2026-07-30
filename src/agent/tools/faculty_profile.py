"""get_faculty_profile — one IIT Delhi faculty member's Directory profile.

Mirrors the /faculty/<kerberos> profile page, assembled from the SAME endpoints
that page uses so the answer matches it:
  1. Resolve the name via the Directory search box (GET /directory/search?q=) and
     take the FIRST result — exactly what a user would land on for that query.
  2. Profile core from GET /directory/faculty/:kerberos/profile.
  3. Latest 10 papers from the research-summary timeline (newest year first).
  4. Latest 10 patents/IP from the inventor-scoped IP search (filters.kerberos),
     the same query the profile page's Patents section runs.

The resolved name is returned explicitly so the bot always states WHOSE profile
it is reporting (the Directory search is fuzzy — the top hit for a loose query
may be a different person than intended). The professor's name is hyperlinked to
`/faculty/<kerberos>`, which renders as the button to open the full profile.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent.tools.meta import annotate_tool
from agent.tools.deps import ToolDeps
from agent.transports.directory import DirectoryClient
from agent.transports.research_search import ResearchSearchClient

logger = logging.getLogger(__name__)

_LATEST = 10
_TITLE_PREFIX = re.compile(r"^\s*(?:prof(?:essor)?|dr|mr|ms|mrs)\.?\s+", re.IGNORECASE)


class FacultyProfileArgs(BaseModel):
    name: str = Field(description="The professor's name to look up (with or without titles like Prof./Dr.), e.g. 'Kodamana' or 'Bhim Singh'.")


def _kerberos_of(faculty: dict) -> str | None:
    return ((faculty.get("email") or "").split("@")[0].lower()) or None


def _latest_papers(summary: dict) -> list[dict[str, Any]]:
    """Flatten the newest-first year timeline into the latest N papers."""
    out: list[dict[str, Any]] = []
    for entry in summary.get("timeline", []) or []:
        year = entry.get("year")
        for p in entry.get("papers", []) or []:
            out.append({
                "title": p.get("title"),
                "year": year,
                "type": p.get("type"),
                "citations": p.get("citations"),
            })
            if len(out) >= _LATEST:
                return out
    return out


def build_tool(deps: ToolDeps) -> BaseTool:
    dir_client = DirectoryClient(deps.config.BACKEND_API_URL)
    ip_client = ResearchSearchClient(deps.config.SEARCH_API_URL)
    cap = deps.config.TOKEN_CAP_FACULTY_PROFILE

    @tool(args_schema=FacultyProfileArgs)
    async def get_faculty_profile(name: str) -> str:
        """Get one specific IIT Delhi faculty member's full profile — the
        Directory profile page for a named PERSON. Use for "tell me about Prof X",
        "who is Prof X", "X's designation / email / research interests / profile",
        or any request for details about a single named professor.

        Resolves the name through the Directory search (top result) and returns
        that person's designation, department, email, h-index, total citations,
        research areas, their latest 10 papers, and their latest 10 patents/IP
        (with totals for each). Hyperlink the person's NAME to `profile_url`
        (e.g. **[Dr Hariprasad Kodamana](/faculty/kodamana)**) — that name-link
        already IS the button to their profile. Do NOT add a separate
        "profile" / "view profile" / "visit his profile" link to the same person.

        For a DEPARTMENT overview use get_department_profile; for papers on a
        TOPIC use search_research; this tool is only for one named individual."""
        query = (name or "").strip()
        if not query:
            return json.dumps({"error": "No professor name provided."})

        try:
            hits = await dir_client.search_faculty(query, limit=5)
        except Exception as exc:
            logger.warning("faculty search failed: %s", exc)
            return json.dumps({"error": f"Directory lookup failed: {type(exc).__name__}"})

        if not hits:
            return json.dumps({"error": f'No IIT Delhi faculty found matching "{query}". Check the spelling.'})

        top = hits[0]
        kerberos = _kerberos_of(top)
        if not kerberos:
            return json.dumps({"error": f'Found "{top.get("name")}" but could not resolve their profile id.'})

        # Profile core: prefer the authoritative /profile record; fall back to the
        # search hit (which already carries almost all the same fields).
        core = top
        try:
            profile = await dir_client.faculty_profile(kerberos)
            if profile:
                core = profile
        except Exception as exc:
            logger.warning("faculty profile fetch failed for %s: %s", kerberos, exc)

        dept = core.get("department") or {}
        result: dict[str, Any] = {
            "resolved_from_query": query,
            "profile": {
                "name": core.get("name"),
                "kerberos": kerberos,
                "profile_url": f"/faculty/{kerberos}",
                "email": core.get("email"),
                "designation": core.get("designation"),
                "department": dept.get("name") if isinstance(dept, dict) else dept,
                "h_index": core.get("hIndex"),
                "total_citations": core.get("citationCount"),
                "research_areas": core.get("research_areas") or [],
                "scopus_id": core.get("scopusId"),
                "google_scholar_id": core.get("googleScholarId"),
                "working_from_year": core.get("workingFromYear"),
            },
        }

        # Latest papers (newest-first) from the research summary.
        try:
            summary = await dir_client.research_summary(kerberos)
            latest = _latest_papers(summary)
            total_papers = (summary.get("stats") or {}).get("totalPapers", len(latest))
            result["papers"] = {"total": total_papers, "showing": len(latest), "latest": latest}
        except Exception as exc:
            logger.warning("research summary fetch failed for %s: %s", kerberos, exc)
            result["papers"] = {"total": None, "showing": 0, "latest": []}

        # Latest patents/IP: inventor-scoped IP search, exactly like the profile
        # page's Patents section (kerberos filter + name in the inventor field).
        try:
            ip_query = _TITLE_PREFIX.sub("", core.get("name") or query).strip()
            ip_data = await ip_client.ip_search({
                "query": ip_query,
                "search_in": ["inventor"],
                "filters": {"kerberos": kerberos},
                "mode": "basic",
                "sort": "date",
                "page": 1,
                "per_page": _LATEST,
            })
            ip_results = ip_data.get("results", []) or []
            ip_total = (ip_data.get("pagination") or {}).get("total", len(ip_results))
            patents = [{
                "title": p.get("title"),
                "type_of_ip": p.get("type_of_ip"),
                "year": p.get("publication_year"),
                "application_number": p.get("application_number"),
                "field_of_invention": p.get("field_of_invention"),
            } for p in ip_results[:_LATEST]]
            result["patents"] = {"total": ip_total, "showing": len(patents), "latest": patents}
        except Exception as exc:
            logger.warning("faculty IP search failed for %s: %s", kerberos, exc)
            result["patents"] = {"total": None, "showing": 0, "latest": []}

        # Note whether other people also matched, so the bot can offer them if the
        # top hit isn't who the user meant.
        others = [h.get("name") for h in hits[1:4] if h.get("name")]
        if others:
            result["other_matches"] = others

        # Budget guard: shed patents first, then papers, if oversized.
        output = json.dumps(result, default=str)
        while len(output) > cap and result["patents"]["latest"]:
            result["patents"]["latest"].pop()
            result["patents"]["showing"] = len(result["patents"]["latest"])
            output = json.dumps(result, default=str)
        while len(output) > cap and len(result["papers"]["latest"]) > 3:
            result["papers"]["latest"].pop()
            result["papers"]["showing"] = len(result["papers"]["latest"])
            output = json.dumps(result, default=str)
        return output

    return annotate_tool(
        get_faculty_profile,
        thinking_label="Loading faculty profile",
        token_cap=cap,
    )
