"""experts_by_research_area — the Research Areas "Browse experts" view.

Wraps the search-api taxonomy /faculty endpoint: given a thematic area (required)
and optional domain / department, returns the IIT Delhi experts working in that
area, resolved to names via the faculty directory. Mirrors the page: to view
experts you must at least pick a thematic area (after an optional department).
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
from agent.transports.taxonomy import TaxonomyClient

logger = logging.getLogger(__name__)

_DEFAULT_LIMIT = 15
_MAX_LIMIT = 40


class ExpertsByResearchAreaArgs(BaseModel):
    theme: str = Field(
        description="The thematic area to find experts in (REQUIRED), e.g. 'Energy, Sustainability & Climate Change'.",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Optionally narrow to a domain, e.g. 'Power Electronics'.",
    )
    department: Optional[str] = Field(
        default=None,
        description="Optionally restrict to a department, e.g. 'Electrical Engineering'.",
    )
    limit: Optional[int] = Field(
        default=None,
        description="How many experts to return (default 15, max 40).",
    )


def build_tool(deps: ToolDeps) -> BaseTool:
    client = TaxonomyClient(deps.config.SEARCH_API_URL)
    faculty_repo = deps.faculty_repo
    cap = deps.config.TOKEN_CAP_CLASSIFICATION_FACULTY

    @tool(args_schema=ExpertsByResearchAreaArgs)
    async def experts_by_research_area(
        theme: str,
        domain: Optional[str] = None,
        department: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        """List the IIT Delhi EXPERTS (faculty) working in a Research Area — the
        Research Areas page's "Browse experts" view. Use for "which professors
        work in the Energy theme", "experts in the Machine Learning domain",
        "faculty in the Healthcare theme in Electrical Engineering".

        Filtering has three levels, applied per the user's request:
          - `department` (optional) — restrict to one department.
          - `theme` (REQUIRED) — the thematic area; experts always need at
            least this, exactly like the page.
          - `domain` (optional) — further narrow within the theme.

        If the user names only a DOMAIN, look up its thematic area in the
        structural reference in your system prompt (each domain belongs to
        exactly one) and pass both. Only if no thematic area can be identified
        at all, ask the user which one — the reference lists them. Never guess.

        Returns the experts (name, department, profile link) and the area's
        paper/faculty totals."""
        n = _DEFAULT_LIMIT if not limit or limit < 1 else min(int(limit), _MAX_LIMIT)
        try:
            themes = await client.themes()
            tmatch = resolve_node(themes, theme)
            if not tmatch:
                return json.dumps({
                    "error": f'No thematic area matching "{theme}" was found. Ask the user to pick one of the available thematic areas.',
                    "available_themes": [t.get("name") for t in themes],
                    "experts": [],
                })
            theme_slug, theme_name = tmatch.get("slug"), tmatch.get("name")

            dept_code = dept_name = None
            if department:
                depts = [d for d in await client.departments() if d.get("code") and d.get("name")]
                dmatch = resolve_node(depts, department, name_key="name", slug_key="code")
                if not dmatch:
                    return json.dumps({"error": f'No department matching "{department}" was found.', "experts": []})
                dept_code, dept_name = dmatch.get("code"), dmatch.get("name")

            domain_slug = domain_name = None
            if domain:
                # Domains cascade under the theme (like the page): only match within
                # this theme's domain list, never a cross-theme domain.
                domains = await client.domains(theme=theme_slug, department=dept_code)
                dmatch = resolve_node(domains, domain)
                if not dmatch:
                    return json.dumps({
                        "error": f'"{domain}" is not a domain under {theme_name}. Offer the user the available domains, or drop the domain filter.',
                        "available_domains": [d.get("name") for d in domains],
                        "experts": [],
                    })
                domain_slug, domain_name = dmatch.get("slug"), dmatch.get("name")

            fac = await client.faculty(
                theme=theme_slug, domain=domain_slug, department=dept_code, page=1, per_page=n
            )
            kerberos_list = (fac.get("kerberos_list") or [])[:n]
            faculty_total = fac.get("faculty_total", len(kerberos_list))

            name_map = await faculty_repo.get_kerberos_to_faculty_map(kerberos_list) if kerberos_list else {}
            experts = []
            for k in kerberos_list:
                info = name_map.get(k, {})
                experts.append({
                    "name": info.get("name") or k,
                    "kerberos": k,
                    "department": info.get("department", ""),
                    "profile_url": f"/faculty/{k}",
                })

            counts = await client.counts(theme=theme_slug, domain=domain_slug, department=dept_code)

            result = {
                "theme": theme_name,
                "domain": domain_name,
                "department": dept_name,
                "paper_count": counts.get("paper_count"),
                "faculty_total": faculty_total,
                "showing": len(experts),
                "experts": experts,
                "selection": {
                    "theme": {"slug": theme_slug, "name": theme_name},
                    "domain": {"slug": domain_slug, "name": domain_name} if domain_slug else None,
                    "department": {"code": dept_code, "name": dept_name} if dept_code else None,
                },
            }
            output = json.dumps(result, default=str)
            while len(output) > cap and result["experts"]:
                result["experts"].pop()
                result["showing"] = len(result["experts"])
                output = json.dumps(result, default=str)

            # Be explicit that a top-N slice is a complete, usable answer — the
            # model must list these rather than calling the result "truncated".
            if result["experts"]:
                shown = result["showing"]
                result["note"] = (
                    f"These are the top {shown} experts"
                    + (f" of {faculty_total} in this area" if faculty_total > shown else "")
                    + ". This is a complete, usable result — list every expert above."
                )
                output = json.dumps(result, default=str)
            return output
        except Exception as exc:
            logger.warning("experts_by_research_area failed: %s", exc)
            return json.dumps({"error": f"Lookup failed: {type(exc).__name__}", "experts": []})

    return annotate_tool(
        experts_by_research_area,
        thinking_label="Finding experts by research area",
        token_cap=cap,
    )
