"""faculty_theme_breakdown tool — one professor's papers across themes/domains."""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent.tools.deps import ToolDeps
from agent.tools.meta import annotate_tool

_MAX_DOMAINS = 10


class FacultyThemeBreakdownArgs(BaseModel):
    faculty_name: str = Field(description="Full name of the IIT Delhi professor")


def _kerberos_from_email(email: str) -> str:
    if email and "@" in email:
        return email.split("@")[0].strip().lower()
    return ""


def build_tool(deps: ToolDeps) -> BaseTool:
    taxonomy_repo = deps.taxonomy_repo
    faculty_repo = deps.faculty_repo
    cap = deps.config.TOKEN_CAP_THEME_BREAKDOWN

    @tool(args_schema=FacultyThemeBreakdownArgs)
    async def faculty_theme_breakdown(faculty_name: str) -> str:
        """Break down a specific IIT Delhi professor's papers across the research
        thematic areas and domains they publish in. Use for "what areas does
        Prof X work in", "Prof X's research profile / breakdown", "which themes
        does Prof X publish in". Names the person; returns their per-theme and
        per-domain paper counts."""
        if taxonomy_repo is None:
            return json.dumps({"themes": [], "domains": [], "error": "Classification data is not available"})

        tokens = [t for t in (faculty_name or "").split() if t]
        if not tokens:
            return json.dumps({"themes": [], "domains": [], "error": "Provide a faculty name."})

        try:
            matches = await faculty_repo.compound_name_search(tokens, limit=1)
        except Exception as exc:
            return json.dumps({"themes": [], "domains": [], "error": f"Faculty lookup failed: {type(exc).__name__}"})
        if not matches:
            return json.dumps({"themes": [], "domains": [], "error": f'No IIT Delhi faculty matching "{faculty_name}" was found.'})

        fac = matches[0]
        kerberos = _kerberos_from_email(fac.get("email", ""))
        display = " ".join(p for p in [fac.get("title", ""), fac.get("firstName", ""), fac.get("lastName", "")] if p).strip()
        if not kerberos:
            return json.dumps({"themes": [], "domains": [], "error": f'Could not resolve a kerberos id for "{display}".'})

        try:
            theme_rows = await taxonomy_repo.theme_distribution(kerberos=kerberos)
            domain_rows = await taxonomy_repo.domain_distribution(kerberos=kerberos, limit=_MAX_DOMAINS)
            theme_names, domain_names = await taxonomy_repo.name_maps()
        except Exception as exc:
            return json.dumps({"themes": [], "domains": [], "error": f"Aggregation failed: {type(exc).__name__}"})

        themes = [{"theme": theme_names.get(str(r.get("_id")), "Unclassified"), "paper_count": r.get("count", 0)} for r in theme_rows]
        domains = [{"domain": domain_names.get(str(r.get("_id")), "Unclassified"), "paper_count": r.get("count", 0)} for r in domain_rows]
        total = sum(t["paper_count"] for t in themes)

        result = {
            "faculty_name": display,
            "kerberos": kerberos,
            "profile_url": f"/faculty/{kerberos}",
            "total_classified_papers": total,
            "themes": themes,
            "domains": domains,
        }
        output = json.dumps(result, default=str)
        while len(output) > cap and result["domains"]:
            result["domains"].pop()
            output = json.dumps(result, default=str)
        return output

    return annotate_tool(
        faculty_theme_breakdown,
        thinking_label="Analyzing researcher's areas",
        token_cap=cap,
    )
