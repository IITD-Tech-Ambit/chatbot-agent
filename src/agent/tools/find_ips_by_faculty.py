"""find_ips_by_faculty tool — patents/IP for a specific IIT Delhi faculty member."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent.tools.deps import ToolDeps
from agent.tools.meta import annotate_tool


class FindIpsByFacultyArgs(BaseModel):
    name: str = Field(description="The faculty member's name (with or without titles like Prof./Dr.)")
    kerberos: Optional[str] = Field(default=None, description="The faculty member's kerberos ID, if known")


def _summarize(docs: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: Counter = Counter()
    by_year: Counter = Counter()
    for d in docs:
        if d.get("type_of_ip"):
            by_type[d["type_of_ip"]] += 1
        if d.get("publication_year"):
            by_year[d["publication_year"]] += 1
    return {
        "by_type": [{"type": t, "count": c} for t, c in by_type.most_common()],
        "by_year": sorted(
            ({"year": y, "count": c} for y, c in by_year.items()),
            key=lambda x: x["year"],
            reverse=True,
        ),
    }


def build_tool(deps: ToolDeps) -> BaseTool:
    ip_repo = deps.ip_repo
    faculty_repo = deps.faculty_repo
    cap = deps.config.TOKEN_CAP_IPS_BY_FACULTY

    @tool(args_schema=FindIpsByFacultyArgs)
    async def find_ips_by_faculty(name: str, kerberos: str | None = None) -> str:
        """Find patents and other IP filed by a specific IIT Delhi faculty member, with a summary of their filings by type and year."""
        if ip_repo is None:
            return json.dumps({"ips": [], "error": "IP lookup is not available"})

        from agent.guardrails.guardrails import faculty_name_matches, name_tokens

        resolved_kerberos = (kerberos or "").lower().strip() or None
        faculty_ref = None
        display_name = name

        if not resolved_kerberos:
            tokens = name_tokens(name)
            if not tokens:
                return json.dumps({"error": "No valid faculty name provided."})
            matches = await faculty_repo.text_search(" ".join(tokens), limit=5)
            if not matches:
                matches = await faculty_repo.regex_search(tokens, limit=5)
            validated = [
                m for m in matches
                if faculty_name_matches(name, m.get("firstName", ""), m.get("lastName", ""))
            ]
            if validated:
                f = validated[0]
                resolved_kerberos = (f.get("email") or "").split("@")[0].lower() or None
                faculty_ref = str(f.get("_id")) if f.get("_id") else None
                display_name = f"{f.get('title', '')} {f.get('firstName', '')} {f.get('lastName', '')}".strip()

        docs = await ip_repo.find_by_inventor(
            kerberos=resolved_kerberos,
            name=None if resolved_kerberos else name,
            faculty_ref=faculty_ref,
        )
        if not docs:
            return json.dumps({
                "faculty": display_name,
                "kerberos": resolved_kerberos,
                "total": 0,
                "ips": [],
                "message": f'No patents or IP filings found for "{display_name}".',
            })

        dept_ids = {d.get("department") for d in docs if d.get("department")}
        dept_name_map = await ip_repo.resolve_department_names(dept_ids)

        ips = [
            {
                "citation_index": i + 1,
                "id": str(d.get("_id", "")),
                "application_number": d.get("application_number"),
                "title": d.get("title"),
                "type_of_ip": d.get("type_of_ip"),
                "department": dept_name_map.get(d.get("department")),
                "classification": (d.get("classification") or [])[:4],
                "publication_year": d.get("publication_year"),
                "filing_date": d.get("filing_date"),
                "inventors": [inv.get("name") for inv in (d.get("inventors") or []) if inv.get("name")][:5],
            }
            for i, d in enumerate(docs)
        ]

        result = {
            "faculty": display_name,
            "kerberos": resolved_kerberos,
            "profile_url": f"/faculty/{resolved_kerberos}" if resolved_kerberos else None,
            "total": len(docs),
            "summary": _summarize(docs),
            "ips": ips,
        }

        output = json.dumps(result, default=str)
        while len(output) > cap and result["ips"]:
            result["ips"].pop()
            output = json.dumps(result, default=str)
        return output

    return annotate_tool(
        find_ips_by_faculty,
        thinking_label="Finding faculty patents & IP",
        token_cap=cap,
    )
