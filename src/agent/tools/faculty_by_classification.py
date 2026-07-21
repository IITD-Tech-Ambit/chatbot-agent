"""faculty_by_classification tool — faculty publishing in a theme / domain."""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import BaseTool, tool

from agent.tools.deps import ToolDeps
from agent.tools.meta import annotate_tool

_DEFAULT_LIMIT = 10
_MAX_LIMIT = 30


def build_tool(deps: ToolDeps) -> BaseTool:
    taxonomy_repo = deps.taxonomy_repo
    faculty_repo = deps.faculty_repo
    cap = deps.config.TOKEN_CAP_CLASSIFICATION_FACULTY

    @tool
    async def faculty_by_classification(
        theme: Optional[str] = None,
        domain: Optional[str] = None,
        department: Optional[str] = None,
        sort_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        """List IIT Delhi faculty who publish in a specific FIXED classification
        category — one of the 9 thematic areas or 80 research domains. Use for
        "which professors work in the Energy theme", "faculty in the Machine
        Learning domain", "top 5 professors by paper count in Manufacturing",
        "who in Physics works on Photonics".

        Use ONLY when the named area is one of those official categories. For an
        arbitrary free-text topic that is NOT a fixed category (e.g. "wearable
        electronics", "graphene batteries"), use `find_faculty_for_topic`.

        Provide at least one of `theme` or `domain` (optionally `department`).
        Knobs:
          - `sort_by`: "h_index" (default — ranks by researcher h-index) or
            "paper_count" (ranks by how many papers each faculty has IN this
            theme/domain). Use "paper_count" whenever the user asks to rank by
            number of papers/publications in the area.
          - `limit`: how many faculty to return (default 10, max 30). Set it to
            the N the user asks for (e.g. "top 5" → 5)."""
        if taxonomy_repo is None:
            return json.dumps({"faculty": [], "error": "Classification data is not available"})
        if not theme and not domain:
            return json.dumps({"faculty": [], "error": "Provide a theme and/or domain."})

        sort_mode = "paper_count" if (sort_by or "").strip().lower() in ("paper_count", "papers", "paper", "publications") else "h_index"
        n = _DEFAULT_LIMIT if not limit or limit < 1 else min(int(limit), _MAX_LIMIT)

        theme_id = domain_id = dept_id = None
        theme_name = domain_name = None
        if theme:
            td = await taxonomy_repo.resolve_theme(theme)
            if not td:
                return json.dumps({"faculty": [], "error": f'No thematic area matching "{theme}" was found.'})
            theme_id, theme_name = td["_id"], td.get("name")
        if domain:
            dd = await taxonomy_repo.resolve_domain(domain)
            if not dd:
                return json.dumps({"faculty": [], "error": f'No research domain matching "{domain}" was found.'})
            domain_id, domain_name = dd["_id"], dd.get("name")
        if department:
            dept = await faculty_repo.find_department(department)
            if not dept:
                return json.dumps({"faculty": [], "error": f'No department matching "{department}" was found.'})
            dept_id = dept["_id"]

        try:
            if sort_mode == "paper_count":
                rows = await taxonomy_repo.faculty_paper_counts(
                    theme_id=theme_id, domain_id=domain_id, department_ref=dept_id, limit=n
                )
                kerberoses = [r["kerberos"] for r in rows]
                paper_count_map = {r["kerberos"]: r["paper_count"] for r in rows}
                faculty_total = len(kerberoses)
            else:
                members = await taxonomy_repo.config_members(
                    theme_id=theme_id, domain_id=domain_id, department_id=dept_id
                )
                full_list = (members or {}).get("kerberos_list", [])
                faculty_total = (members or {}).get("faculty_total", len(full_list))
                kerberoses = full_list[:n]
                paper_count_map = {}
            name_map = await faculty_repo.get_kerberos_to_faculty_map(kerberoses) if kerberoses else {}
        except Exception as exc:
            return json.dumps({"faculty": [], "error": f"Lookup failed: {type(exc).__name__}"})

        faculty = []
        for k in kerberoses:
            info = name_map.get(k, {})
            entry = {
                "name": info.get("name") or k,
                "kerberos": k,
                "department": info.get("department", ""),
                "profile_url": f"/faculty/{k}",
            }
            if sort_mode == "paper_count":
                entry["papers_in_area"] = paper_count_map.get(k, 0)
            faculty.append(entry)

        result = {
            "theme": theme_name, "domain": domain_name, "department": department,
            "sorted_by": "papers in this area" if sort_mode == "paper_count" else "h-index",
            "faculty_total": faculty_total, "showing": len(faculty), "faculty": faculty,
        }
        output = json.dumps(result, default=str)
        while len(output) > cap and result["faculty"]:
            result["faculty"].pop()
            output = json.dumps(result, default=str)
        return output

    return annotate_tool(
        faculty_by_classification,
        thinking_label="Finding faculty by classification",
        token_cap=cap,
    )
