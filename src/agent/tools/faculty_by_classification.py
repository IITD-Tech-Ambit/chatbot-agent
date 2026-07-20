"""faculty_by_classification tool — faculty publishing in a theme / domain."""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import BaseTool, tool

from agent.tools.deps import ToolDeps
from agent.tools.meta import annotate_tool

_MAX_FACULTY = 20


def build_tool(deps: ToolDeps) -> BaseTool:
    taxonomy_repo = deps.taxonomy_repo
    faculty_repo = deps.faculty_repo
    cap = deps.config.TOKEN_CAP_CLASSIFICATION_FACULTY

    @tool
    async def faculty_by_classification(
        theme: Optional[str] = None,
        domain: Optional[str] = None,
        department: Optional[str] = None,
    ) -> str:
        """List IIT Delhi faculty who publish in a specific thematic area and/or
        research domain, ranked by h-index. Use for "which professors work in
        the Energy theme", "faculty in the Machine Learning domain", "who in
        Physics works on Photonics". Provide at least one of `theme` or `domain`
        (optionally `department`). Backed by the precomputed classification
        roster, so counts are exact."""
        if taxonomy_repo is None:
            return json.dumps({"faculty": [], "error": "Classification data is not available"})
        if not theme and not domain:
            return json.dumps({"faculty": [], "error": "Provide a theme and/or domain."})

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
            members = await taxonomy_repo.config_members(
                theme_id=theme_id, domain_id=domain_id, department_id=dept_id
            )
            kerberos_list = (members or {}).get("kerberos_list", [])
            faculty_total = (members or {}).get("faculty_total", len(kerberos_list))
            top = kerberos_list[:_MAX_FACULTY]
            name_map = await faculty_repo.get_kerberos_to_faculty_map(top) if top else {}
        except Exception as exc:
            return json.dumps({"faculty": [], "error": f"Lookup failed: {type(exc).__name__}"})

        faculty = []
        for k in top:
            info = name_map.get(k, {})
            faculty.append({
                "name": info.get("name") or k,
                "kerberos": k,
                "department": info.get("department", ""),
                "profile_url": f"/faculty/{k}",
            })

        result = {
            "theme": theme_name, "domain": domain_name, "department": department,
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
