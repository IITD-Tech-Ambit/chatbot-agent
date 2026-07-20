"""papers_by_classification tool — papers in a thematic area / domain."""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import BaseTool, tool

from agent.repositories.research_repo import _resolve_paper_url
from agent.tools.deps import ToolDeps
from agent.tools.meta import annotate_tool


def build_tool(deps: ToolDeps) -> BaseTool:
    taxonomy_repo = deps.taxonomy_repo
    faculty_repo = deps.faculty_repo
    cap = deps.config.TOKEN_CAP_CLASSIFICATION_PAPERS

    @tool
    async def papers_by_classification(
        theme: Optional[str] = None,
        domain: Optional[str] = None,
        department: Optional[str] = None,
    ) -> str:
        """List IIT Delhi papers CLASSIFIED into a specific thematic area and/or
        research domain, most recent and most cited first. Use when the user
        names a taxonomy category — e.g. "papers in the Energy theme", "show
        papers in the Machine Learning domain", "Photonics papers from Physics".
        Provide at least one of `theme` or `domain` (optionally `department`).
        This is CATEGORY browsing by the fixed classification, NOT free-text
        topic search — for an arbitrary topic or keyword use `search_papers`."""
        if taxonomy_repo is None:
            return json.dumps({"papers": [], "error": "Classification data is not available"})
        if not theme and not domain:
            return json.dumps({"papers": [], "error": "Provide a theme and/or domain to browse by classification."})

        theme_id = domain_id = dept_ref = None
        theme_name = domain_name = None
        if theme:
            td = await taxonomy_repo.resolve_theme(theme)
            if not td:
                return json.dumps({"papers": [], "error": f'No thematic area matching "{theme}" was found.'})
            theme_id, theme_name = td["_id"], td.get("name")
        if domain:
            dd = await taxonomy_repo.resolve_domain(domain)
            if not dd:
                return json.dumps({"papers": [], "error": f'No research domain matching "{domain}" was found.'})
            domain_id, domain_name = dd["_id"], dd.get("name")
        if department:
            dept = await faculty_repo.find_department(department)
            if not dept:
                return json.dumps({"papers": [], "error": f'No department matching "{department}" was found.'})
            dept_ref = dept["_id"]

        try:
            docs, total = await taxonomy_repo.papers_in_context(
                theme_id=theme_id, domain_id=domain_id, department_ref=dept_ref, limit=10
            )
        except Exception as exc:
            return json.dumps({"papers": [], "error": f"Lookup failed: {type(exc).__name__}"})

        papers = []
        for i, d in enumerate(docs):
            abstract = d.get("abstract") or ""
            if len(abstract) > 150:
                abstract = abstract[:150] + "..."
            papers.append({
                "citation_index": i + 1,
                "id": str(d.get("_id", "")),
                "title": d.get("title", ""),
                "year": d.get("publication_year"),
                "document_type": d.get("document_type"),
                "citations": d.get("citation_count", 0),
                "topics": (d.get("classification") or {}).get("topics", []),
                "url": _resolve_paper_url(d),
                "abstract": abstract,
            })

        result = {
            "theme": theme_name, "domain": domain_name, "department": department,
            "total_matching": total, "papers": papers,
        }
        output = json.dumps(result, default=str)
        while len(output) > cap and result["papers"]:
            result["papers"].pop()
            output = json.dumps(result, default=str)
        return output

    return annotate_tool(
        papers_by_classification,
        thinking_label="Finding papers by classification",
        token_cap=cap,
    )
