"""search_papers tool — semantic search over IIT Delhi research papers."""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent.repositories.research_repo import _resolve_paper_url
from agent.tools.meta import annotate_tool
from agent.tools.deps import ToolDeps


class SearchPapersArgs(BaseModel):
    query: str = Field(description="Research topic or question to search for")
    year_from: Optional[int] = Field(default=None, description="Only papers published in or after this year")
    year_to: Optional[int] = Field(default=None, description="Only papers published in or before this year")


def build_tool(deps: ToolDeps) -> BaseTool:
    retriever = deps.retriever
    faculty_repo = deps.faculty_repo
    cap = deps.config.TOKEN_CAP_SEARCH_PAPERS

    @tool(args_schema=SearchPapersArgs)
    async def search_papers(query: str, year_from: int | None = None, year_to: int | None = None) -> str:
        """Semantic search over IIT Delhi research papers. Use for questions about research content, findings, or topics."""
        try:
            papers = await retriever.retrieve(query, abstract_max_chars=150)
        except Exception as exc:
            return json.dumps({"papers": [], "error": f"Retrieval failed: {type(exc).__name__}"})

        if year_from:
            papers = [p for p in papers if p.get("publication_year") is None or p["publication_year"] >= year_from]
        if year_to:
            papers = [p for p in papers if p.get("publication_year") is None or p["publication_year"] <= year_to]

        for i, p in enumerate(papers):
            p["citation_index"] = i + 1

        kerberoses = [p["kerberos"] for p in papers if p.get("kerberos")]
        faculty_map = await faculty_repo.get_kerberos_to_faculty_map(kerberoses) if kerberoses else {}

        result = {
            "papers": [
                {
                    "citation_index": p["citation_index"],
                    "id": p.get("id", ""),
                    "title": p["title"],
                    "authors": p["authors"][:3],
                    "year": p.get("publication_year"),
                    "document_type": p.get("document_type"),
                    "field": p.get("field_associated"),
                    "citations": p.get("citation_count", 0),
                    "link": p.get("link"),
                    "document_scopus_id": p.get("document_scopus_id"),
                    "document_eid": p.get("document_eid"),
                    "url": _resolve_paper_url(p),
                    "kerberos": p.get("kerberos"),
                    "faculty_name": faculty_map.get(p.get("kerberos", ""), {}).get("name"),
                    "abstract": p["abstract"],
                }
                for p in papers
            ]
        }

        output = json.dumps(result, default=str)
        while len(output) > cap and result["papers"]:
            result["papers"].pop()
            output = json.dumps(result, default=str)
        return output

    return annotate_tool(
        search_papers,
        thinking_label="Searching indexed publications",
        token_cap=cap,
    )
