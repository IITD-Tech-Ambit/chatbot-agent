"""search_research — the Explore page's advanced paper search, as a bot tool.

Wraps search-api `POST /api/v1/search`: hybrid BM25 + semantic search over IIT
Delhi research papers with the full facet-filter / sort / field-scope surface,
returning matching papers, related faculty, and facet distributions.
"""

from __future__ import annotations

import asyncio
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
_SORTS = {"relevance", "date", "citations", "impact", "normalized"}
_SEARCH_IN = {"title", "abstract", "author", "subject_area", "field"}

# The People list: faculty aggregated across ALL matching papers, returned as a
# flat top-N ranked by matching-paper count (department shown on each entry).
_MAX_FACULTY = 10
_MAX_DEPT_TOTALS = 6


async def _build_faculty_sections(people: dict, faculty_repo) -> dict:
    """Flatten the faculty-for-query aggregation into the top faculty by
    matching-paper count, each tagged with their department, plus per-department
    totals. All counts span the whole result set, not the returned page."""
    depts = people.get("departments") or []

    flat: list[dict] = []
    for d in depts:
        dept_name = d.get("name")
        for f in d.get("faculty", []) or []:
            flat.append({
                "name": f.get("name"),
                "papers": f.get("paper_count", 0),
                "department": dept_name,
                "author_id": f.get("author_id"),
            })
    flat.sort(key=lambda f: f.get("papers", 0), reverse=True)
    top = flat[:_MAX_FACULTY]

    # Resolve Scopus author_id -> faculty record so names can link to profiles.
    expert_ids = [f["author_id"] for f in top if f.get("author_id")]
    by_expert: dict = {}
    if expert_ids:
        try:
            docs = await faculty_repo.find_by_expert_ids(expert_ids)
            by_expert = {d.get("expert_id"): d for d in docs}
        except Exception as exc:  # profile links are a nicety, never fatal
            logger.warning("faculty profile resolution failed: %s", exc)

    top_faculty = []
    for f in top:
        doc = by_expert.get(f.get("author_id")) or {}
        kerb = (doc.get("email") or "").split("@")[0].lower() or None
        top_faculty.append({
            "name": f.get("name"),
            "papers": f.get("papers"),
            "department": f.get("department"),
            "kerberos": kerb,
            "profile_url": f"/faculty/{kerb}" if kerb else None,
        })

    dept_totals = sorted(
        (
            {"department": d.get("name"), "papers": d.get("total_paper_count", 0)}
            for d in depts if d.get("name")
        ),
        key=lambda d: d["papers"],
        reverse=True,
    )[:_MAX_DEPT_TOTALS]

    return {
        "total_faculty": people.get("total_faculty"),
        "total_matching_papers": people.get("total_matching_papers"),
        "showing": len(top_faculty),
        "note": (
            "Paper counts are across ALL matching papers for this query, not just "
            "the papers listed above. top_faculty is the overall top "
            f"{_MAX_FACULTY} researchers ranked by matching papers."
        ),
        "top_faculty": top_faculty,
        "papers_by_department": dept_totals,
    }


class SearchResearchArgs(BaseModel):
    query: str = Field(description="The research topic, keywords, or a faculty/author name to search for.")
    year_from: Optional[int] = Field(default=None, description="Only papers published in or after this year.")
    year_to: Optional[int] = Field(default=None, description="Only papers published in or before this year.")
    document_types: Optional[list[str]] = Field(default=None, description="Restrict to document types, e.g. ['Article','Review','Conference Paper'].")
    subject_areas: Optional[list[str]] = Field(default=None, description="Restrict to Scopus subject-area labels.")
    sort: Optional[str] = Field(default=None, description="One of: relevance (default), date (newest), citations (most cited), impact (citation-weighted).")
    search_in: Optional[list[str]] = Field(default=None, description="Restrict matching to these fields only: title, abstract, author, subject_area, field. Use ['author'] to search by a person's name.")
    first_author_only: Optional[bool] = Field(default=None, description="Only papers where the matched IITD author is the first author.")
    interdisciplinary: Optional[bool] = Field(default=None, description="Only papers spanning 3+ subject areas.")


def _author_names(authors: list, limit: int = 3) -> list[str]:
    names: list[str] = []
    for a in authors or []:
        if isinstance(a, dict):
            n = a.get("author_name") or a.get("name")
        else:
            n = a
        if n:
            names.append(n)
        if len(names) >= limit:
            break
    return names


def _facet_summary(facets: dict) -> dict:
    if not isinstance(facets, dict):
        return {}
    out: dict = {}
    for key in ("document_types", "fields", "subject_areas", "year_ranges"):
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
    faculty_repo = deps.faculty_repo
    cap = deps.config.TOKEN_CAP_SEARCH_PAPERS

    @tool(args_schema=SearchResearchArgs)
    async def search_research(
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        document_types: Optional[list[str]] = None,
        subject_areas: Optional[list[str]] = None,
        sort: Optional[str] = None,
        search_in: Optional[list[str]] = None,
        first_author_only: Optional[bool] = None,
        interdisciplinary: Optional[bool] = None,
    ) -> str:
        """Search IIT Delhi research PAPERS — the primary tool for any question
        about publications, research topics, or which professors work on a
        subject. This is the same hybrid (keyword + semantic) engine as the
        Explore page, so it understands meaning, not just exact words.

        Use it for: "papers on perovskite solar cells", "what has Prof X
        published about Y" (put the name in `query` and add search_in=['author']
        or just include the name), "recent reviews on graphene since 2020",
        "most-cited papers on drug delivery", "who works on wearable
        electronics" (answer that from the `faculty` section).

        Honor the user's constraints with the knobs:
          - year_from / year_to for date ranges.
          - sort: 'citations' for "most cited", 'date' for "latest/recent",
            else leave default (relevance).
          - document_types (['Review'], ['Conference Paper'], ...).
          - search_in=['author'] to search specifically by a person's name;
            ['title'] / ['abstract'] to restrict where the keywords must appear.
          - first_author_only / interdisciplinary flags.

        There is NO department filter for papers. For "papers of the <X>
        department on <topic>", search the TOPIC only (do not try to filter by
        department) and then read `faculty.top_faculty` — each entry carries the
        researcher's department. Never claim a filter you did not apply.

        Returns:
          - `papers` — the top 10 matching papers (title, authors, year,
            citations, link).
          - `faculty` — the People list. `top_faculty` is the top 10 researchers
            ranked by how many matching papers they have, each with their
            `department`; `papers_by_department` gives per-department totals;
            `total_faculty` is how many researchers matched overall. ALL these
            counts cover the ENTIRE result set, not just the 10 papers above —
            use them for "who works on X" and "which department leads on X".
          - `facets` — year/type/field distributions.
        Do NOT use this for patents/IP — use search_ip."""
        filters: dict = {}
        if year_from:
            filters["year_from"] = year_from
        if year_to:
            filters["year_to"] = year_to
        if document_types:
            filters["document_types"] = document_types
        if subject_areas:
            filters["subject_area"] = subject_areas
        if first_author_only:
            filters["first_author_only"] = True
        if interdisciplinary:
            filters["interdisciplinary"] = True

        body: dict = {"query": query, "mode": "advanced", "per_page": _PER_PAGE}
        body["sort"] = sort if sort in _SORTS else "relevance"
        if filters:
            body["filters"] = filters
        if search_in:
            valid = [f for f in search_in if f in _SEARCH_IN]
            if valid:
                body["search_in"] = valid

        # Papers and the People aggregation run in parallel — the People list is
        # built from the FULL matching corpus, so it must use the same filters.
        try:
            data, people = await asyncio.gather(
                client.search(body),
                client.faculty_for_query(
                    query,
                    mode="advanced",
                    search_in=body.get("search_in"),
                    filters=filters or None,
                ),
                return_exceptions=True,
            )
        except Exception as exc:
            logger.warning("search_research call failed: %s", exc)
            return json.dumps({"papers": [], "error": f"Search failed: {type(exc).__name__}"})

        if isinstance(data, BaseException):
            logger.warning("search_research paper search failed: %s", data)
            return json.dumps({"papers": [], "error": f"Search failed: {type(data).__name__}"})
        if isinstance(people, BaseException):
            # The faculty sidebar is supplementary — never fail the whole search.
            logger.warning("faculty-for-query failed: %s", people)
            people = {}

        results = data.get("results", []) or []
        papers = []
        for i, p in enumerate(results):
            papers.append({
                "citation_index": i + 1,
                "id": p.get("_id") or p.get("open_search_id", ""),
                "title": p.get("title", ""),
                "authors": _author_names(p.get("authors", [])),
                "year": p.get("publication_year"),
                "document_type": p.get("document_type"),
                "field": p.get("field_associated"),
                "citations": p.get("citation_count", 0),
                "link": p.get("link"),
                "document_scopus_id": p.get("document_scopus_id"),
                "document_eid": p.get("document_eid"),
                "kerberos": p.get("kerberos"),
            })

        faculty_sections = await _build_faculty_sections(people or {}, faculty_repo)

        pagination = data.get("pagination", {}) or {}
        result = {
            "query": query,
            "sorted_by": body["sort"],
            "total_matching_papers": pagination.get("total", len(papers)),
            "showing": len(papers),
            "papers": papers,
            "faculty": faculty_sections,
            "facets": _facet_summary(data.get("facets", {})),
            "suggestions": data.get("suggestions", [])[:5],
        }
        if not papers and data.get("message"):
            result["message"] = data["message"]

        output = json.dumps(result, default=str)
        # Shed papers first, then trailing department sections, so an oversized
        # payload degrades gracefully instead of losing the faculty list wholesale.
        while len(output) > cap and len(result["papers"]) > 3:
            result["papers"].pop()
            result["showing"] = len(result["papers"])
            output = json.dumps(result, default=str)
        while len(output) > cap and len(result["faculty"].get("papers_by_department", [])) > 1:
            result["faculty"]["papers_by_department"].pop()
            output = json.dumps(result, default=str)
        while len(output) > cap and result["papers"]:
            result["papers"].pop()
            result["showing"] = len(result["papers"])
            output = json.dumps(result, default=str)
        return output

    return annotate_tool(
        search_research,
        thinking_label="Searching IIT Delhi research",
        token_cap=cap,
    )
