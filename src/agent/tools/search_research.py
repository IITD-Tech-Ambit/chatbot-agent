"""search_research — operate the Explore page's paper search as a bot tool.

Wraps the SAME search-api the Explore page calls and replicates its client-side
behaviour (sort dropdown, Group-by-Department, author drill-down), so the papers
the tool returns are the same list, in the same order, the user would see on
Explore for the same query + filters (truncated to the top 10).

Explore's visible controls are client-side transforms of a server relevance
page: the "Sort by" dropdown re-sorts the fetched page (relevance | citations),
"Group by Dept" groups it by field, and clicking a professor runs an
author-scoped search. This tool mirrors all three.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from difflib import SequenceMatcher
from typing import Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent.tools.deps import ToolDeps
from agent.tools.meta import annotate_tool
from agent.transports.research_search import ResearchSearchClient

logger = logging.getLogger(__name__)

# Explore loads 20 results per page and applies its client-side sort/group to
# that page; the top 10 of that is what the user sees first. We fetch the same
# window, apply the same transforms, and return the top 10.
_FETCH_PAGE = 20
_RETURN = 10
_SORTS = {"relevance", "citations"}

_MAX_FACULTY = 10
_MAX_DEPT_TOTALS = 6


def _name_tokens(name: str) -> list[str]:
    return [t for t in re.split(r"\s+", (name or "").strip()) if t]


_TITLES = frozenset({"prof", "professor", "dr", "mr", "ms", "mrs", "the"})
_MATCH_THRESHOLD = 0.6


def _name_parts(name: str) -> list[str]:
    """Ordered, lowercased name tokens with titles/punctuation stripped (keeps
    order so the last token can be treated as the surname)."""
    return [t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if t and t not in _TITLES]


def _name_similarity(requested: str, candidate: str) -> float:
    """Score how well faculty full name `candidate` matches the `requested` name.

    Guards against picking a different person who merely shares a surname:
      - Surnames must be CLOSE (SequenceMatcher ≥ 0.8) — this tolerates real
        spelling variants like "Agarwal" vs "Agrawal".
      - EVERY full given-name token (≥3 chars) in the request must appear in the
        candidate's given names (equal or prefix). So "Ashwini K Agarwal" scores
        0 against "A K Agarwala" (given name is just the initial "A"), but high
        against "Ashwini K Agrawal". Initials match by first letter only.
    Returns 0.0 when it is not a genuine match."""
    req, cand = _name_parts(requested), _name_parts(candidate)
    if not req or not cand:
        return 0.0
    sur_sim = SequenceMatcher(None, req[-1], cand[-1]).ratio()
    if sur_sim < 0.8:
        return 0.0
    cand_firsts = cand[:-1]
    for q in req[:-1]:
        if len(q) >= 3:
            ok = any(c == q or c.startswith(q) or (q.startswith(c) and len(c) >= 3) for c in cand_firsts)
        else:  # an initial in the request — match by first letter
            ok = any(c.startswith(q) for c in cand_firsts + [cand[-1]])
        if not ok:
            return 0.0
    return 0.6 * sur_sim + 0.4


def _best_sidebar_author(people: dict, requested: str) -> tuple[Optional[str], Optional[str]]:
    """Resolve `requested` against the topic-scoped People sidebar. The sidebar
    lists only faculty who actually have papers matching the query, and its
    author_id is the one the Explore People list uses — so the deep-link
    highlights the professor exactly like a click. Returns (author_id, name)."""
    best_id = best_name = None
    best_score = 0.0
    for d in people.get("departments", []) or []:
        for f in d.get("faculty", []) or []:
            nm = f.get("name") or ""
            s = _name_similarity(requested, nm)
            if s > best_score and f.get("author_id") is not None:
                best_score, best_id, best_name = s, f.get("author_id"), nm
    if best_id is not None and best_score >= _MATCH_THRESHOLD:
        return str(best_id), best_name
    return None, None


async def _resolve_author(faculty_repo, name: str) -> tuple[Optional[str], Optional[str]]:
    """Fallback resolution via the faculty directory (topic-blind): a professor's
    name → Scopus author_id (+ canonical name). Validates the match with
    `_name_similarity` so a loose surname-only hit never silently returns the
    WRONG person; returns (None, None) when nothing matches well enough."""
    toks = _name_tokens(name)
    if not toks:
        return None, None
    try:
        docs = await faculty_repo.compound_name_search(toks, limit=5)
    except Exception as exc:
        logger.warning("author name resolution failed: %s", exc)
        return None, None
    best: tuple[str, str] | None = None
    best_score = 0.0
    for d in docs:
        sc = d.get("scopus_id")
        scopus = sc[0] if isinstance(sc, list) and sc else (sc if isinstance(sc, str) else None)
        if not scopus:
            continue
        cand = " ".join(x for x in (d.get("firstName"), d.get("lastName")) if x).strip()
        s = _name_similarity(name, cand)
        if s > best_score:
            best_score = s
            full = " ".join(x for x in (d.get("title"), d.get("firstName"), d.get("lastName")) if x).strip()
            best = (str(scopus), full or name)
    if best and best_score >= _MATCH_THRESHOLD:
        return best
    return None, None


def _client_sort(results: list[dict], sort_mode: str) -> list[dict]:
    """Explore's client-side sort of the fetched page: citations = by citation
    count desc; relevance = keep the server order."""
    if sort_mode == "citations":
        return sorted(results, key=lambda p: p.get("citation_count", 0) or 0, reverse=True)
    return results


def _author_names(authors: list, limit: int = 3) -> list[str]:
    names: list[str] = []
    for a in authors or []:
        n = (a.get("author_name") or a.get("name")) if isinstance(a, dict) else a
        if n:
            names.append(n)
        if len(names) >= limit:
            break
    return names


def _shape_paper(p: dict, i: int) -> dict:
    return {
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
    }


def _group_by_department(papers: list[dict]) -> list[dict]:
    """Mirror Explore's Group-by-Dept: bucket the returned papers by field,
    departments alphabetical with 'Other' last. References papers by their
    `citation_index` (not full copies) so the flat `papers` list always keeps
    all 10 and the payload stays small."""
    groups: dict[str, list[int]] = {}
    for p in papers:
        dept = p.get("field") or "Other"
        groups.setdefault(dept, []).append(p.get("citation_index"))
    ordered = sorted(groups.keys(), key=lambda d: (d == "Other", d.lower()))
    return [{"department": d, "paper_indexes": groups[d]} for d in ordered]


async def _build_faculty_sections(people: dict, faculty_repo) -> dict:
    """The Explore People sidebar: top faculty by matching-paper count (across
    the WHOLE result set), each tagged with department, plus per-dept totals."""
    depts = people.get("departments") or []
    flat: list[dict] = []
    for d in depts:
        for f in d.get("faculty", []) or []:
            flat.append({
                "name": f.get("name"),
                "papers": f.get("paper_count", 0),
                "department": d.get("name"),
                "author_id": f.get("author_id"),
            })
    flat.sort(key=lambda f: f.get("papers", 0), reverse=True)
    top = flat[:_MAX_FACULTY]

    expert_ids = [f["author_id"] for f in top if f.get("author_id")]
    by_expert: dict = {}
    if expert_ids:
        try:
            docs = await faculty_repo.find_by_expert_ids(expert_ids)
            by_expert = {d.get("expert_id"): d for d in docs}
        except Exception as exc:
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
        ({"department": d.get("name"), "papers": d.get("total_paper_count", 0)}
         for d in depts if d.get("name")),
        key=lambda d: d["papers"], reverse=True,
    )[:_MAX_DEPT_TOTALS]

    return {
        "total_faculty": people.get("total_faculty"),
        "total_matching_papers": people.get("total_matching_papers"),
        "showing": len(top_faculty),
        "note": (
            "Paper counts span ALL matching papers, not just the papers listed. "
            f"top_faculty is the overall top {_MAX_FACULTY} researchers by matching papers."
        ),
        "top_faculty": top_faculty,
        "papers_by_department": dept_totals,
    }


def _facet_summary(facets: dict) -> dict:
    if not isinstance(facets, dict):
        return {}
    out: dict = {}
    for key in ("document_types", "fields", "subject_areas", "year_ranges"):
        buckets = facets.get(key)
        if isinstance(buckets, list) and buckets:
            out[key] = [
                {"value": b.get("key"), "count": b.get("doc_count", b.get("count"))}
                for b in buckets[:6] if isinstance(b, dict)
            ]
    return out


class SearchResearchArgs(BaseModel):
    query: str = Field(description="The research topic or keywords to search for.")
    year_from: Optional[int] = Field(default=None, description="Only papers published in or after this year.")
    year_to: Optional[int] = Field(default=None, description="Only papers published in or before this year.")
    sort: Optional[str] = Field(default=None, description="The Explore 'Sort by' control: 'relevance' (default) or 'citations' (most-cited first among the results).")
    group_by_department: Optional[bool] = Field(default=None, description="Explore's 'Group by Dept': group the returned papers by their field/department. Ignored when `author` is set.")
    author: Optional[str] = Field(default=None, description="A professor's name to drill into: return only THAT professor's papers matching the query (Explore's click-a-professor view). When set, the People list is omitted.")


def build_tool(deps: ToolDeps) -> BaseTool:
    client = ResearchSearchClient(deps.config.SEARCH_API_URL)
    faculty_repo = deps.faculty_repo
    cap = deps.config.TOKEN_CAP_SEARCH_PAPERS

    @tool(args_schema=SearchResearchArgs)
    async def search_research(
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        sort: Optional[str] = None,
        group_by_department: Optional[bool] = None,
        author: Optional[str] = None,
    ) -> str:
        """Search IIT Delhi research PAPERS — the primary tool for publications,
        research topics, and which researchers work on a subject. Same hybrid
        (keyword + semantic) engine as the Explore page: the papers returned are
        the SAME list, in the SAME order, a user would see on Explore for this
        query + filters (top 10).

        Explore's filters, all supported here:
          - `sort`: 'relevance' (default) or 'citations' (most-cited among the
            results) — this mirrors Explore's Sort-by dropdown.
          - `year_from` / `year_to`: publication-year range.
          - `group_by_department`: group the returned papers by field/department.
          - `author`: a professor's NAME to drill into — returns only that
            professor's papers matching the query (Explore's click-a-professor
            view). When set, the People list is omitted and grouping does not
            apply.

        There is NO department filter for papers. For "papers of the <X>
        department on <topic>", search the TOPIC and read `faculty.top_faculty`
        (each entry has a department). Never claim a filter you didn't apply.

        Returns:
          - `papers` — the top 10 matching papers (title, authors, year,
            citations, link), same order as Explore. When group_by_department
            is on, an extra `papers_grouped` lists each department with the
            `paper_indexes` (into `papers`) that belong to it.
          - `faculty` — the People list (`top_faculty` = top researchers by
            matching papers across the WHOLE result set, `papers_by_department`
            = per-dept totals). OMITTED when `author` is set.
          - `facets` — year/type/field distributions.
        A button to open this exact search on Explore is added automatically.
        Do NOT use this for patents/IP — use search_ip."""
        sort_mode = sort if sort in _SORTS else "relevance"
        filters: dict = {}
        if year_from:
            filters["year_from"] = year_from
        if year_to:
            filters["year_to"] = year_to

        # Server sort is always relevance (Explore fetches the relevance page and
        # sorts client-side); we replicate the client sort below.
        body: dict = {"query": query, "mode": "advanced", "per_page": _FETCH_PAGE, "sort": "relevance"}
        if filters:
            body["filters"] = filters

        author_id = author_name = None
        people: dict | None = None

        try:
            if author:
                # Resolve against the topic-scoped People sidebar FIRST: it lists
                # only faculty who actually have papers matching this query, its
                # fuzzy match tolerates the user's spelling (e.g. Agarwal vs the
                # stored Agrawal), and its author_id is the one Explore uses — so
                # the deep-link highlights the professor. Fall back to the
                # (topic-blind) directory search only if the sidebar has no match.
                try:
                    sidebar = await client.faculty_for_query(query, mode="advanced", filters=filters or None)
                    author_id, author_name = _best_sidebar_author(sidebar, author)
                except Exception as exc:
                    logger.warning("sidebar author lookup failed: %s", exc)
                    author_id = author_name = None
                if not author_id:
                    author_id, author_name = await _resolve_author(faculty_repo, author)
                if not author_id:
                    return json.dumps({
                        "papers": [],
                        "error": f'Could not confidently identify a professor named "{author}" with work matching "{query}". Do not guess a different person.',
                    })
                data = await client.author_scope({**body, "author_id": author_id})
            else:
                data, people_raw = await asyncio.gather(
                    client.search(body),
                    client.faculty_for_query(query, mode="advanced", filters=filters or None),
                    return_exceptions=True,
                )
                if isinstance(data, BaseException):
                    raise data
                people = {} if isinstance(people_raw, BaseException) else people_raw
        except Exception as exc:
            logger.warning("search_research call failed: %s", exc)
            return json.dumps({"papers": [], "error": f"Search failed: {type(exc).__name__}"})

        # Replicate Explore's client-side sort, then take the first 10 shown.
        results = _client_sort(data.get("results", []) or [], sort_mode)
        top = [_shape_paper(p, i) for i, p in enumerate(results[:_RETURN])]

        pagination = data.get("pagination", {}) or {}
        grouped = bool(group_by_department) and not author

        result: dict = {
            "query": query,
            "sorted_by": ("similarity" if author else "relevance") if sort_mode == "relevance" else "citations",
            "total_matching_papers": pagination.get("total", len(top)),
            "showing": len(top),
        }
        if author:
            result["author"] = {"name": author_name, "profile_hint": author}
            result["note"] = f"Papers by {author_name} matching '{query}' (People list omitted for an author drill-down)."
        if grouped:
            result["papers_grouped"] = _group_by_department(top)
            result["papers"] = top  # flat copy for source citations
        else:
            result["papers"] = top
        if people is not None:
            result["faculty"] = await _build_faculty_sections(people or {}, faculty_repo)
        result["facets"] = _facet_summary(data.get("facets", {}))
        if not top and data.get("message"):
            result["message"] = data["message"]

        # Deep-link descriptor for the "Explore all results" button (same filters).
        link_filters: dict = {}
        if year_from:
            link_filters["year_from"] = year_from
        if year_to:
            link_filters["year_to"] = year_to
        result["explore_link"] = {
            "kind": "research",
            "query": query,
            "sort": sort_mode,
            "group_by_department": grouped,
            "filters": link_filters or None,
            "author": {"id": author_id, "name": author_name} if author_id else None,
        }

        output = json.dumps(result, default=str)
        # Degrade gracefully if oversized: shed papers, then dept totals, then more papers.
        def _papers_len() -> int:
            return len(result.get("papers", []))

        while len(output) > cap and _papers_len() > 3:
            result["papers"].pop()
            if grouped:
                result["papers_grouped"] = _group_by_department(result["papers"])
            result["showing"] = _papers_len()
            output = json.dumps(result, default=str)
        while len(output) > cap and isinstance(result.get("faculty"), dict) and len(result["faculty"].get("papers_by_department", [])) > 1:
            result["faculty"]["papers_by_department"].pop()
            output = json.dumps(result, default=str)
        return output

    return annotate_tool(
        search_research,
        thinking_label="Searching IIT Delhi research",
        token_cap=cap,
    )
