"""Structured-query router: fast-path regex patterns that skip the LLM entirely.

About 25% of queries are pure metadata lookups (h-index, citations, faculty list)
that can be answered directly from MongoDB in ~200ms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent.repositories.faculty_repo import FacultyRepository
from agent.repositories.research_repo import ResearchRepository


@dataclass
class RouteMatch:
    handler: str
    capture: str


PATTERNS: list[tuple[re.Pattern, str]] = [  # type: ignore[type-arg]
    # ── top faculty by h-index / citations ──
    (re.compile(r"top\s+\d*\s*(?:professor|faculty|researcher|scientist)s?\s+(?:by\s+)?(?:h[.\-\s]?index|h-index)", re.I), "top_faculty_hindex"),
    (re.compile(r"(?:highest|best|most)\s+h[.\-\s]?index", re.I), "top_faculty_hindex"),
    (re.compile(r"top\s+\d*\s*(?:professor|faculty|researcher|scientist)s?\s+(?:by\s+)?citation", re.I), "top_faculty_citations"),
    (re.compile(r"most\s+cited\s+(?:professor|faculty|researcher|scientist)s?", re.I), "top_faculty_citations"),
    (re.compile(r"(?:rank(?:ed|ing)?)\s+(?:by|on)\s+(?:h[.\-\s]?index|citations?)", re.I), "top_faculty_hindex"),
    # ── per-name metric lookups ──
    (re.compile(r"h[.\-\s]?index\s+(?:of|for)\s+(.+)", re.I), "get_h_index"),
    (re.compile(r"citation(?:s| count|count)?\s+(?:of|for)\s+(.+)", re.I), "get_citations"),
    # ── faculty in a specific department (must come before broad faculty-count patterns) ──
    (re.compile(r"(?:faculty|professors?)\s+(?:in|from|of)\s+(.+?)\s*(?:dept|department)?\.?\s*$", re.I), "get_faculty_by_dept"),
    # ── papers by a specific author ──
    (re.compile(r"papers?\s+by\s+(.+)", re.I), "get_papers_by_author"),
    # ── department listing — ANCHORED: departments must be the whole point of the query ──
    # ^ and $ ensure "departments" cannot be buried as context inside a longer analytical sentence.
    # Covers: "what/which/list/show/tell me departments [at IIT]", "IIT Delhi departments", etc.
    # Does NOT match: "plot ML across all the departments" — "departments" is not the leading subject.
    (re.compile(
        r"^\s*"
        r"(?:(?:what|which|list|show|tell|give)\s+)?"  # optional leading verb
        r"(?:me\s+|us\s+)?"                             # optional pronoun: "show ME", "tell ME"
        r"(?:are\s+)?"                                  # optional "are"
        r"(?:all\s+(?:the\s+)?)?"                       # optional "all [the]"
        r"(?:the\s+)?"                                  # optional "the"
        r"(?:iit(?:[\s\-]delhi(?:'s)?)?\s+)?"           # optional "IIT [Delhi]"
        r"(?:various\s+|different\s+|academic\s+)?"     # optional adjective
        r"departments?\b"                                # THE SUBJECT
        r"(?:\s+(?:at|in|of|does|do|are)\s*"           # optional trailing: "at IIT", "does IIT have"
        r"(?:iit(?:[\s\-]delhi)?)?"
        r"(?:\s+have)?)?"
        r"\s*\??\s*$",
        re.I,
    ), "list_departments"),
    # "departments at IIT" / "departments list" (departments leads the query)
    (re.compile(
        r"^\s*departments?\b"
        r"(?:\s+(?:at|in|of|does|do|are|list|available))?"
        r"(?:\s+iit(?:[\s\-]delhi)?)?"
        r"\s*\??\s*$",
        re.I,
    ), "list_departments"),
    # ── faculty / professor count — anchored to a direct count question ──
    (re.compile(
        r"^\s*(?:how\s+many|total(?:\s+number\s+of)?|number\s+of|count\s+of|strength\s+of)\s+"
        r"(?:the\s+)?(?:faculty|professors?|teachers?|staff)\b"
        r"(?:\s+(?:are\s+(?:there\s+)?)?(?:at|in)\s+iit(?:[\s\-]delhi)?)?"
        r"\s*\??\s*$",
        re.I,
    ), "get_total_faculty_count"),
    (re.compile(
        r"^\s*(?:faculty|professors?)\s+(?:count|number|total|strength|size)\b\s*\??\s*$",
        re.I,
    ), "get_total_faculty_count"),
]


def match_structured(message: str) -> RouteMatch | None:
    msg = message.strip()
    for pattern, handler in PATTERNS:
        m = pattern.search(msg)
        if m:
            capture = m.group(1).strip() if m.lastindex else ""
            return RouteMatch(handler=handler, capture=capture)
    return None


async def execute_structured(
    route: RouteMatch,
    faculty_repo: FacultyRepository,
    research_repo: ResearchRepository,
) -> dict[str, Any]:
    handlers = {
        "get_h_index": _get_h_index,
        "get_citations": _get_citations,
        "get_faculty_by_dept": _get_faculty_by_dept,
        "get_papers_by_author": _get_papers_by_author,
        "list_departments": _list_departments,
        "get_total_faculty_count": _get_total_faculty_count,
        "top_faculty_hindex": _top_faculty_hindex,
        "top_faculty_citations": _top_faculty_citations,
    }
    fn = handlers.get(route.handler)
    if fn is None:
        return {"error": f"Unknown structured handler: {route.handler}"}
    return await fn(route.capture, faculty_repo, research_repo)


async def _get_h_index(name: str, faculty_repo: FacultyRepository, _: ResearchRepository) -> dict[str, Any]:
    from agent.guardrails.guardrails import name_tokens, faculty_name_matches

    tokens = name_tokens(name)
    if not tokens:
        return {"error": f'Could not identify a faculty name in "{name}".'}
    docs = await faculty_repo.text_search(" ".join(tokens), limit=3)
    for doc in docs:
        if faculty_name_matches(name, doc.get("firstName", ""), doc.get("lastName", "")):
            full = f"{doc.get('title', '')} {doc.get('firstName', '')} {doc.get('lastName', '')}".strip()
            return {"text": f"**{full}** has an h-index of **{doc.get('h_index', 'N/A')}**."}
    return {"error": f'No faculty named "{name}" found.'}


async def _get_citations(name: str, faculty_repo: FacultyRepository, _: ResearchRepository) -> dict[str, Any]:
    from agent.guardrails.guardrails import name_tokens, faculty_name_matches

    tokens = name_tokens(name)
    if not tokens:
        return {"error": f'Could not identify a faculty name in "{name}".'}
    docs = await faculty_repo.text_search(" ".join(tokens), limit=3)
    for doc in docs:
        if faculty_name_matches(name, doc.get("firstName", ""), doc.get("lastName", "")):
            full = f"{doc.get('title', '')} {doc.get('firstName', '')} {doc.get('lastName', '')}".strip()
            return {"text": f"**{full}** has **{doc.get('citation_count', 'N/A')}** total citations."}
    return {"error": f'No faculty named "{name}" found.'}


async def _get_faculty_by_dept(dept_name: str, faculty_repo: FacultyRepository, _: ResearchRepository) -> dict[str, Any]:
    dept = await faculty_repo.find_department(dept_name)
    if not dept:
        return {"error": f'No department matching "{dept_name}" was found.'}
    top_faculty = await faculty_repo.find_top_faculty_global(
        sort_by="h_index", limit=15, department_name=dept.get("name", dept_name)
    )
    if not top_faculty:
        return {"error": f'No faculty found in "{dept.get("name", dept_name)}".'}
    lines = [f'**{dept.get("name")}** — top faculty by H-index:\n']
    for i, f in enumerate(top_faculty, 1):
        dept_info = f.get("department") or {}
        name = f"{f.get('title', '')} {f.get('firstName', '')} {f.get('lastName', '')}".strip()
        email = f.get("email", "N/A")
        h = f.get("h_index", "N/A")
        lines.append(f"{i}. **{name}** — {email} (H-index: {h})")
    return {"text": "\n".join(lines)}


async def _get_papers_by_author(name: str, faculty_repo: FacultyRepository, research_repo: ResearchRepository) -> dict[str, Any]:
    from agent.guardrails.guardrails import name_tokens, faculty_name_matches

    tokens = name_tokens(name)
    if not tokens:
        return {"error": f'Could not identify a faculty name in "{name}".'}
    docs = await faculty_repo.text_search(" ".join(tokens), limit=3)
    for doc in docs:
        if faculty_name_matches(name, doc.get("firstName", ""), doc.get("lastName", "")):
            kerberos = (doc.get("email") or "").split("@")[0].lower()
            scopus_ids = [str(s) for s in (doc.get("scopus_id") or [])]
            or_clauses: list[dict] = []
            if kerberos:
                or_clauses.append({"kerberos": kerberos})
            if scopus_ids:
                or_clauses.append({"authors.author_id": {"$in": scopus_ids}})
            if not or_clauses:
                continue
            total = await research_repo.count_documents({"$or": or_clauses})
            full = f"{doc.get('title', '')} {doc.get('firstName', '')} {doc.get('lastName', '')}".strip()
            return {"text": f"**{full}** has **{total}** indexed publications."}
    return {"error": f'No faculty named "{name}" found.'}


async def _list_departments(_: str, faculty_repo: FacultyRepository, __: ResearchRepository) -> dict[str, Any]:
    all_depts = await faculty_repo.list_all_departments()
    grouped: dict[str, list[str]] = {}
    for dept in all_depts:
        cat = dept.get("category", "Other")
        grouped.setdefault(cat, []).append(dept.get("name", ""))
    lines = [f"IIT Delhi has **{len(all_depts)}** academic units:\n"]
    for cat, names in sorted(grouped.items()):
        lines.append(f"\n**{cat}s** ({len(names)}):")
        for n in names[:10]:
            lines.append(f"  - {n}")
        if len(names) > 10:
            lines.append(f"  - _(and {len(names) - 10} more)_")
    return {"text": "\n".join(lines)}


async def _get_total_faculty_count(_: str, faculty_repo: FacultyRepository, __: ResearchRepository) -> dict[str, Any]:
    total = await faculty_repo.count_all_faculty()
    return {"text": f"IIT Delhi has **{total}** faculty members across all departments."}


async def _top_faculty_hindex(_: str, faculty_repo: FacultyRepository, __: ResearchRepository) -> dict[str, Any]:
    docs = await faculty_repo.find_top_faculty_global(sort_by="h_index", limit=10)
    if not docs:
        return {"error": "No faculty data available."}
    lines = ["**Top 10 IIT Delhi Professors by H-Index:**\n"]
    for i, d in enumerate(docs, 1):
        name = f"{d.get('title', '')} {d.get('firstName', '')} {d.get('lastName', '')}".strip()
        dept_info = d.get("department") or {}
        dept = dept_info.get("name", "") if isinstance(dept_info, dict) else ""
        email = d.get("email", "N/A")
        h = d.get("h_index", "N/A")
        lines.append(f"{i}. **{name}** (H-index: {h})  \n   {dept}  \n   📧 {email}")
    return {"text": "\n".join(lines)}


async def _top_faculty_citations(_: str, faculty_repo: FacultyRepository, __: ResearchRepository) -> dict[str, Any]:
    docs = await faculty_repo.find_top_faculty_global(sort_by="citation_count", limit=10)
    if not docs:
        return {"error": "No faculty data available."}
    lines = ["**Top 10 IIT Delhi Professors by Total Citations:**\n"]
    for i, d in enumerate(docs, 1):
        name = f"{d.get('title', '')} {d.get('firstName', '')} {d.get('lastName', '')}".strip()
        dept_info = d.get("department") or {}
        dept = dept_info.get("name", "") if isinstance(dept_info, dict) else ""
        email = d.get("email", "N/A")
        cites = d.get("citation_count")
        cites_str = f"{cites:,}" if isinstance(cites, (int, float)) else "N/A"
        lines.append(f"{i}. **{name}** ({cites_str} citations)  \n   {dept}  \n   📧 {email}")
    return {"text": "\n".join(lines)}
