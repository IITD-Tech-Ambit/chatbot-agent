"""Map tool paper payloads to the SSE sources contract."""

from __future__ import annotations

from typing import Any


def papers_to_sources(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by title and shape papers for the frontend sources event."""
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for p in papers:
        title = p.get("title", "")
        if not title or title in seen:
            continue
        seen.add(title)
        deduped.append({
            "index": p.get("citation_index") if "citation_index" in p else p.get("index"),
            "id": p.get("id", ""),
            "title": title,
            "authors": p.get("authors", []),
            "publication_year": p.get("publication_year", p.get("year")),
            "document_type": p.get("document_type"),
            "field_associated": p.get("field_associated", p.get("field")),
            "citation_count": p.get("citation_count", p.get("citations", 0)),
            "link": p.get("link"),
            "document_scopus_id": p.get("document_scopus_id"),
            "document_eid": p.get("document_eid"),
            "kerberos": p.get("kerberos"),
            "faculty_name": p.get("faculty_name"),
        })
    return deduped
