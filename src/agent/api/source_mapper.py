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


def ips_to_sources(ips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by application number/title and shape IP filings for the sources event."""
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for ip in ips:
        key = ip.get("application_number") or ip.get("title", "")
        if not key or key in seen:
            continue
        seen.add(key)
        inventors = ip.get("inventors", [])
        authors = [inv.get("name") if isinstance(inv, dict) else inv for inv in inventors]
        deduped.append({
            "index": ip.get("citation_index") if "citation_index" in ip else ip.get("index"),
            "id": ip.get("id", ""),
            "title": ip.get("title", ""),
            "authors": [a for a in authors if a],
            "application_number": ip.get("application_number"),
            "document_type": ip.get("type_of_ip"),
            "publication_year": ip.get("publication_year"),
            "field_associated": ip.get("field_of_invention") or ip.get("department"),
        })
    return deduped
