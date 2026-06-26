#!/usr/bin/env python3
"""
Stratified sample of 500 indexed papers from MongoDB for v2 eval fixtures.

Criteria:
  - open_search_id present and not pending_*
  - kerberos and abstract present
  - diverse publication years (stratified proportional allocation)

Output: eval/fixtures/corpus_v2.json

Usage:
  cd chatbot-agent
  .venv/bin/python eval/scripts/sample_corpus_v2.py
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from pymongo import MongoClient

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

DEFAULT_SAMPLE_SIZE = 500
DEFAULT_COLLECTION = "researchmetadatascopus"
OUTPUT_PATH = _ROOT / "eval" / "fixtures" / "corpus_v2.json"

INDEXED_MATCH: dict[str, Any] = {
    "open_search_id": {"$exists": True, "$ne": None, "$ne": ""},
    "$expr": {
        "$not": {"$regexMatch": {"input": "$open_search_id", "regex": "^pending_"}}
    },
    "kerberos": {"$exists": True, "$nin": [None, ""]},
    "abstract": {"$exists": True, "$nin": [None, "", "(No abstract available)"]},
}

PROJECT_FIELDS = {
    "_id": 1,
    "title": 1,
    "abstract": 1,
    "authors": 1,
    "publication_year": 1,
    "document_type": 1,
    "field_associated": 1,
    "subject_area": 1,
    "citation_count": 1,
    "reference_count": 1,
    "document_eid": 1,
    "document_scopus_id": 1,
    "link": 1,
    "kerberos": 1,
    "open_search_id": 1,
}


def _db_name_from_uri(uri: str) -> str:
    path = urlparse(uri).path.lstrip("/")
    return path.split("/")[0] if path else "research_ambit"


def _load_faculty_map(db) -> dict[str, dict[str, str]]:
    """Map kerberos -> {firstName, lastName, name, department, expertise}."""
    faculties = db["faculties"]
    departments = db["departments"]
    dept_ids = set()
    faculty_docs = list(
        faculties.find(
            {},
            {
                "email": 1,
                "firstName": 1,
                "lastName": 1,
                "title": 1,
                "department": 1,
                "expertise": 1,
                "brief_expertise": 1,
            },
        )
    )
    for doc in faculty_docs:
        dept = doc.get("department")
        if dept is not None:
            dept_ids.add(dept)

    dept_map: dict[Any, str] = {}
    if dept_ids:
        for d in departments.find({"_id": {"$in": list(dept_ids)}}, {"name": 1}):
            dept_map[d["_id"]] = d.get("name", "")

    result: dict[str, dict[str, str]] = {}
    for doc in faculty_docs:
        email = doc.get("email") or ""
        kerberos = email.split("@")[0].lower().strip()
        if not kerberos:
            continue
        dept_id = doc.get("department")
        dept_name = dept_map.get(dept_id, "") if dept_id is not None else ""
        parts = [doc.get("title", ""), doc.get("firstName", ""), doc.get("lastName", "")]
        name = " ".join(p for p in parts if p).strip()
        expertise = doc.get("expertise") or doc.get("brief_expertise") or []
        if isinstance(expertise, list):
            expertise_str = "; ".join(str(e) for e in expertise[:5])
        else:
            expertise_str = str(expertise)
        result[kerberos] = {
            "firstName": doc.get("firstName", ""),
            "lastName": doc.get("lastName", ""),
            "name": name,
            "department": dept_name,
            "expertise": expertise_str,
        }
    return result


def _allocate_per_year(year_counts: list[tuple[int | None, int]], total: int) -> dict[int, int]:
    """Proportional allocation with at least one doc per year bucket when possible."""
    valid = [(y, c) for y, c in year_counts if y is not None and c > 0]
    if not valid:
        return {}

    pool_total = sum(c for _, c in valid)
    alloc: dict[int, int] = {}
    for year, count in valid:
        share = max(1, round(total * count / pool_total))
        alloc[year] = min(share, count)

    # Trim or pad to exact total
    current = sum(alloc.values())
    years_sorted = sorted(alloc.keys())
    rng = random.Random(42)

    while current > total:
        year = rng.choice(years_sorted)
        if alloc[year] > 1:
            alloc[year] -= 1
            current -= 1

    while current < total:
        best = max(valid, key=lambda yc: yc[1] - alloc.get(yc[0], 0))
        if alloc[best[0]] < best[1]:
            alloc[best[0]] += 1
            current += 1
        else:
            break

    return alloc


def stratified_sample(
    collection,
    match: dict[str, Any],
    sample_size: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pipeline = [
        {"$match": match},
        {"$group": {"_id": "$publication_year", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    year_counts = [(row["_id"], row["count"]) for row in collection.aggregate(pipeline)]
    allocations = _allocate_per_year(year_counts, sample_size)

    docs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for year, count in sorted(allocations.items()):
        year_match = {**match, "publication_year": year}
        cursor = collection.aggregate(
            [
                {"$match": year_match},
                {"$sample": {"size": count}},
                {"$project": PROJECT_FIELDS},
            ]
        )
        for doc in cursor:
            mid = str(doc["_id"])
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            docs.append(doc)

    # Top up if stratification under-filled (e.g. duplicate _id collisions)
    if len(docs) < sample_size:
        need = sample_size - len(docs)
        extra = list(
            collection.aggregate(
                [
                    {"$match": {**match, "_id": {"$nin": [d["_id"] for d in docs]}}},
                    {"$sample": {"size": need}},
                    {"$project": PROJECT_FIELDS},
                ]
            )
        )
        docs.extend(extra)

    rng.shuffle(docs)
    return docs[:sample_size]


def _normalize_authors(authors: list | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for a in authors or []:
        if isinstance(a, dict):
            name = a.get("author_name") or a.get("name") or ""
            author_id = a.get("author_id") or a.get("id") or ""
        else:
            name, author_id = str(a), ""
        if name:
            out.append({"name": name, "id": str(author_id)})
    return out


def _keywords_from_doc(doc: dict[str, Any]) -> list[str]:
    if doc.get("keywords"):
        return list(doc["keywords"])[:20]
    areas = doc.get("subject_area") or []
    if isinstance(areas, list):
        return [str(a) for a in areas[:10]]
    return []


def build_corpus_payload(
    docs: list[dict[str, Any]],
    total_indexed: int,
    faculty_map: dict[str, dict[str, str]],
    sample_size: int,
) -> dict[str, Any]:
    documents = []
    for d in docs:
        kerberos = (d.get("kerberos") or "").lower().strip()
        fac = faculty_map.get(kerberos, {})
        documents.append(
            {
                "mongo_id": str(d["_id"]),
                "title": d.get("title"),
                "abstract": d.get("abstract"),
                "authors": _normalize_authors(d.get("authors")),
                "publication_year": d.get("publication_year"),
                "document_type": d.get("document_type"),
                "field_associated": d.get("field_associated"),
                "subject_area": d.get("subject_area"),
                "citation_count": d.get("citation_count") or 0,
                "reference_count": d.get("reference_count") or 0,
                "kerberos": kerberos,
                "keywords": _keywords_from_doc(d),
                "open_search_id": d.get("open_search_id"),
                "faculty_name": fac.get("name"),
                "faculty_department": fac.get("department"),
                "faculty_expertise": fac.get("expertise"),
            }
        )

    years = [d["publication_year"] for d in documents if d.get("publication_year")]
    with_kerberos = sum(1 for d in documents if d.get("kerberos"))
    with_abstract = sum(
        1 for d in documents if d.get("abstract") and d["abstract"] != "(No abstract available)"
    )

    return {
        "version": 2,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(documents),
        "target_sample_size": sample_size,
        "total_indexed_in_db": total_indexed,
        "total_documents": len(documents),
        "sampling": "stratified_by_publication_year",
        "filters": {
            "indexed": True,
            "exclude_pending_open_search_id": True,
            "require_kerberos": True,
            "require_abstract": True,
        },
        "fields_summary": {
            "with_abstract": with_abstract,
            "with_kerberos": with_kerberos,
            "with_faculty_match": sum(1 for d in documents if d.get("faculty_name")),
            "document_types": sorted({d["document_type"] for d in documents if d.get("document_type")}),
            "year_range": {"min": min(years) if years else None, "max": max(years) if years else None},
            "unique_years": len(set(years)),
            "unique_kerberos": len({d["kerberos"] for d in documents if d.get("kerberos")}),
            "subject_areas": sorted(
                {a for d in documents for a in (d.get("subject_area") or []) if a}
            ),
        },
        "documents": documents,
    }


def main() -> None:
    load_dotenv(_ROOT / ".env")
    uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/research_ambit")
    collection_name = os.environ.get("MONGODB_COLLECTION", DEFAULT_COLLECTION)
    sample_size = int(os.environ.get("CORPUS_LIMIT", str(DEFAULT_SAMPLE_SIZE)))

    print("=== Corpus v2 stratified sample ===\n")
    print(f"  URI db: {_db_name_from_uri(uri)}")
    print(f"  Collection: {collection_name}")
    print(f"  Target size: {sample_size}")

    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    db = client[_db_name_from_uri(uri)]
    collection = db[collection_name]

    total_indexed = collection.count_documents(INDEXED_MATCH)
    print(f"  Eligible indexed docs: {total_indexed}")

    if total_indexed == 0:
        print("\nNo eligible documents. Ensure indexer has run.")
        sys.exit(1)

    faculty_map = _load_faculty_map(db)
    print(f"  Faculty kerberos map: {len(faculty_map)} entries")

    docs = stratified_sample(collection, INDEXED_MATCH, min(sample_size, total_indexed))
    print(f"  Sampled: {len(docs)} documents")

    corpus = build_corpus_payload(docs, total_indexed, faculty_map, sample_size)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(corpus, indent=2), encoding="utf-8")

    fs = corpus["fields_summary"]
    print(f"\nWritten: {OUTPUT_PATH}")
    print(f"  Years: {fs['year_range']['min']}–{fs['year_range']['max']} ({fs['unique_years']} buckets)")
    print(f"  With abstract: {fs['with_abstract']}")
    print(f"  With faculty match: {fs['with_faculty_match']}")
    print(f"  Unique kerberos: {fs['unique_kerberos']}")
    client.close()


if __name__ == "__main__":
    main()
