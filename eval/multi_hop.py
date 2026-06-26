"""Multi-hop and edge-case query generation from corpus fixtures."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any


def _tokenize_title(title: str) -> list[str]:
    return [
        w for w in re.findall(r"[a-z0-9]{4,}", (title or "").lower())
        if w not in {"using", "based", "study", "analysis", "from", "with"}
    ]


def _faculty_display(doc: dict) -> str:
    first = (doc.get("faculty_first_name") or "").strip()
    last = (doc.get("faculty_last_name") or "").strip()
    if first or last:
        return " ".join(p for p in (first, last) if p)
    raw = (doc.get("faculty_name") or doc.get("kerberos") or "").strip()
    return re.sub(r"^(Prof\.?|Dr\.?)\s+", "", raw, flags=re.IGNORECASE)


def generate_multi_hop_queries(
    corpus: dict[str, Any],
    *,
    seed: int = 42,
    count: int = 15,
) -> list[dict[str, Any]]:
    """Create comparative / multi-hop questions from corpus documents.

    Uses title/abstract tokens and faculty kerberos — not field_associated or author surnames.
    """
    docs = corpus.get("documents", [])
    rng = random.Random(seed)
    queries: list[dict[str, Any]] = []

    # Topic bigram clusters for cross-paper comparisons (title + abstract)
    bigram_docs: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        words = _tokenize_title(d.get("title", "")) + _tokenize_title(d.get("abstract", ""))
        seen: set[str] = set()
        for i in range(len(words) - 1):
            bg = f"{words[i]} {words[i + 1]}"
            if bg not in seen:
                seen.add(bg)
                bigram_docs[bg].append(d)

    for bigram, cluster in sorted(bigram_docs.items(), key=lambda x: -len(x[1])):
        if len(cluster) < 2:
            continue
        pair = rng.sample(cluster, 2)
        queries.append({
            "id": f"multihop-topic-{len(queries)+1}",
            "type": "multi_hop_compare_topic",
            "query": (
                f"Compare research on {bigram} between "
                f"\"{pair[0]['title'][:60]}\" and \"{pair[1]['title'][:60]}\""
            ),
            "relevant": {pair[0]["mongo_id"]: 3, pair[1]["mongo_id"]: 3},
            "difficulty": "high",
        })
        if len(queries) >= count // 3:
            break

    # Faculty kerberos + topic multi-hop
    kerberos_docs = [d for d in docs if d.get("kerberos")]
    by_kerberos: dict[str, list[dict]] = defaultdict(list)
    for d in kerberos_docs:
        by_kerberos[d["kerberos"]].append(d)

    for kerberos, papers in rng.sample(
        list(by_kerberos.items()), min(5, len(by_kerberos))
    ):
        d = papers[0]
        kw = _tokenize_title(d["title"])[:2]
        if not kw:
            continue
        first = d.get("faculty_first_name") or ""
        last = d.get("faculty_last_name") or ""
        name = _faculty_display(d)
        dept = d.get("faculty_department")
        matching = [
            p for p in papers
            if any(
                w in (p.get("title") or "").lower() or w in (p.get("abstract") or "").lower()
                for w in kw
            )
        ]
        relevant = {p["mongo_id"]: 3 for p in matching[:3]} or {d["mongo_id"]: 3}
        queries.append({
            "id": f"multihop-faculty-{len(queries)+1}",
            "type": "multi_hop_faculty_topic",
            "query": (
                f"What has {name} from {dept} published about {' '.join(kw)}?"
                if dept else f"What has {name} published about {' '.join(kw)}?"
            ),
            "kerberos": kerberos,
            "relevant": relevant,
            "difficulty": "high",
        })

    # Temporal multi-hop (year + title keywords)
    recent = sorted(docs, key=lambda x: x.get("publication_year") or 0, reverse=True)[:30]
    for d in rng.sample(recent, min(5, len(recent))):
        year = d.get("publication_year")
        if not year:
            continue
        kw = _tokenize_title(d["title"])[:2]
        if not kw:
            continue
        queries.append({
            "id": f"multihop-year-{len(queries)+1}",
            "type": "multi_hop_temporal",
            "query": f"Summarize {' '.join(kw)} research from {year} related to {d['title'][:50]}",
            "relevant": {d["mongo_id"]: 3},
            "difficulty": "high",
        })

    return queries[:count]


def generate_edge_case_queries(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    """Ambiguous, empty-ish, and adversarial retrieval queries."""
    docs = corpus.get("documents", [])
    common_word = "analysis"
    ambiguous = [d for d in docs if common_word in (d.get("title") or "").lower()]

    return [
        {
            "id": "edge-empty",
            "type": "edge_empty",
            "query": "   ",
            "relevant": {},
            "expect_error": True,
        },
        {
            "id": "edge-ambiguous-common",
            "type": "edge_ambiguous",
            "query": common_word,
            "relevant": {d["mongo_id"]: 1 for d in ambiguous[:10]},
            "difficulty": "high",
            "notes": f"Ambiguous single token; {len(ambiguous)} corpus matches",
        },
        {
            "id": "edge-typo",
            "type": "edge_typo",
            "query": "machne lerning neurual netwroks",
            "relevant": {},
            "difficulty": "medium",
            "notes": "Heavy typos — tests fuzziness",
        },
        {
            "id": "edge-long",
            "type": "edge_long",
            "query": " ".join(["machine learning"] * 80),
            "relevant": {},
            "difficulty": "medium",
            "expect_truncation": True,
        },
        {
            "id": "edge-nonexistent",
            "type": "edge_nonexistent",
            "query": "xyzzyxnonexistentquantumfoam12345",
            "relevant": {},
            "difficulty": "low",
            "expect_empty": True,
        },
        {
            "id": "edge-mixed-lang",
            "type": "edge_mixed",
            "query": "solar cell प्रदर्शन photovoltaic efficiency",
            "relevant": {},
            "difficulty": "medium",
        },
    ]
