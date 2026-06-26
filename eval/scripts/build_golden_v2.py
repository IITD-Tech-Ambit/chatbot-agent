#!/usr/bin/env python3
"""
Build v2 golden sets from corpus_v2.json.

Retrieval-relevant fields: title, abstract, keywords only (no author_names, no field_associated).

Outputs:
  eval/fixtures/golden_comprehensive_v2.json  (~80-120 queries)
  eval/fixtures/golden_hard_v2.json            (~30-50 queries)

Usage:
  cd chatbot-agent
  .venv/bin/python eval/scripts/build_golden_v2.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_PATH = _ROOT / "eval" / "fixtures" / "corpus_v2.json"
COMPREHENSIVE_PATH = _ROOT / "eval" / "fixtures" / "golden_comprehensive_v2.json"
HARD_PATH = _ROOT / "eval" / "fixtures" / "golden_hard_v2.json"

STOPWORDS = {
    "a", "an", "the", "of", "in", "to", "for", "and", "or", "is", "are", "was", "were",
    "on", "at", "by", "with", "from", "as", "it", "its", "this", "that", "be", "been",
    "has", "have", "had", "not", "but", "can", "will", "do", "does", "did", "which",
    "using", "based", "study", "paper", "research", "results", "analysis", "approach",
    "method", "proposed", "present", "show", "new", "used", "two", "one", "high", "low",
    "different", "effect", "effects", "however", "investigated", "examined", "various",
}

COMMON_TITLE_WORDS = {
    "energy", "synthesis", "properties", "india", "carbon", "metal", "power", "control",
    "optimization", "simulation", "detection", "analysis", "treatment", "water", "thermal",
    "electrical", "magnetic", "optical", "composite", "structure", "process", "processing",
    "algorithm", "algorithms", "wireless", "sensor", "sensors", "network", "networks",
    "image", "signal", "frequency", "antenna", "film", "films", "thin", "layer", "phase",
    "temperature", "stress", "strain", "impact", "damage", "failure", "microstructure",
    "nanoparticle", "nanoparticles", "graphene", "quantum", "laser", "plasma", "catalyst",
    "polymer", "finite", "element", "machine", "learning", "deep", "neural", "data", "hybrid",
    "nonlinear", "solar", "from", "using", "based",
}


def pick_deterministic(arr: list[dict], n: int, stride: int = 7) -> list[dict]:
    if not arr:
        return []
    sorted_arr = sorted(arr, key=lambda d: d["mongo_id"])
    out: list[dict] = []
    i = 0
    while len(out) < n and i < len(sorted_arr) * 2:
        doc = sorted_arr[i % len(sorted_arr)]
        if doc not in out:
            out.append(doc)
        i += stride
    return out[:n]


def extract_keywords(text: str, count: int = 4) -> list[str]:
    if not text:
        return []
    words = re.sub(r"[^a-z0-9\s-]", " ", text.lower()).split()
    filtered = [w for w in words if len(w) > 3 and w not in STOPWORDS and not w.isdigit()]
    seen: set[str] = set()
    out: list[str] = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= count:
            break
    return out


def title_snippet(title: str, max_len: int = 60) -> str:
    if not title:
        return ""
    if len(title) <= max_len:
        return title
    cut = title[:max_len]
    return re.sub(r"\s\S*$", "", cut).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{3,}", (text or "").lower())


def abstract_only_terms(doc: dict, min_len: int = 7, count: int = 4) -> list[str]:
    title_set = set(tokenize(doc.get("title", "")))
    terms = [
        w
        for w in tokenize(doc.get("abstract", ""))
        if w not in title_set and len(w) >= min_len and w not in COMMON_TITLE_WORDS
    ]
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= count:
            break
    return out


def common_word_ratio(title: str) -> float:
    words = extract_keywords(title, 12)
    if not words:
        return 1.0
    return sum(1 for w in words if w in COMMON_TITLE_WORDS) / len(words)


def discover_topic_clusters(docs: list[dict], min_docs: int = 3, max_clusters: int = 8) -> list[dict]:
    bigram_counts: dict[str, int] = defaultdict(int)
    for d in docs:
        words = extract_keywords(d.get("title", ""), 12)
        for i in range(len(words) - 1):
            bigram_counts[f"{words[i]} {words[i + 1]}"] += 1
    clusters = []
    for query, count in sorted(bigram_counts.items(), key=lambda x: (-x[1], x[0])):
        if count < min_docs:
            continue
        w1, w2 = query.split(" ", 1)
        matching = [
            d for d in docs
            if w1 in (d.get("title") or "").lower() and w2 in (d.get("title") or "").lower()
        ]
        if len(matching) >= min_docs:
            clusters.append({"query": query, "matching": matching, "count": len(matching)})
        if len(clusters) >= max_clusters:
            break
    return clusters


def build_topic_cluster(docs: list[dict], query_words: list[str], min_docs: int = 3, max_judged: int = 8) -> dict | None:
    terms = [w.lower() for w in query_words]
    matching = [
        d for d in docs
        if all(w in (d.get("title") or "").lower() for w in terms)
    ]
    if len(matching) < min_docs:
        return None
    ranked = sorted(
        matching,
        key=lambda d: (
            sum(1 for w in terms if w in (d.get("title") or "").lower()),
            d.get("citation_count") or 0,
        ),
        reverse=True,
    )
    relevant: dict[str, int] = {}
    for i, d in enumerate(ranked[:max_judged]):
        relevant[d["mongo_id"]] = 3 if i == 0 else (2 if i < 3 else 1)
    return {"query": " ".join(query_words), "matching_docs": len(matching), "anchor": ranked[0], "relevant": relevant}


PARAPHRASE_TEMPLATES: list[Callable[[dict], str | None]] = [
    lambda d: (
        f"innovations related to {' and '.join(extract_keywords(d['title'], 3))}"
        if len(extract_keywords(d["title"], 3)) >= 2
        else None
    ),
    lambda d: (
        f"how does {extract_keywords(d['title'], 2)[0]} relate to {extract_keywords(d['title'], 2)[1]}"
        if len(extract_keywords(d["title"], 2)) >= 2
        else None
    ),
    lambda d: (
        f"recent advances in {' '.join(extract_keywords(d['title'], 2))} techniques"
        if len(extract_keywords(d["title"], 2)) >= 2
        else None
    ),
]

HARD_PARAPHRASES: list[Callable[[dict], str | None]] = [
    lambda d: (
        f"research investigating {' and '.join(extract_keywords(d['title'], 3))} mechanisms"
        if len(extract_keywords(d["title"], 3)) >= 2
        else None
    ),
    lambda d: (
        f"what approaches exist for {extract_keywords(d['title'], 2)[0]} related {extract_keywords(d['title'], 2)[1]} problems"
        if len(extract_keywords(d["title"], 2)) >= 2
        else None
    ),
    lambda d: (
        f"work on {' '.join(extract_keywords(d['title'], 2))} in applied science"
        if len(extract_keywords(d["title"], 2)) >= 2
        else None
    ),
]

FACULTY_QUERY_TEMPLATES = [
    "papers by Dr {last}",
    "publications by Professor {last}",
    "research by Dr {first} {last}",
    "what has {first} {last} published",
    "papers authored by Dr {last} on {topic}",
]


def _kerberos_groups(docs: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        k = (d.get("kerberos") or "").lower().strip()
        if k:
            groups[k].append(d)
    return groups


def _next_id(prefix: str, counter: list[int]) -> str:
    counter[0] += 1
    return f"{prefix}-{counter[0]}"


def build_comprehensive_golden(corpus: dict[str, Any]) -> dict[str, Any]:
    docs = corpus.get("documents", [])
    docs_rich = [d for d in docs if d.get("abstract") and len(d["abstract"]) > 80 and len(d.get("title", "")) > 10]
    queries: list[dict[str, Any]] = []
    qid = [0]

    for doc in pick_deterministic(docs_rich, 20, 5):
        queries.append({
            "id": _next_id("exact-title", qid),
            "query": title_snippet(doc["title"]),
            "type": "exact_title",
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "relevant": {doc["mongo_id"]: 3},
        })

    for doc in pick_deterministic(docs_rich, 20, 6):
        kw = extract_keywords(doc["title"], 4)
        if len(kw) < 2:
            continue
        queries.append({
            "id": _next_id("partial-title", qid),
            "query": " ".join(kw),
            "type": "partial_title",
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "relevant": {doc["mongo_id"]: 3},
        })

    for doc in pick_deterministic(docs_rich, 15, 8):
        kw = extract_keywords(doc["abstract"], 4)
        if len(kw) < 3:
            continue
        queries.append({
            "id": _next_id("abstract-kw", qid),
            "query": " ".join(kw),
            "type": "abstract_keyword",
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "relevant": {doc["mongo_id"]: 2},
        })

    for i, doc in enumerate(pick_deterministic(docs_rich, 12, 13)):
        paraphrase = PARAPHRASE_TEMPLATES[i % len(PARAPHRASE_TEMPLATES)](doc)
        if not paraphrase:
            continue
        queries.append({
            "id": _next_id("semantic-paraphrase", qid),
            "query": paraphrase,
            "type": "semantic_paraphrase",
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "relevant": {doc["mongo_id"]: 2},
        })

    gap_docs = sorted(
        [(d, abstract_only_terms(d)) for d in docs_rich],
        key=lambda x: len(x[1]),
        reverse=True,
    )
    for doc in pick_deterministic([d for d, t in gap_docs if len(t) >= 3], 10, 7):
        terms = abstract_only_terms(doc)
        if len(terms) < 3:
            continue
        queries.append({
            "id": _next_id("abstract-gap", qid),
            "query": " ".join(terms[:3]),
            "type": "abstract_gap",
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "relevant": {doc["mongo_id"]: 3},
            "notes": "Terms in abstract but not title",
        })

    # faculty_kerberos — link via kerberos, not author surname search
    groups = _kerberos_groups(docs)
    faculty_candidates = sorted(
        [(k, p) for k, p in groups.items() if p and p[0].get("faculty_name")],
        key=lambda x: (-len(x[1]), x[0]),
    )
    for kerberos, papers in faculty_candidates[:12]:
        lead = papers[0]
        name_parts = (lead.get("faculty_name") or kerberos).split()
        first = name_parts[0] if name_parts else kerberos
        last = name_parts[-1] if name_parts else kerberos
        topic_kw = extract_keywords(lead.get("title", ""), 1)
        topic = topic_kw[0] if topic_kw else "research"
        template = FACULTY_QUERY_TEMPLATES[len(queries) % len(FACULTY_QUERY_TEMPLATES)]
        query = template.format(first=first, last=last, topic=topic)
        relevant = {p["mongo_id"]: 3 if i == 0 else 2 for i, p in enumerate(papers[:5])}
        queries.append({
            "id": _next_id("faculty-kerberos", qid),
            "query": query,
            "type": "faculty_kerberos",
            "kerberos": kerberos,
            "faculty_name": lead.get("faculty_name"),
            "source_mongo_id": lead["mongo_id"],
            "source_title": lead["title"],
            "relevant": relevant,
            "notes": "Faculty resolved via kerberos on indexed papers",
        })

    # multi_hop temporal
    by_year: dict[int, list[dict]] = defaultdict(list)
    for d in docs_rich:
        y = d.get("publication_year")
        if y:
            by_year[y].append(d)
    years_sorted = sorted(by_year.keys(), reverse=True)
    for year in pick_deterministic([{"mongo_id": str(y), "year": y, "docs": by_year[y]} for y in years_sorted], 6, 3):
        y = year["year"]
        doc = pick_deterministic(by_year[y], 1, 1)[0]
        kw = extract_keywords(doc["title"], 2)
        if not kw:
            continue
        year_docs = [d for d in by_year[y] if any(w in (d.get("title") or "").lower() for w in kw)]
        relevant = {d["mongo_id"]: 2 for d in year_docs[:5]}
        relevant[doc["mongo_id"]] = 3
        queries.append({
            "id": _next_id("multihop-temporal", qid),
            "query": f"What research on {' '.join(kw)} was published in {y}?",
            "type": "multi_hop_temporal",
            "publication_year": y,
            "source_mongo_id": doc["mongo_id"],
            "relevant": relevant,
            "difficulty": "high",
        })

    # multi_hop topic — compare two papers sharing title bigram
    for cluster in discover_topic_clusters(docs_rich, min_docs=2, max_clusters=6):
        pair = pick_deterministic(cluster["matching"], 2, 1)
        if len(pair) < 2:
            continue
        queries.append({
            "id": _next_id("multihop-topic", qid),
            "query": (
                f"Compare findings on {cluster['query']} between "
                f"\"{title_snippet(pair[0]['title'], 50)}\" and \"{title_snippet(pair[1]['title'], 50)}\""
            ),
            "type": "multi_hop_topic",
            "relevant": {pair[0]["mongo_id"]: 3, pair[1]["mongo_id"]: 3},
            "difficulty": "high",
        })

    # multi_hop faculty + topic
    for kerberos, papers in faculty_candidates[:6]:
        doc = papers[0]
        kw = extract_keywords(doc.get("title", ""), 2)
        if not kw:
            continue
        name = doc.get("faculty_name") or kerberos
        matching = [p for p in papers if any(w in (p.get("title") or "").lower() for w in kw)]
        relevant = {p["mongo_id"]: 3 for p in matching[:3]} or {doc["mongo_id"]: 3}
        queries.append({
            "id": _next_id("multihop-faculty", qid),
            "query": f"What has {name} published about {' '.join(kw)}?",
            "type": "multi_hop_faculty",
            "kerberos": kerberos,
            "relevant": relevant,
            "difficulty": "high",
        })

    for cluster in discover_topic_clusters(docs_rich):
        matching = cluster["matching"]
        relevant = {}
        lead = cluster["query"].split()[0]
        for d in pick_deterministic(matching, min(8, len(matching)), 2):
            relevant[d["mongo_id"]] = 3 if lead in (d.get("title") or "").lower() else 2
        queries.append({
            "id": _next_id("topic-cluster", qid),
            "query": cluster["query"],
            "type": "multi_relevant",
            "matching_docs": cluster["count"],
            "relevant": relevant,
        })

    return {
        "version": 2,
        "description": "Comprehensive v2 golden set — title/abstract/keywords retrieval; faculty via kerberos",
        "corpus_documents": corpus.get("total_documents", len(docs)),
        "corpus_sample_size": corpus.get("sample_size", len(docs)),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_fields": ["title", "abstract", "keywords"],
        "categories": {
            "exact_title": "Title substring — source doc in top 10",
            "partial_title": "Title keywords — source doc in top 50",
            "abstract_keyword": "Distinctive abstract terms",
            "semantic_paraphrase": "Paraphrased title — vector recall",
            "abstract_gap": "Abstract-only terms absent from title",
            "faculty_kerberos": "Faculty name query; relevance via paper.kerberos",
            "multi_hop_temporal": "Year + topic combined query",
            "multi_hop_topic": "Cross-paper topic comparison",
            "multi_hop_faculty": "Faculty + topic multi-hop",
            "multi_relevant": "Title bigram cluster (≥3 docs)",
        },
        "queries": queries,
    }


def build_hard_golden(corpus: dict[str, Any]) -> dict[str, Any]:
    docs = [d for d in corpus.get("documents", []) if len(d.get("title", "")) > 10]
    docs_with_abstract = [d for d in docs if d.get("abstract") and len(d["abstract"]) > 120]
    queries: list[dict[str, Any]] = []
    qid = [0]

    for doc in pick_deterministic(
        [d for d in docs if len(d.get("title", "")) >= 90 and len(extract_keywords(d["title"], 5)) >= 4],
        8,
        5,
    ):
        queries.append({
            "id": _next_id("hard-exact", qid),
            "query": title_snippet(doc["title"], 85),
            "type": "hard_exact_rank1",
            "difficulty": "high",
            "metric_focus": ["mrr", "precision_1", "ndcg_10"],
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "relevant": {doc["mongo_id"]: 3},
            "notes": "Long title truncated; source must rank first",
        })

    hard_partial = sorted(
        [d for d in docs if common_word_ratio(d["title"]) >= 0.55 and len(extract_keywords(d["title"], 5)) >= 3],
        key=lambda d: common_word_ratio(d["title"]),
        reverse=True,
    )
    for doc in pick_deterministic(hard_partial, 10, 4):
        kw = extract_keywords(doc["title"], 4)
        queries.append({
            "id": _next_id("hard-partial", qid),
            "query": " ".join(kw),
            "type": "hard_partial_common",
            "difficulty": "high",
            "metric_focus": ["precision_10", "mrr", "recall_50"],
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "relevant": {doc["mongo_id"]: 3},
            "notes": "Corpus-common title terms; many distractors",
        })

    manual_clusters = [
        ["induction", "motor"],
        ["thin", "films"],
        ["power", "quality"],
        ["finite", "element"],
        ["neural", "network"],
    ]
    for terms in manual_clusters:
        cluster = build_topic_cluster(docs, terms, min_docs=2, max_judged=6)
        if not cluster:
            continue
        queries.append({
            "id": _next_id("hard-cluster", qid),
            "query": cluster["query"],
            "type": "hard_graded_cluster",
            "difficulty": "high",
            "metric_focus": ["ndcg_10", "recall_50", "precision_10"],
            "matching_docs": cluster["matching_docs"],
            "source_mongo_id": cluster["anchor"]["mongo_id"],
            "source_title": cluster["anchor"]["title"],
            "relevant": cluster["relevant"],
            "notes": "Graded 3/2/1 cluster",
        })

    for cluster in discover_topic_clusters(docs, min_docs=3, max_clusters=4):
        ranked = pick_deterministic(cluster["matching"], min(6, len(cluster["matching"])), 2)
        relevant = {d["mongo_id"]: 3 if i == 0 else (2 if i < 3 else 1) for i, d in enumerate(ranked)}
        queries.append({
            "id": _next_id("hard-cluster", qid),
            "query": cluster["query"],
            "type": "hard_graded_cluster",
            "difficulty": "medium",
            "matching_docs": cluster["count"],
            "source_mongo_id": ranked[0]["mongo_id"],
            "relevant": relevant,
            "notes": "Auto-mined title bigram cluster",
        })

    for i, doc in enumerate(pick_deterministic(docs_with_abstract, 10, 8)):
        paraphrase = HARD_PARAPHRASES[i % len(HARD_PARAPHRASES)](doc)
        if not paraphrase:
            continue
        title_tokens = set(tokenize(doc["title"]))
        para_tokens = tokenize(paraphrase)
        overlap = sum(1 for t in para_tokens if t in title_tokens) / (len(para_tokens) or 1)
        if overlap > 0.6:
            continue
        queries.append({
            "id": _next_id("hard-paraphrase", qid),
            "query": paraphrase,
            "type": "hard_paraphrase",
            "difficulty": "high",
            "metric_focus": ["mrr", "recall_50", "ndcg_10"],
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "lexical_overlap": round(overlap, 2),
            "relevant": {doc["mongo_id"]: 2},
            "notes": "NL paraphrase with low title overlap",
        })

    abstract_gap_docs = sorted(
        [(d, abstract_only_terms(d, min_len=8, count=5)) for d in docs_with_abstract],
        key=lambda x: len(x[1]),
        reverse=True,
    )
    for doc in pick_deterministic([d for d, t in abstract_gap_docs if len(t) >= 4], 12, 6):
        terms = abstract_only_terms(doc, min_len=8, count=5)
        if len(terms) < 4:
            continue
        queries.append({
            "id": _next_id("very-hard-abstract", qid),
            "query": " ".join(terms[:4]),
            "type": "very_hard_abstract_only",
            "difficulty": "very_high",
            "metric_focus": ["mrr", "recall_50", "ndcg_10"],
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "abstract_terms": terms,
            "relevant": {doc["mongo_id"]: 3},
            "notes": "Long abstract-only terms; minimal title overlap",
        })

    return {
        "version": 2,
        "description": "Hard v2 golden set — stresses paraphrase, noise, clusters, abstract-only recall",
        "corpus_documents": corpus.get("total_documents", len(docs)),
        "corpus_sample_size": corpus.get("sample_size", len(docs)),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_fields": ["title", "abstract", "keywords"],
        "categories": {
            "hard_exact_rank1": "Truncated long title — strict rank 1",
            "hard_partial_common": "Common-word partial title query",
            "hard_graded_cluster": "Multi-doc graded cluster",
            "hard_paraphrase": "Low-overlap NL paraphrase",
            "very_hard_abstract_only": "Abstract-only long terms",
        },
        "metric_definitions": {
            "recall_50": "Fraction of judged relevant docs in top 50",
            "precision_1": "Relevant doc at rank 1",
            "precision_5": "Relevant docs in top 5 / 5",
            "precision_10": "Relevant docs in top 10 / 10",
            "ndcg_10": "Graded ranking quality in top 10",
            "mrr": "Reciprocal rank of first relevant doc",
        },
        "queries": queries,
    }


def _type_counts(queries: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for q in queries:
        counts[q["type"]] += 1
    return dict(counts)


def main() -> None:
    print("=== Build golden v2 sets ===\n")
    if not CORPUS_PATH.exists():
        print(f"Missing corpus: {CORPUS_PATH}")
        print("Run: .venv/bin/python eval/scripts/sample_corpus_v2.py")
        sys.exit(1)

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    print(f"  Corpus: {corpus.get('total_documents')} docs")

    comprehensive = build_comprehensive_golden(corpus)
    hard = build_hard_golden(corpus)

    COMPREHENSIVE_PATH.write_text(json.dumps(comprehensive, indent=2), encoding="utf-8")
    HARD_PATH.write_text(json.dumps(hard, indent=2), encoding="utf-8")

    print(f"\nComprehensive: {COMPREHENSIVE_PATH}")
    print(f"  Queries: {len(comprehensive['queries'])}")
    for t, c in sorted(_type_counts(comprehensive["queries"]).items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")

    print(f"\nHard: {HARD_PATH}")
    print(f"  Queries: {len(hard['queries'])}")
    for t, c in sorted(_type_counts(hard["queries"]).items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")


if __name__ == "__main__":
    main()
