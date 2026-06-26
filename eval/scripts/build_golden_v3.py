#!/usr/bin/env python3
"""
Build v3 golden sets from corpus_v3.json.

Query dimensions (ONLY): title, abstract, keywords, department, professor name (via kerberos).
Never uses field_associated or author_names.

Outputs:
  eval/fixtures/golden_comprehensive_v3.json  (~150 queries)
  eval/fixtures/golden_hard_v3.json            (~60 hard/extra-hard queries)

Usage:
  cd chatbot-agent
  .venv/bin/python eval/scripts/build_golden_v3.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from eval.scripts._golden_common import (  # noqa: E402
    HARD_PARAPHRASES,
    PARAPHRASE_TEMPLATES,
    abstract_only_terms,
    build_topic_cluster,
    common_word_ratio,
    cross_department_topics,
    department_topic_candidates,
    discover_abstract_clusters,
    discover_title_clusters,
    extract_keywords,
    format_faculty_name,
    kerberos_groups,
    next_id,
    pick_deterministic,
    surname_disambiguation_pairs,
    title_snippet,
    tokenize,
)

CORPUS_PATH = _ROOT / "eval" / "fixtures" / "corpus_v3.json"
COMPREHENSIVE_PATH = _ROOT / "eval" / "fixtures" / "golden_comprehensive_v3.json"
HARD_PATH = _ROOT / "eval" / "fixtures" / "golden_hard_v3.json"

FACULTY_TEMPLATES = [
    "papers by {full_name}",
    "publications by {full_name}",
    "research by {full_name}",
    "what has {full_name} published",
    "papers by {full_name} on {topic}",
]

FACULTY_DISAMBIG_TEMPLATES = [
    "papers by {full_name} from {department}",
    "{full_name} ({department}) publications",
    "research by {full_name} in {department} on {topic}",
]


def _faculty_query(doc: dict, template_idx: int, *, force_dept: bool = False) -> str:
    full_name = format_faculty_name(doc)
    dept = (doc.get("faculty_department") or "").strip()
    topic_kw = extract_keywords(doc.get("title", ""), 1)
    topic = topic_kw[0] if topic_kw else "research"
    if force_dept and dept:
        tpl = FACULTY_DISAMBIG_TEMPLATES[template_idx % len(FACULTY_DISAMBIG_TEMPLATES)]
        return tpl.format(full_name=full_name, department=dept, topic=topic)
    tpl = FACULTY_TEMPLATES[template_idx % len(FACULTY_TEMPLATES)]
    return tpl.format(full_name=full_name, topic=topic)


def build_comprehensive_golden(corpus: dict[str, Any]) -> dict[str, Any]:
    docs = corpus.get("documents", [])
    docs_rich = [
        d for d in docs
        if d.get("abstract") and len(d["abstract"]) > 80 and len(d.get("title", "")) > 10
    ]
    queries: list[dict[str, Any]] = []
    qid = [0]

    # Standard tier
    for doc in pick_deterministic(docs_rich, 25, 5):
        queries.append({
            "id": next_id("exact-title", qid),
            "query": title_snippet(doc["title"]),
            "type": "exact_title",
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "relevant": {doc["mongo_id"]: 3},
        })

    for doc in pick_deterministic(docs_rich, 25, 6):
        kw = extract_keywords(doc["title"], 4)
        if len(kw) < 2:
            continue
        queries.append({
            "id": next_id("partial-title", qid),
            "query": " ".join(kw),
            "type": "partial_title",
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "relevant": {doc["mongo_id"]: 3},
        })

    for doc in pick_deterministic(docs_rich, 20, 8):
        kw = extract_keywords(doc["abstract"], 4)
        if len(kw) < 3:
            continue
        queries.append({
            "id": next_id("abstract-kw", qid),
            "query": " ".join(kw),
            "type": "abstract_keyword",
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "relevant": {doc["mongo_id"]: 2},
        })

    for i, doc in enumerate(pick_deterministic(docs_rich, 15, 13)):
        paraphrase = PARAPHRASE_TEMPLATES[i % len(PARAPHRASE_TEMPLATES)](doc)
        if not paraphrase:
            continue
        queries.append({
            "id": next_id("semantic-paraphrase", qid),
            "query": paraphrase,
            "type": "semantic_paraphrase",
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "relevant": {doc["mongo_id"]: 2},
        })

    # Hard tier (in comprehensive)
    gap_docs = sorted(
        [(d, abstract_only_terms(d)) for d in docs_rich],
        key=lambda x: len(x[1]),
        reverse=True,
    )
    for doc in pick_deterministic([d for d, t in gap_docs if len(t) >= 3], 12, 7):
        terms = abstract_only_terms(doc)
        if len(terms) < 3:
            continue
        queries.append({
            "id": next_id("abstract-gap", qid),
            "query": " ".join(terms[:3]),
            "type": "abstract_gap",
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "relevant": {doc["mongo_id"]: 3},
            "notes": "Terms in abstract but not title",
        })

    groups = kerberos_groups(docs)
    surname_pairs = {p["last_name"] for p in surname_disambiguation_pairs(docs)}
    faculty_candidates = sorted(
        [(k, p) for k, p in groups.items() if p and format_faculty_name(p[0])],
        key=lambda x: (-len(x[1]), x[0]),
    )
    for i, (kerberos, papers) in enumerate(faculty_candidates[:15]):
        lead = papers[0]
        last = (lead.get("faculty_last_name") or "").lower()
        force_dept = last in surname_pairs
        query = _faculty_query(lead, i, force_dept=force_dept)
        relevant = {p["mongo_id"]: 3 if j == 0 else 2 for j, p in enumerate(papers[:5])}
        queries.append({
            "id": next_id("faculty-kerberos", qid),
            "query": query,
            "type": "faculty_kerberos",
            "kerberos": kerberos,
            "faculty_name": format_faculty_name(lead),
            "faculty_department": lead.get("faculty_department"),
            "source_mongo_id": lead["mongo_id"],
            "source_title": lead["title"],
            "relevant": relevant,
            "notes": "Faculty via kerberos; department included when surname ambiguous",
        })

    for i, cand in enumerate(department_topic_candidates(docs, min_papers=4)[:12]):
        dept = cand["department"]
        lead = pick_deterministic(cand["papers"], 1, 3)[0]
        topic_kw = extract_keywords(lead.get("title", ""), 2) or extract_keywords(
            lead.get("abstract", ""), 2
        )
        if not topic_kw:
            continue
        topic = " ".join(topic_kw)
        dept_papers = cand["papers"]
        matching = [
            p for p in dept_papers
            if any(w in (p.get("title") or "").lower() or w in (p.get("abstract") or "").lower()
                     for w in topic_kw)
        ]
        relevant = {p["mongo_id"]: 3 if j == 0 else 2 for j, p in enumerate(matching[:6])}
        if not relevant:
            relevant = {lead["mongo_id"]: 3}
        queries.append({
            "id": next_id("dept-scoped", qid),
            "query": f"Papers from {dept} on {topic}",
            "type": "department_scoped",
            "department": dept,
            "topic": topic,
            "source_mongo_id": lead["mongo_id"],
            "relevant": relevant,
            "notes": "Department via faculty kerberos link, topic from title/abstract",
        })

    # Multi-hop
    by_year: dict[int, list[dict]] = defaultdict(list)
    for d in docs_rich:
        y = d.get("publication_year")
        if y:
            by_year[y].append(d)
    years_sorted = sorted(by_year.keys(), reverse=True)
    for year in pick_deterministic(
        [{"mongo_id": str(y), "year": y} for y in years_sorted], 8, 3, sort_key="year"
    ):
        y = year["year"]
        doc = pick_deterministic(by_year[y], 1, 1)[0]
        kw = extract_keywords(doc["title"], 2)
        if not kw:
            continue
        year_docs = [
            d for d in by_year[y]
            if any(w in (d.get("title") or "").lower() or w in (d.get("abstract") or "").lower()
                     for w in kw)
        ]
        relevant = {d["mongo_id"]: 2 for d in year_docs[:5]}
        relevant[doc["mongo_id"]] = 3
        queries.append({
            "id": next_id("multihop-temporal", qid),
            "query": f"What research on {' '.join(kw)} was published in {y}?",
            "type": "multi_hop_temporal",
            "publication_year": y,
            "source_mongo_id": doc["mongo_id"],
            "relevant": relevant,
            "difficulty": "high",
        })

    for cluster in discover_title_clusters(docs_rich, min_docs=2, max_clusters=8):
        pair = pick_deterministic(cluster["matching"], 2, 1)
        if len(pair) < 2:
            continue
        queries.append({
            "id": next_id("multihop-topic", qid),
            "query": (
                f"Compare findings on {cluster['query']} between "
                f"\"{title_snippet(pair[0]['title'], 50)}\" and "
                f"\"{title_snippet(pair[1]['title'], 50)}\""
            ),
            "type": "multi_hop_topic",
            "relevant": {pair[0]["mongo_id"]: 3, pair[1]["mongo_id"]: 3},
            "difficulty": "high",
        })

    for kerberos, papers in faculty_candidates[:10]:
        doc = papers[0]
        kw = extract_keywords(doc.get("title", ""), 2) or extract_keywords(doc.get("abstract", ""), 2)
        if not kw:
            continue
        name = format_faculty_name(doc)
        matching = [
            p for p in papers
            if any(w in (p.get("title") or "").lower() or w in (p.get("abstract") or "").lower()
                     for w in kw)
        ]
        relevant = {p["mongo_id"]: 3 for p in matching[:3]} or {doc["mongo_id"]: 3}
        entry: dict[str, Any] = {
            "id": next_id("multihop-faculty", qid),
            "query": f"What has {name} published about {' '.join(kw)}?",
            "type": "multi_hop_faculty",
            "kerberos": kerberos,
            "relevant": relevant,
            "difficulty": "high",
        }
        if (doc.get("faculty_last_name") or "").lower() in surname_pairs and doc.get("faculty_department"):
            entry["query"] = (
                f"What has {name} from {doc['faculty_department']} published about {' '.join(kw)}?"
            )
        queries.append(entry)

    return {
        "version": 3,
        "description": "Comprehensive v3 — title/abstract/keywords/department/faculty via kerberos",
        "corpus_documents": corpus.get("total_documents", len(docs)),
        "corpus_sample_size": corpus.get("sample_size", len(docs)),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_fields": ["title", "abstract", "keywords", "kerberos", "faculty_department"],
        "excluded_from_queries": ["field_associated", "authors", "author_names"],
        "categories": {
            "exact_title": "Title substring — source doc in top 10",
            "partial_title": "Title keywords — source doc in top 50",
            "abstract_keyword": "Distinctive abstract terms",
            "semantic_paraphrase": "Paraphrased title — vector recall",
            "abstract_gap": "Abstract-only terms absent from title",
            "faculty_kerberos": "Full faculty name; relevance via paper.kerberos",
            "department_scoped": "Department + topic scoped retrieval",
            "multi_hop_temporal": "Year + topic combined query",
            "multi_hop_topic": "Cross-paper topic comparison",
            "multi_hop_faculty": "Faculty + topic multi-hop",
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
        8, 5,
    ):
        queries.append({
            "id": next_id("hard-exact", qid),
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
        [
            d for d in docs
            if common_word_ratio(d["title"]) >= 0.45 and len(extract_keywords(d["title"], 5)) >= 3
        ],
        key=lambda d: common_word_ratio(d["title"]),
        reverse=True,
    )
    for doc in pick_deterministic(hard_partial, 8, 4):
        kw = extract_keywords(doc["title"], 4)
        queries.append({
            "id": next_id("hard-partial", qid),
            "query": " ".join(kw),
            "type": "hard_partial_common",
            "difficulty": "high",
            "metric_focus": ["precision_10", "mrr", "recall_50"],
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "relevant": {doc["mongo_id"]: 3},
            "notes": "Corpus-common title terms; many distractors",
        })

    for cluster in discover_abstract_clusters(docs_with_abstract, min_docs=3, max_clusters=10):
        ranked = pick_deterministic(cluster["matching"], min(8, len(cluster["matching"])), 2)
        relevant = {
            d["mongo_id"]: 3 if i == 0 else (2 if i < 3 else 1)
            for i, d in enumerate(ranked)
        }
        queries.append({
            "id": next_id("hard-cluster", qid),
            "query": cluster["query"],
            "type": "hard_graded_cluster",
            "difficulty": "high",
            "metric_focus": ["ndcg_10", "recall_50", "precision_10"],
            "matching_docs": cluster["count"],
            "source_mongo_id": ranked[0]["mongo_id"],
            "source_title": ranked[0]["title"],
            "relevant": relevant,
            "notes": "Graded cluster from abstract bigrams (not title)",
        })
        if len([q for q in queries if q["type"] == "hard_graded_cluster"]) >= 10:
            break

    manual_abstract_clusters = [
        ["photovoltaic", "efficiency"],
        ["finite", "element"],
        ["neural", "network"],
        ["machine", "learning"],
        ["quantum", "dots"],
    ]
    for terms in manual_abstract_clusters:
        cluster = build_topic_cluster(docs_with_abstract, terms, field="abstract", min_docs=2, max_judged=6)
        if not cluster:
            continue
        queries.append({
            "id": next_id("hard-cluster", qid),
            "query": cluster["query"],
            "type": "hard_graded_cluster",
            "difficulty": "medium",
            "matching_docs": cluster["matching_docs"],
            "source_mongo_id": cluster["anchor"]["mongo_id"],
            "source_title": cluster["anchor"]["title"],
            "relevant": cluster["relevant"],
            "notes": "Manual abstract keyword cluster",
        })

    for i, doc in enumerate(pick_deterministic(docs_with_abstract, 10, 8)):
        paraphrase = HARD_PARAPHRASES[i % len(HARD_PARAPHRASES)](doc)
        if not paraphrase:
            continue
        title_tokens = set(tokenize(doc["title"]))
        para_tokens = tokenize(paraphrase)
        overlap = sum(1 for t in para_tokens if t in title_tokens) / (len(para_tokens) or 1)
        if overlap > 0.35:
            continue
        queries.append({
            "id": next_id("hard-paraphrase", qid),
            "query": paraphrase,
            "type": "hard_paraphrase",
            "difficulty": "high",
            "metric_focus": ["mrr", "recall_50", "ndcg_10"],
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "lexical_overlap": round(overlap, 2),
            "relevant": {doc["mongo_id"]: 2},
            "notes": "NL paraphrase from abstract terms; low title overlap",
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
            "id": next_id("very-hard-abstract", qid),
            "query": " ".join(terms[:4]),
            "type": "very_hard_abstract_only",
            "difficulty": "very_high",
            "metric_focus": ["mrr", "recall_50", "ndcg_10"],
            "source_mongo_id": doc["mongo_id"],
            "source_title": doc["title"],
            "abstract_terms": terms,
            "relevant": {doc["mongo_id"]: 3},
            "notes": "Rare abstract-only n-grams; minimal title overlap",
        })

    groups = kerberos_groups(docs)
    for pair in surname_disambiguation_pairs(docs)[:6]:
        for member in pair["members"][:2]:
            full_name = format_faculty_name(member)
            dept = member.get("faculty_department", "")
            kerberos = member["kerberos"]
            papers = groups.get(kerberos, [member])
            query = f"papers by {full_name} from {dept}" if dept else f"papers by {full_name}"
            relevant = {p["mongo_id"]: 3 if i == 0 else 2 for i, p in enumerate(papers[:4])}
            queries.append({
                "id": next_id("kerberos-disambig", qid),
                "query": query,
                "type": "kerberos_disambiguation",
                "kerberos": kerberos,
                "faculty_name": full_name,
                "faculty_department": dept,
                "shared_surname": pair["last_name"],
                "difficulty": "very_high",
                "relevant": relevant,
                "notes": "Same surname different kerberos; department disambiguates",
            })
            if len([q for q in queries if q["type"] == "kerberos_disambiguation"]) >= 6:
                break
        if len([q for q in queries if q["type"] == "kerberos_disambiguation"]) >= 6:
            break

    for cand in cross_department_topics(docs)[:6]:
        term = cand["term"]
        depts = cand["departments"][:3]
        papers = cand["papers"]
        relevant: dict[str, int] = {}
        for i, p in enumerate(papers[:8]):
            relevant[p["mongo_id"]] = 3 if i < 2 else (2 if i < 5 else 1)
        queries.append({
            "id": next_id("cross-dept", qid),
            "query": f"Research on {term} across departments ({', '.join(depts[:2])})",
            "type": "cross_department_topic",
            "term": term,
            "departments": depts,
            "difficulty": "very_high",
            "source_mongo_id": papers[0]["mongo_id"],
            "relevant": relevant,
            "notes": "Topic in abstract spanning multiple faculty departments",
        })

    return {
        "version": 3,
        "description": "Hard/extra-hard v3 — abstract clusters, disambiguation, cross-department",
        "corpus_documents": corpus.get("total_documents", len(docs)),
        "corpus_sample_size": corpus.get("sample_size", len(docs)),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_fields": ["title", "abstract", "keywords", "kerberos", "faculty_department"],
        "excluded_from_queries": ["field_associated", "authors", "author_names"],
        "categories": {
            "hard_exact_rank1": "Truncated long title — strict rank 1",
            "hard_partial_common": "Common-word partial title query",
            "hard_graded_cluster": "Multi-doc graded cluster (abstract keywords)",
            "hard_paraphrase": "Low-overlap NL paraphrase from abstract",
            "very_hard_abstract_only": "Rare abstract-only n-grams",
            "kerberos_disambiguation": "Same surname, different kerberos + department",
            "cross_department_topic": "Abstract topic spanning departments",
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
    print("=== Build golden v3 sets ===\n")
    if not CORPUS_PATH.exists():
        print(f"Missing corpus: {CORPUS_PATH}")
        print("Run: .venv/bin/python eval/scripts/sample_corpus_v3.py")
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
