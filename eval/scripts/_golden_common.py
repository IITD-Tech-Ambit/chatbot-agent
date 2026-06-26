"""Shared helpers for v3 corpus sampling and golden-set construction.

Query crafting rules (aligned with chatbot retriever):
  - Judgments and queries use ONLY: title, abstract, keywords, department, professor name.
  - Professor name is resolved via paper.kerberos -> faculties.email prefix.
  - NEVER use field_associated, authors, or author_names for query text or relevance.
  - Faculty display names use firstName + lastName only (no honorific title field).
  - Department comes from faculties.department -> departments.name.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable

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


def format_faculty_name(doc: dict[str, Any]) -> str:
    """Clean faculty name: firstName + lastName only (no Prof/Dr from title field)."""
    if doc.get("faculty_first_name") or doc.get("faculty_last_name"):
        return " ".join(
            p for p in (doc.get("faculty_first_name", ""), doc.get("faculty_last_name", "")) if p
        ).strip()
    raw = (doc.get("faculty_name") or "").strip()
    if not raw:
        return ""
    # Strip leading honorifics if legacy corpus still has combined title+name
    return re.sub(r"^(Prof\.?|Dr\.?|Mr\.?|Mrs\.?|Ms\.?)\s+", "", raw, flags=re.IGNORECASE).strip()


def pick_deterministic(
    arr: list[dict], n: int, stride: int = 7, *, sort_key: str = "mongo_id"
) -> list[dict]:
    if not arr:
        return []
    sorted_arr = sorted(arr, key=lambda d: str(d.get(sort_key, d.get("mongo_id", ""))))
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


def discover_title_clusters(docs: list[dict], min_docs: int = 3, max_clusters: int = 8) -> list[dict]:
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


def discover_abstract_clusters(
    docs: list[dict], min_docs: int = 3, max_clusters: int = 10
) -> list[dict]:
    """Topic clusters from abstract-only bigrams (not title)."""
    bigram_counts: dict[str, int] = defaultdict(int)
    doc_bigrams: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        words = extract_keywords(d.get("abstract", ""), 20)
        seen_bg: set[str] = set()
        for i in range(len(words) - 1):
            bg = f"{words[i]} {words[i + 1]}"
            if bg not in seen_bg:
                seen_bg.add(bg)
                bigram_counts[bg] += 1
                doc_bigrams[d["mongo_id"]].append(bg)
    clusters = []
    for query, count in sorted(bigram_counts.items(), key=lambda x: (-x[1], x[0])):
        if count < min_docs:
            continue
        w1, w2 = query.split(" ", 1)
        matching = [
            d for d in docs
            if w1 in (d.get("abstract") or "").lower() and w2 in (d.get("abstract") or "").lower()
        ]
        if len(matching) >= min_docs:
            clusters.append({"query": query, "matching": matching, "count": len(matching)})
        if len(clusters) >= max_clusters:
            break
    return clusters


def build_topic_cluster(
    docs: list[dict],
    query_words: list[str],
    *,
    field: str = "title",
    min_docs: int = 3,
    max_judged: int = 8,
) -> dict | None:
    terms = [w.lower() for w in query_words]
    text_key = "abstract" if field == "abstract" else "title"
    matching = [
        d for d in docs
        if all(w in (d.get(text_key) or "").lower() for w in terms)
    ]
    if len(matching) < min_docs:
        return None
    ranked = sorted(
        matching,
        key=lambda d: (
            sum(1 for w in terms if w in (d.get(text_key) or "").lower()),
            d.get("citation_count") or 0,
        ),
        reverse=True,
    )
    relevant: dict[str, int] = {}
    for i, d in enumerate(ranked[:max_judged]):
        relevant[d["mongo_id"]] = 3 if i == 0 else (2 if i < 3 else 1)
    return {
        "query": " ".join(query_words),
        "matching_docs": len(matching),
        "anchor": ranked[0],
        "relevant": relevant,
    }


def kerberos_groups(docs: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        k = (d.get("kerberos") or "").lower().strip()
        if k:
            groups[k].append(d)
    return groups


def surname_disambiguation_pairs(docs: list[dict]) -> list[dict]:
    """Find kerberos pairs sharing a last name but different people in corpus."""
    by_last: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        last = (d.get("faculty_last_name") or "").strip().lower()
        k = (d.get("kerberos") or "").lower()
        if last and k and d.get("faculty_department"):
            by_last[last].append(d)
    pairs: list[dict] = []
    for last, group in by_last.items():
        by_k: dict[str, dict] = {}
        for d in group:
            by_k.setdefault(d["kerberos"], d)
        if len(by_k) < 2:
            continue
        members = list(by_k.values())
        pairs.append({"last_name": last, "members": members[:3]})
    pairs.sort(key=lambda p: -len(p["members"]))
    return pairs


def department_topic_candidates(docs: list[dict], min_papers: int = 3) -> list[dict]:
    """Departments with enough papers for scoped queries."""
    by_dept: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        dept = (d.get("faculty_department") or "").strip()
        if dept:
            by_dept[dept].append(d)
    out = []
    for dept, papers in sorted(by_dept.items(), key=lambda x: -len(x[1])):
        if len(papers) >= min_papers:
            out.append({"department": dept, "papers": papers})
    return out


def cross_department_topics(docs: list[dict], min_depts: int = 2, min_papers: int = 4) -> list[dict]:
    """Abstract terms that appear across multiple departments."""
    term_depts: dict[str, set[str]] = defaultdict(set)
    term_docs: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        dept = (d.get("faculty_department") or "").strip()
        if not dept:
            continue
        for term in extract_keywords(d.get("abstract", ""), 6):
            if term in COMMON_TITLE_WORDS or len(term) < 5:
                continue
            term_depts[term].add(dept)
            if len(term_docs[term]) < 20:
                term_docs[term].append(d)
    candidates = []
    for term, depts in term_depts.items():
        if len(depts) >= min_depts and len(term_docs[term]) >= min_papers:
            candidates.append({
                "term": term,
                "departments": sorted(depts),
                "papers": term_docs[term],
            })
    candidates.sort(key=lambda c: (-len(c["departments"]), -len(c["papers"])))
    return candidates


def next_id(prefix: str, counter: list[int]) -> str:
    counter[0] += 1
    return f"{prefix}-{counter[0]}"


PARAPHRASE_TEMPLATES: list[Callable[[dict], str | None]] = []


def _init_paraphrase_templates() -> None:
    global PARAPHRASE_TEMPLATES
    if PARAPHRASE_TEMPLATES:
        return
    PARAPHRASE_TEMPLATES.extend([
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
    ])


HARD_PARAPHRASES: list[Callable[[dict], str | None]] = []


def _init_hard_paraphrases() -> None:
    global HARD_PARAPHRASES
    if HARD_PARAPHRASES:
        return
    HARD_PARAPHRASES.extend([
        lambda d: (
            f"research investigating {' and '.join(abstract_only_terms(d, 6, 3))} mechanisms"
            if len(abstract_only_terms(d, 6, 3)) >= 2
            else None
        ),
        lambda d: (
            f"what approaches exist for {abstract_only_terms(d, 5, 2)[0]} in applied science"
            if len(abstract_only_terms(d, 5, 2)) >= 1
            else None
        ),
        lambda d: (
            f"studies addressing {' '.join(abstract_only_terms(d, 6, 2))} challenges"
            if len(abstract_only_terms(d, 6, 2)) >= 2
            else None
        ),
    ])


_init_paraphrase_templates()
_init_hard_paraphrases()
