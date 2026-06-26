"""Heuristic answer quality metrics without an LLM judge."""

from __future__ import annotations

import re
from typing import Any


_STOPWORDS = {
    "about", "after", "also", "been", "being", "between", "both", "from",
    "have", "into", "more", "other", "research", "some", "such", "than",
    "that", "their", "there", "these", "this", "those", "through", "using",
    "were", "what", "when", "where", "which", "with", "would", "your",
    "papers", "paper", "study", "studies", "iit", "delhi", "assistant",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]{4,}", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def relevance_score(query: str, answer: str) -> float:
    """Query term recall in the answer."""
    q_terms = _tokenize(query)
    if not q_terms:
        return 0.0
    a_terms = _tokenize(answer)
    if not a_terms:
        return 0.0
    return len(q_terms & a_terms) / len(q_terms)


def faithfulness_score(answer: str, sources: list[dict[str, Any]]) -> float:
    """Fraction of substantive answer terms supported by source text."""
    source_text = " ".join(
        f"{s.get('title', '')} {s.get('abstract', '')}"
        for s in sources
    ).lower()
    if not source_text.strip():
        return 0.0

    answer_terms = _tokenize(answer)
    if not answer_terms:
        return 0.0

    supported = sum(1 for t in answer_terms if t in source_text)
    return supported / len(answer_terms)


def citation_consistency(answer: str, sources: list[dict[str, Any]]) -> float | None:
    """Check [N] citations map to plausible source titles mentioned nearby."""
    citations = re.findall(r"\[(\d+)\]", answer)
    if not citations:
        return None

    by_index = {s.get("index") or s.get("citation_index"): s for s in sources}
    valid = 0
    for cite in citations:
        idx = int(cite)
        if idx in by_index or any(s.get("index") == idx for s in sources):
            valid += 1
    return valid / len(citations)


def hallucination_flags(answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect unsupported specific claims (years, large numbers, author surnames)."""
    source_blob = " ".join(
        f"{s.get('title', '')} {s.get('abstract', '')} {s.get('authors', '')}"
        for s in sources
    ).lower()

    flags: list[str] = []

    # Years not in sources
    answer_years = set(re.findall(r"\b(?:19|20)\d{2}\b", answer))
    source_years = set(re.findall(r"\b(?:19|20)\d{2}\b", source_blob))
    unsupported_years = answer_years - source_years
    if unsupported_years:
        flags.append(f"unsupported_years:{','.join(sorted(unsupported_years))}")

    # Large citation counts
    for m in re.finditer(r"\b(\d{2,5})\s+citations?\b", answer, re.I):
        num = m.group(1)
        if num not in source_blob:
            flags.append(f"unsupported_citation_count:{num}")

    # Faculty names not in sources (simple heuristic: capitalized two-word names)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", answer):
        name = m.group(1).lower()
        if name not in source_blob and "research assistant" not in name.lower():
            flags.append(f"possible_unsupported_name:{m.group(1)}")

    rate = min(1.0, len(flags) / 3.0) if flags else 0.0
    return {"flags": flags, "hallucination_rate": rate}


def evaluate_answer(
    query: str,
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    hall = hallucination_flags(answer, sources)
    return {
        "relevance": relevance_score(query, answer),
        "faithfulness": faithfulness_score(answer, sources),
        "citation_consistency": citation_consistency(answer, sources),
        "hallucination_rate": hall["hallucination_rate"],
        "hallucination_flags": hall["flags"],
        "answer_length": len(answer),
    }
