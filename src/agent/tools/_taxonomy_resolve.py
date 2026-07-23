"""Resolve a user-supplied theme/domain/department NAME to the slug/code the
taxonomy API expects. Three tiers: exact (slug or name) → substring →
token-overlap. The module name starts with `_` so the tool registry skips it.
"""

from __future__ import annotations

import re
from typing import Any

_STOPWORDS = frozenset({
    "the", "of", "and", "for", "in", "a", "an", "to", "&",
    "research", "area", "areas", "theme", "thematic", "domain", "domains",
    "field", "fields", "department", "dept", "studies",
})


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w and w not in _STOPWORDS}


def resolve_node(
    items: list[dict[str, Any]], term: str, *, name_key: str = "name", slug_key: str = "slug"
) -> dict[str, Any] | None:
    """Best match for `term` among `items` (each with a name + slug/code)."""
    if not term or not items:
        return None
    t = term.strip().lower()

    # 1) exact slug or name
    for it in items:
        if str(it.get(slug_key, "")).lower() == t or str(it.get(name_key, "")).lower() == t:
            return it

    # 2) slugified term equals a slug
    tslug = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    if tslug:
        for it in items:
            if str(it.get(slug_key, "")).lower() == tslug:
                return it

    # 3) substring either direction
    for it in items:
        n = str(it.get(name_key, "")).lower()
        if n and (t in n or n in t):
            return it

    # 4) token-overlap (Jaccard), require a reasonable overlap
    tt = _tokens(term)
    if not tt:
        return None
    best, best_score = None, 0.0
    for it in items:
        it_tokens = _tokens(str(it.get(name_key, "")))
        if not it_tokens:
            continue
        score = len(tt & it_tokens) / len(tt | it_tokens)
        if score > best_score:
            best, best_score = it, score
    return best if best_score >= 0.3 else None
