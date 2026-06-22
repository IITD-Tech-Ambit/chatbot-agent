"""find_interdisciplinary_papers tool — papers at the intersection of multiple fields."""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from agent.tools._registry import get_retriever, get_research_repo, get_config

logger = logging.getLogger(__name__)


class InterdisciplinaryPapersArgs(BaseModel):
    fields: list[str] = Field(
        description="List of 2-4 research fields or topics to find intersection papers for, e.g. ['machine learning', 'healthcare']",
        min_length=2,
        max_length=4,
    )
    year_from: Optional[int] = Field(default=None, ge=1950, le=2030)
    year_to: Optional[int] = Field(default=None, ge=1950, le=2030)
    limit: int = Field(default=8, ge=1, le=15)

    @model_validator(mode="after")
    def _check_year_range(self) -> "InterdisciplinaryPapersArgs":
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from must be <= year_to")
        return self

    @model_validator(mode="after")
    def _check_fields_length(self) -> "InterdisciplinaryPapersArgs":
        if len(self.fields) < 2:
            raise ValueError("At least 2 fields are required")
        return self


@tool("find_interdisciplinary_papers", args_schema=InterdisciplinaryPapersArgs)
async def find_interdisciplinary_papers(
    fields: list[str],
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 8,
) -> str:
    """Find IIT Delhi research papers that sit at the intersection of multiple research fields or topics (e.g. ML + healthcare, nanotechnology + energy)."""
    retriever = get_retriever()
    research_repo = get_research_repo()
    cfg = get_config()

    # Retrieve top-k papers per field using hybrid BM25+kNN
    per_field_ids: list[set[str]] = []
    for field in fields:
        try:
            papers = await retriever.retrieve(field, top_k=30, abstract_max_chars=0)
            per_field_ids.append({p["id"] for p in papers if p.get("id")})
        except Exception as exc:
            logger.warning("Retrieval failed for field '%s': %s", field, exc)
            per_field_ids.append(set())

    if not any(per_field_ids):
        return json.dumps({"fields": fields, "papers": [], "message": "No papers found."})

    # Intersection first; fall back to papers appearing in ≥2 fields
    common_ids: set[str] = set.intersection(*per_field_ids) if len(per_field_ids) > 1 else per_field_ids[0]
    if not common_ids:
        counter: Counter = Counter(pid for s in per_field_ids for pid in s)
        common_ids = {pid for pid, cnt in counter.most_common(limit * 3) if cnt >= 2}

    if not common_ids:
        # MongoDB fallback
        mongo_docs = await research_repo.find_interdisciplinary_papers(fields, limit=limit)
    else:
        ids_list = list(common_ids)[: limit * 2]
        mongo_docs = await research_repo.find_by_ids(ids_list)

    # Apply year filter post-fetch
    if year_from or year_to:
        mongo_docs = [
            d for d in mongo_docs
            if (not year_from or (d.get("publication_year") or 0) >= year_from)
            and (not year_to or (d.get("publication_year") or 9999) <= year_to)
        ]

    mongo_docs = mongo_docs[:limit]

    papers = [
        {
            "title": d.get("title", ""),
            "abstract": (d.get("abstract") or "")[:150],
            "authors": [a.get("author_name", "") for a in (d.get("authors") or [])[:4]],
            "year": d.get("publication_year"),
            "field": d.get("field_associated", ""),
            "citations": d.get("citation_count", 0),
            "link": d.get("link", ""),
        }
        for d in mongo_docs
    ]

    result = {
        "fields": fields,
        "year_range": {"from": year_from, "to": year_to},
        "count": len(papers),
        "papers": papers,
    }

    output = json.dumps(result, default=str)
    cap = cfg.TOKEN_CAP_INTERDISCIPLINARY
    if len(output) > cap:
        output = output[:cap] + '..."}'
    return output
