"""find_similar_papers tool — re-embed title+abstract, then kNN 'more like this'."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class SimilarPapersArgs(BaseModel):
    title: str = Field(description="Title of the reference paper")
    abstract: str = Field(default="", description="Abstract of the reference paper (optional)")
    top_k: int = Field(default=5, description="Number of similar papers to return")


@tool(args_schema=SimilarPapersArgs)
async def find_similar_papers(title: str, abstract: str = "", top_k: int = 5) -> str:
    """Find papers similar to a given paper by re-embedding its title and abstract."""
    from agent.tools._registry import get_retriever

    retriever = get_retriever()

    combined = f"{title}. {abstract}".strip()
    try:
        papers = await retriever.retrieve(combined, top_k=min(top_k, 10), abstract_max_chars=150)
    except Exception as exc:
        return json.dumps({"reference": title, "similar_papers": [], "error": f"Retrieval failed: {type(exc).__name__}"})

    papers = [p for p in papers if p.get("title", "").lower().strip() != title.lower().strip()]

    return json.dumps({
        "reference": title,
        "similar_papers": [
            {
                "title": p["title"],
                "authors": p["authors"][:3],
                "year": p.get("publication_year"),
                "citations": p.get("citation_count", 0),
                "abstract": p["abstract"],
            }
            for p in papers[:top_k]
        ],
    }, default=str)
