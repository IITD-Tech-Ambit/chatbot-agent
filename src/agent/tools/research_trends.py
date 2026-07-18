"""get_research_trends tool — paper count time series for a topic.

Uses OpenSearch retriever for semantic paper discovery (never field_associated),
then aggregates retrieved paper IDs by year in MongoDB.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent.tools.meta import annotate_tool
from agent.tools.deps import ToolDeps


class ResearchTrendsArgs(BaseModel):
    topic: str = Field(description="Research topic or field to get trends for, e.g. 'machine learning', 'renewable energy'")
    year_from: Optional[int] = Field(default=None, description="Start year (inclusive)")
    year_to: Optional[int] = Field(default=None, description="End year (inclusive)")


def build_tool(deps: ToolDeps) -> BaseTool:
    retriever = deps.retriever
    research_repo = deps.research_repo

    @tool(args_schema=ResearchTrendsArgs)
    async def get_research_trends(topic: str, year_from: int | None = None, year_to: int | None = None) -> str:
        """Get research publication trends (paper counts per year) for a topic or field."""
        try:
            hits = await retriever.retrieve(topic, top_k=400)
        except Exception as exc:
            return json.dumps({"topic": topic, "trend": [], "error": f"Retrieval failed: {type(exc).__name__}"})

        if not hits:
            return json.dumps({
                "topic": topic,
                "trend": [],
                "message": f"No publications found for '{topic}' in the IIT Delhi database.",
            })

        paper_ids = [r["id"] for r in hits if r.get("id")]
        trend_data = await research_repo.trend_by_ids(paper_ids, year_from, year_to)

        return json.dumps({
            "topic": topic,
            "year_from": year_from,
            "year_to": year_to,
            "total_papers_sampled": len(paper_ids),
            "trend": [
                {"year": e["_id"], "papers": e["count"]}
                for e in trend_data
                if e.get("_id") is not None
            ],
        }, default=str)

    return annotate_tool(
        get_research_trends,
        thinking_label="Analyzing publication trends",
        token_cap=deps.config.TOKEN_CAP_PUBLICATION_STATS,
    )
