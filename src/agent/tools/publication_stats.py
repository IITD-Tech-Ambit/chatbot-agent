"""get_publication_stats tool — publication counts by department / year / type.

Topic filtering always uses the OpenSearch retriever — never field_associated.
Department attribution uses: paper.kerberos → faculty.email prefix → faculty.department.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class PublicationStatsArgs(BaseModel):
    department: Optional[str] = Field(default=None, description='Specific department name, e.g. "Civil Engineering"')
    topic: Optional[str] = Field(default=None, description='Research topic filter, e.g. "machine learning". Uses semantic search.')
    year_from: Optional[int] = Field(default=None, description="Start year (inclusive)")
    year_to: Optional[int] = Field(default=None, description="End year (inclusive)")
    group_by: Optional[str] = Field(default=None, description="Group results by: department, year, or document_type")


@tool(args_schema=PublicationStatsArgs)
async def get_publication_stats(
    department: str | None = None,
    topic: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    group_by: str | None = None,
) -> str:
    """Get publication count statistics for IIT Delhi. Supports filtering by department, topic, and year range.
    Use group_by='department' to see papers per department. Add topic= to scope to a research area."""
    from agent.tools._registry import get_faculty_repo, get_research_repo, get_retriever, get_config

    faculty_repo = get_faculty_repo()
    research_repo = get_research_repo()
    retriever = get_retriever()
    cfg = get_config()

    # ── Year filter ──
    year_match: dict = {}
    if year_from:
        year_match["$gte"] = year_from
    if year_to:
        year_match["$lte"] = year_to
    base_match = {"publication_year": year_match} if year_match else {}

    # ── Topic filter: use retriever to get paper IDs, never field_associated ──
    topic_ids: list[str] | None = None
    if topic:
        hits = await retriever.retrieve(topic, top_k=400)
        topic_ids = [r["id"] for r in hits if r.get("id")]
        if not topic_ids:
            return json.dumps({
                "topic": topic,
                "total_papers": 0,
                "groups": [],
                "message": f"No publications found for topic '{topic}' in the IIT Delhi database.",
            })

    # ── Single-department stats ──
    if department:
        result = await _department_stats(department, base_match, topic_ids, faculty_repo, research_repo)
        output = json.dumps(result, default=str)
        cap = cfg.TOKEN_CAP_PUBLICATION_STATS
        return output[:cap] + '..."}' if len(output) > cap else output

    # ── Group by department (default) ──
    if not group_by or group_by == "department":
        result = await _global_department_stats(
            base_match, year_from, year_to, topic, topic_ids, faculty_repo, research_repo
        )
        output = json.dumps(result, default=str)
        cap = cfg.TOKEN_CAP_PUBLICATION_STATS
        return output[:cap] + '..."}' if len(output) > cap else output

    # ── Group by year or document_type ──
    if topic_ids is not None:
        # Filter aggregate to topic IDs
        result = await _aggregate_by_dimension_for_ids(
            topic_ids, base_match, group_by, year_from, year_to, research_repo
        )
    else:
        dimension = {"year": "$publication_year", "document_type": "$document_type"}.get(
            group_by, "$publication_year"
        )
        sort_field = "_id" if group_by == "year" else "count"
        total, buckets = await research_repo.global_stats(base_match, dimension, sort_field, limit=25)
        label_key = {"year": "year", "document_type": "type"}.get(group_by, "year")
        result = {
            "total_papers": total,
            "year_from": year_from,
            "year_to": year_to,
            "grouped_by": group_by,
            "groups": [
                {label_key: b["_id"], "papers": b["count"]}
                for b in buckets if b.get("_id") not in (None, "")
            ],
        }

    output = json.dumps(result, default=str)
    cap = cfg.TOKEN_CAP_PUBLICATION_STATS
    return output[:cap] + '..."}' if len(output) > cap else output


async def _aggregate_by_dimension_for_ids(
    paper_ids: list[str],
    base_match: dict,
    group_by: str,
    year_from: int | None,
    year_to: int | None,
    research_repo,
) -> dict:
    from bson import ObjectId

    oids = [ObjectId(i) for i in paper_ids if ObjectId.is_valid(str(i))]
    match = {**base_match, "_id": {"$in": oids}}
    dimension = {"year": "$publication_year", "document_type": "$document_type"}.get(
        group_by, "$publication_year"
    )
    label_key = {"year": "year", "document_type": "type"}.get(group_by, "year")
    sort_field = "_id" if group_by == "year" else "count"

    import asyncio
    total, buckets = await asyncio.gather(
        research_repo.count_documents(match),
        research_repo.aggregate([
            {"$match": match},
            {"$group": {"_id": dimension, "count": {"$sum": 1}}},
            {"$sort": {sort_field: -1}},
            {"$limit": 25},
        ]),
    )
    return {
        "total_papers": total,
        "year_from": year_from,
        "year_to": year_to,
        "grouped_by": group_by,
        "groups": [
            {label_key: b["_id"], "papers": b["count"]}
            for b in buckets if b.get("_id") not in (None, "")
        ],
    }


async def _global_department_stats(
    base_match: dict,
    year_from: int | None,
    year_to: int | None,
    topic: str | None,
    topic_ids: list[str] | None,
    faculty_repo,
    research_repo,
) -> dict:
    """Count papers per IIT Delhi department.
    Join: paper.kerberos → faculty.email prefix → faculty.department → department.name.
    If topic_ids provided, restrict to those papers only.
    """
    kerberos_to_dept = await faculty_repo.get_kerberos_to_dept_map()

    if topic_ids is not None:
        kerberos_rows = await research_repo.kerberos_counts_for_ids(topic_ids, base_match)
    else:
        kerberos_rows = await research_repo.papers_by_kerberos(base_match)

    dept_counts: dict[str, int] = {}
    for row in kerberos_rows:
        k = (row.get("_id") or "").lower().strip()
        dept = kerberos_to_dept.get(k)
        if dept:
            dept_counts[dept] = dept_counts.get(dept, 0) + row["count"]

    groups = sorted(
        [{"department": dept, "papers": cnt} for dept, cnt in dept_counts.items()],
        key=lambda x: x["papers"],
        reverse=True,
    )
    return {
        "total_papers": sum(g["papers"] for g in groups),
        "year_from": year_from,
        "year_to": year_to,
        "topic": topic,
        "grouped_by": "department",
        "groups": groups[:20],
    }


async def _department_stats(
    department: str, base_match: dict, topic_ids: list[str] | None, faculty_repo, research_repo
) -> dict:
    dept = await faculty_repo.find_department(department)
    if not dept:
        return {"error": f'No department matching "{department}" was found in the IIT Delhi database.'}

    faculty_docs = await faculty_repo.find_faculty_by_department_id(dept["_id"])
    kerberos_ids = [(f.get("email") or "").split("@")[0].lower() for f in faculty_docs if f.get("email")]
    scopus_ids = [str(s) for f in faculty_docs for s in (f.get("scopus_id") or [])]

    or_clauses: list[dict] = []
    if kerberos_ids:
        or_clauses.append({"kerberos": {"$in": kerberos_ids}})
    if scopus_ids:
        or_clauses.append({"authors.author_id": {"$in": scopus_ids}})

    if not or_clauses:
        return {"error": f'No faculty found in "{dept.get("name", department)}".'}

    match: dict = {**base_match, "$or": or_clauses}
    if topic_ids is not None:
        from bson import ObjectId
        oids = [ObjectId(i) for i in topic_ids if ObjectId.is_valid(str(i))]
        match["_id"] = {"$in": oids}

    stats = await research_repo.department_stats(match)
    return {
        "department": dept.get("name", department),
        "faculty_count": len(faculty_docs),
        **stats,
    }
