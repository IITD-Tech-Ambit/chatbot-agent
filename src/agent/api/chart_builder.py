"""Convert tool output dicts into typed chart SSE payloads.

Tools return domain data (JSON dicts). This module transforms that data into
ChartEvent instances for the frontend to render. Keeps tools pure and
the HTTP/SSE layer responsible for presentation decisions.
"""

from __future__ import annotations

from typing import Any, Callable

from agent.api.sse_events import (
    BarChartData,
    ChartEvent,
    ChartPayload,
    ChartSeries,
    DataPoint,
    LineChartData,
    PieChartData,  # kept for compare_faculty and other future pie uses
)


def build_chart_for_tool(tool_name: str, tool_output: dict[str, Any]) -> ChartEvent | None:
    builder = _CHART_BUILDERS.get(tool_name)
    if not builder:
        return None
    chart_data = builder(tool_output)
    if chart_data is None:
        return None
    return ChartEvent(tool_name=tool_name, chart=chart_data)


# ── Per-tool builders ──

def _build_research_trends_chart(data: dict[str, Any]) -> LineChartData | None:
    trend = data.get("trend", [])
    if not trend:
        return None
    topic = data.get("topic", "Research")
    points = sorted(
        [DataPoint(x=t["year"], y=t["papers"]) for t in trend if t.get("year") and t.get("papers") is not None],
        key=lambda p: p.x,
    )
    if not points:
        return None
    return LineChartData(
        title=f"Research Trends — {topic}",
        x_label="Year",
        y_label="Papers Published",
        series=[ChartSeries(label=topic, data=points)],
    )


def _build_compare_faculty_chart(data: dict[str, Any]) -> BarChartData | None:
    comparison = data.get("comparison", [])
    if len(comparison) < 2:
        return None
    names = [p.get("name", f"Faculty {i+1}") for i, p in enumerate(comparison)]
    series = [
        {"label": "H-Index", "data": [p.get("h_index", 0) or 0 for p in comparison]},
        {"label": "Citations (÷100)", "data": [round((p.get("total_citations", 0) or 0) / 100, 1) for p in comparison]},
        {"label": "Total Papers", "data": [p.get("total_papers", 0) or 0 for p in comparison]},
    ]
    return BarChartData(
        title="Faculty Comparison",
        x_label="Metric",
        y_label="Value",
        categories=names,
        series=series,
    )


def _build_department_profile_chart(data: dict[str, Any]) -> BarChartData | None:
    by_year = data.get("publication_stats", {}).get("papers_by_recent_year", [])
    if not by_year:
        return None
    by_year_sorted = sorted(
        [y for y in by_year if y.get("year")],
        key=lambda y: y["year"],
    )
    dept_name = data.get("department", {}).get("name", "Department")
    return BarChartData(
        title=f"{dept_name} — Publications by Year",
        x_label="Year",
        y_label="Papers",
        categories=[str(y["year"]) for y in by_year_sorted],
        series=[{"label": "Papers", "data": [y["count"] for y in by_year_sorted]}],
    )


def _build_publication_stats_chart(data: dict[str, Any]) -> ChartPayload | None:
    groups = data.get("groups", [])
    if not groups:
        return None
    grouped_by = data.get("grouped_by", "field")
    if grouped_by == "year":
        sorted_groups = sorted([g for g in groups if g.get("year")], key=lambda g: g["year"])
        return BarChartData(
            title="Publications by Year",
            x_label="Year",
            y_label="Papers",
            categories=[str(g["year"]) for g in sorted_groups],
            series=[{"label": "Papers", "data": [g["papers"] for g in sorted_groups]}],
        )
    label_key = {"type": "type", "department": "department"}.get(grouped_by, "field")
    top = [g for g in groups if g.get("papers")][:15]
    top_sorted = sorted(top, key=lambda g: g["papers"])
    title_map = {
        "department": "Publications by Department",
        "type": "Publications by Document Type",
        "field": "Publications by Research Field",
    }
    return BarChartData(
        title=title_map.get(grouped_by, f"Publications by {grouped_by.replace('_', ' ').title()}"),
        x_label="Papers",
        y_label="",
        layout="horizontal",
        categories=[str(g.get(label_key, "Unknown")) for g in top_sorted],
        series=[{"label": "Papers", "data": [g["papers"] for g in top_sorted]}],
    )


_CHART_BUILDERS: dict[str, Callable[[dict[str, Any]], ChartPayload | None]] = {
    "get_research_trends": _build_research_trends_chart,
    "compare_faculty": _build_compare_faculty_chart,
    "get_department_profile": _build_department_profile_chart,
    "get_publication_stats": _build_publication_stats_chart,
}
