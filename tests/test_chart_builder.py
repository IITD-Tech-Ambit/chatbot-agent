"""Tests for chart_builder.py and sse_events.py schemas."""

from __future__ import annotations

import pytest

from agent.api.chart_builder import build_chart_for_tool
from agent.api.sse_events import (
    BarChartData,
    ChartEvent,
    DoneEvent,
    ErrorEvent,
    LineChartData,
    PieChartData,
    StatusEvent,
    ThinkingEvent,
    TokenEvent,
)


# ── SSE event schema tests ──

class TestSSEEventSchemas:
    def test_status_event_serialization(self):
        ev = StatusEvent(text="Searching...")
        d = ev.model_dump()
        assert d == {"text": "Searching..."}

    def test_thinking_event_with_detail(self):
        ev = ThinkingEvent(step="Loading faculty", detail="CSE department")
        d = ev.model_dump()
        assert d["step"] == "Loading faculty"
        assert d["detail"] == "CSE department"

    def test_thinking_event_no_detail(self):
        ev = ThinkingEvent(step="Searching publications")
        assert ev.detail is None

    def test_token_event(self):
        ev = TokenEvent(text="Hello ")
        assert ev.model_dump() == {"text": "Hello "}

    def test_done_event_defaults(self):
        ev = DoneEvent(took_ms=123)
        d = ev.model_dump()
        assert d["cached"] is False
        assert d["took_ms"] == 123

    def test_done_event_cached(self):
        ev = DoneEvent(took_ms=5, cached=True)
        assert ev.cached is True

    def test_error_event(self):
        ev = ErrorEvent(message="Something failed")
        assert ev.model_dump() == {"message": "Something failed"}

    def test_line_chart_discriminator(self):
        chart = LineChartData(title="Test", series=[])
        assert chart.chart_type == "line"

    def test_bar_chart_discriminator(self):
        chart = BarChartData(title="Test", categories=[], series=[])
        assert chart.chart_type == "bar"

    def test_pie_chart_discriminator(self):
        chart = PieChartData(title="Test", slices=[])
        assert chart.chart_type == "pie"

    def test_chart_event_model_dump(self):
        chart = LineChartData(title="Trend", series=[])
        ev = ChartEvent(tool_name="get_research_trends", chart=chart)
        d = ev.model_dump()
        assert d["tool_name"] == "get_research_trends"
        assert d["chart"]["chart_type"] == "line"
        assert d["chart"]["title"] == "Trend"


# ── build_chart_for_tool tests ──

class TestBuildChartForTool:
    def test_unknown_tool_returns_none(self):
        result = build_chart_for_tool("search_papers", {"papers": []})
        assert result is None

    def test_empty_trend_returns_none(self):
        result = build_chart_for_tool("get_research_trends", {"trend": [], "topic": "ML"})
        assert result is None

    def test_research_trends_builds_line_chart(self):
        data = {
            "topic": "Machine Learning",
            "trend": [
                {"year": 2020, "papers": 5},
                {"year": 2021, "papers": 8},
                {"year": 2022, "papers": 12},
            ],
        }
        result = build_chart_for_tool("get_research_trends", data)
        assert result is not None
        assert isinstance(result, ChartEvent)
        assert result.chart.chart_type == "line"
        assert "Machine Learning" in result.chart.title
        assert result.tool_name == "get_research_trends"

    def test_research_trends_series_length(self):
        data = {
            "topic": "ML",
            "trend": [{"year": 2020, "papers": 5}, {"year": 2021, "papers": 7}],
        }
        result = build_chart_for_tool("get_research_trends", data)
        assert result is not None
        chart = result.chart
        assert len(chart.series) == 1  # type: ignore[union-attr]
        assert len(chart.series[0].data) == 2  # type: ignore[union-attr]

    def test_compare_faculty_needs_two_entries(self):
        data = {"comparison": [{"name": "Prof. A", "h_index": 20}]}
        result = build_chart_for_tool("compare_faculty", data)
        assert result is None  # Only 1 person — no comparison chart

    def test_compare_faculty_builds_bar_chart(self):
        data = {
            "comparison": [
                {"name": "Prof. A", "h_index": 25, "total_citations": 3000, "total_papers": 80},
                {"name": "Prof. B", "h_index": 18, "total_citations": 1500, "total_papers": 50},
            ]
        }
        result = build_chart_for_tool("compare_faculty", data)
        assert result is not None
        assert result.chart.chart_type == "bar"
        assert len(result.chart.categories) == 2  # type: ignore[union-attr]
        assert "Prof. A" in result.chart.categories  # type: ignore[union-attr]

    def test_department_profile_builds_bar_chart(self):
        data = {
            "department": {"name": "Computer Science"},
            "publication_stats": {
                "papers_by_recent_year": [
                    {"year": 2021, "count": 40},
                    {"year": 2022, "count": 55},
                    {"year": 2023, "count": 70},
                ]
            },
        }
        result = build_chart_for_tool("get_department_profile", data)
        assert result is not None
        assert result.chart.chart_type == "bar"
        assert "Computer Science" in result.chart.title

    def test_department_profile_empty_returns_none(self):
        data = {"department": {"name": "CS"}, "publication_stats": {}}
        result = build_chart_for_tool("get_department_profile", data)
        assert result is None

    def test_publication_stats_year_builds_bar(self):
        data = {
            "grouped_by": "year",
            "groups": [{"year": 2020, "papers": 50}, {"year": 2021, "papers": 60}],
        }
        result = build_chart_for_tool("get_publication_stats", data)
        assert result is not None
        assert result.chart.chart_type == "bar"

    def test_publication_stats_field_builds_horizontal_bar(self):
        data = {
            "grouped_by": "field",
            "groups": [
                {"field": "CS", "papers": 200},
                {"field": "EE", "papers": 150},
            ],
        }
        result = build_chart_for_tool("get_publication_stats", data)
        assert result is not None
        assert result.chart.chart_type == "bar"
        assert result.chart.layout == "horizontal"  # type: ignore[attr-defined]

    def test_publication_stats_empty_groups(self):
        result = build_chart_for_tool("get_publication_stats", {"groups": []})
        assert result is None

    def test_chart_event_is_serializable(self):
        """ChartEvent.model_dump() must be JSON-serializable for SSE output."""
        import json
        data = {
            "topic": "AI",
            "trend": [{"year": 2022, "papers": 10}],
        }
        result = build_chart_for_tool("get_research_trends", data)
        assert result is not None
        payload = result.model_dump()
        # Should not raise
        json_str = json.dumps(payload)
        assert "line" in json_str
