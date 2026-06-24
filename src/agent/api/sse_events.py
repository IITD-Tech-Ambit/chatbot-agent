"""Typed Pydantic models for all SSE event payloads.

All SSE output flows through these models — never raw dicts.
Wire format is unchanged: _sse(event_name, model.model_dump()).
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel


class StatusEvent(BaseModel):
    text: str


class ThinkingEvent(BaseModel):
    """Streamed insight while agent processes — replaces raw tool-name exposure."""
    step: str
    detail: str | None = None


class TokenEvent(BaseModel):
    text: str


class DoneEvent(BaseModel):
    took_ms: int
    cached: bool = False


class ErrorEvent(BaseModel):
    message: str


class SourcePaper(BaseModel):
    index: int | None = None
    id: str = ""
    title: str
    authors: list[str] = []
    publication_year: int | None = None
    document_type: str | None = None
    field_associated: str | None = None
    citation_count: int = 0
    link: str | None = None
    kerberos: str | None = None
    faculty_name: str | None = None


# ── Chart payloads ──

class DataPoint(BaseModel):
    x: str | int | float
    y: int | float


class ChartSeries(BaseModel):
    label: str
    data: list[DataPoint]


class LineChartData(BaseModel):
    chart_type: Literal["line"] = "line"
    title: str
    x_label: str = ""
    y_label: str = ""
    series: list[ChartSeries]


class BarChartData(BaseModel):
    chart_type: Literal["bar"] = "bar"
    title: str
    x_label: str = ""
    y_label: str = ""
    layout: Literal["horizontal", "vertical"] = "vertical"
    categories: list[str]
    series: list[dict[str, Any]]


class PieChartData(BaseModel):
    chart_type: Literal["pie"] = "pie"
    title: str
    slices: list[dict[str, Any]]


ChartPayload = Union[LineChartData, BarChartData, PieChartData]


class ChartEvent(BaseModel):
    tool_name: str
    chart: ChartPayload
