"""Tests for structured fast-path routing."""

from __future__ import annotations

import pytest

from agent.routing.structured import match_structured, execute_structured, RouteMatch
from tests.conftest import FakeFacultyRepo, FakeResearchRepo


class TestMatchStructured:
    """Pattern matching tests — no I/O."""

    @pytest.mark.parametrize("msg,expected_handler", [
        ("h-index of Prof. Amit Kumar", "get_h_index"),
        ("h index for Sunita Singh", "get_h_index"),
        ("h.index of rajeev", "get_h_index"),
        ("citations of Prof. Kumar", "get_citations"),
        ("citation count for Amit", "get_citations"),
        ("faculty in Computer Science department", "get_faculty_by_dept"),
        ("professors from EE dept", "get_faculty_by_dept"),
        ("papers by Amit Kumar", "get_papers_by_author"),
        ("paper by Prof. Singh", "get_papers_by_author"),
    ])
    def test_known_pattern_matches(self, msg: str, expected_handler: str):
        match = match_structured(msg)
        assert match is not None
        assert match.handler == expected_handler

    @pytest.mark.parametrize("msg,expected_capture", [
        ("h-index of Prof. Amit Kumar", "Prof. Amit Kumar"),
        ("papers by Amit Kumar", "Amit Kumar"),
        ("citations of Singh", "Singh"),
    ])
    def test_capture_group_extracted(self, msg: str, expected_capture: str):
        match = match_structured(msg)
        assert match is not None
        assert match.capture == expected_capture

    @pytest.mark.parametrize("msg", [
        "list all departments",
        "list departments at IIT Delhi",
        "show me all departments",
        "what are the departments",
        "show all departments at IIT",
    ])
    def test_list_departments_pattern(self, msg: str):
        match = match_structured(msg)
        assert match is not None
        assert match.handler == "list_departments"

    @pytest.mark.parametrize("msg", [
        "What research is done on machine learning?",
        "Tell me about solar cells",
        "Compare Prof. A and Prof. B",
        "Which faculty have expertise in VLSI?",
    ])
    def test_non_matching_queries(self, msg: str):
        match = match_structured(msg)
        assert match is None


class TestExecuteStructured:
    """Handler execution tests with fake repos."""

    @pytest.fixture
    def faculty_repo(self):
        return FakeFacultyRepo()

    @pytest.fixture
    def research_repo(self):
        return FakeResearchRepo()

    @pytest.mark.asyncio
    async def test_get_h_index_found(self, faculty_repo, research_repo):
        route = RouteMatch(handler="get_h_index", capture="Amit Kumar")
        result = await execute_structured(route, faculty_repo, research_repo)
        assert "text" in result
        assert "25" in result["text"]  # h_index from SAMPLE_FACULTY
        assert "Amit" in result["text"]

    @pytest.mark.asyncio
    async def test_get_h_index_not_found(self, faculty_repo, research_repo):
        route = RouteMatch(handler="get_h_index", capture="Unknown Person XYZ")
        result = await execute_structured(route, faculty_repo, research_repo)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_citations_found(self, faculty_repo, research_repo):
        route = RouteMatch(handler="get_citations", capture="Amit Kumar")
        result = await execute_structured(route, faculty_repo, research_repo)
        assert "text" in result
        assert "3000" in result["text"] or "citation" in result["text"].lower()

    @pytest.mark.asyncio
    async def test_get_faculty_by_dept_found(self, faculty_repo, research_repo):
        route = RouteMatch(handler="get_faculty_by_dept", capture="Computer Science")
        result = await execute_structured(route, faculty_repo, research_repo)
        assert "text" in result
        assert "Computer Science" in result["text"]

    @pytest.mark.asyncio
    async def test_get_faculty_by_dept_not_found(self, faculty_repo, research_repo):
        route = RouteMatch(handler="get_faculty_by_dept", capture="Nonexistent Department")
        result = await execute_structured(route, faculty_repo, research_repo)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_papers_by_author(self, faculty_repo, research_repo):
        route = RouteMatch(handler="get_papers_by_author", capture="Amit Kumar")
        result = await execute_structured(route, faculty_repo, research_repo)
        assert "text" in result
        assert "publication" in result["text"].lower() or "indexed" in result["text"].lower()

    @pytest.mark.asyncio
    async def test_list_departments_handler(self, faculty_repo, research_repo):
        route = RouteMatch(handler="list_departments", capture="")
        result = await execute_structured(route, faculty_repo, research_repo)
        assert "text" in result
        text = result["text"]
        # Should list department categories
        assert "Department" in text or "IIT Delhi" in text

    @pytest.mark.asyncio
    async def test_unknown_handler_returns_error(self, faculty_repo, research_repo):
        route = RouteMatch(handler="nonexistent_handler", capture="test")
        result = await execute_structured(route, faculty_repo, research_repo)
        assert "error" in result
