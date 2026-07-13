"""Tests for get_top_faculty tool and related structured routing fixes."""

from __future__ import annotations

import json

import pytest

from tests.conftest import FakeFacultyRepo, FakeResearchRepo, make_tool_deps
from agent.routing.structured import match_structured, execute_structured, RouteMatch


@pytest.fixture
def top_faculty_tool(fake_faculty_repo, fake_research_repo):
    from agent.tools.top_faculty import build_tool
    return build_tool(make_tool_deps(faculty_repo=fake_faculty_repo, research_repo=fake_research_repo))


class TestGetTopFaculty:
    @pytest.mark.asyncio
    async def test_returns_ranked_list_by_hindex(self, top_faculty_tool):
        result_str = await top_faculty_tool.ainvoke({"sort_by": "h_index", "limit": 10})
        data = json.loads(result_str)

        assert "faculty" in data
        assert data["count"] >= 2
        assert data["ranked_by"] == "H-Index"
        assert data["faculty"][0]["h_index"] == 25
        assert "Amit" in data["faculty"][0]["name"]

    @pytest.mark.asyncio
    async def test_returns_ranked_list_by_citations(self, top_faculty_tool):
        result_str = await top_faculty_tool.ainvoke({"sort_by": "citation_count"})
        data = json.loads(result_str)

        assert data["ranked_by"] == "Total Citations"
        assert data["faculty"][0]["citation_count"] == 3000

    @pytest.mark.asyncio
    async def test_emails_are_included(self, top_faculty_tool):
        result_str = await top_faculty_tool.ainvoke({})
        data = json.loads(result_str)

        for f in data["faculty"]:
            assert "email" in f
            assert "@" in f["email"]

    @pytest.mark.asyncio
    async def test_rank_field_sequential(self, top_faculty_tool):
        result_str = await top_faculty_tool.ainvoke({})
        data = json.loads(result_str)

        ranks = [f["rank"] for f in data["faculty"]]
        assert ranks == list(range(1, len(ranks) + 1))

    @pytest.mark.asyncio
    async def test_department_filter(self, top_faculty_tool):
        result_str = await top_faculty_tool.ainvoke({
            "department": "Computer Science",
        })
        data = json.loads(result_str)
        assert data["count"] >= 1
        assert data["department_filter"] == "Computer Science"

    @pytest.mark.asyncio
    async def test_unknown_department_returns_error(self, top_faculty_tool):
        result_str = await top_faculty_tool.ainvoke({
            "department": "Underwater Basket Weaving",
        })
        data = json.loads(result_str)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_limit_respected(self, top_faculty_tool):
        result_str = await top_faculty_tool.ainvoke({"limit": 1})
        data = json.loads(result_str)
        assert len(data["faculty"]) <= 1

    @pytest.mark.asyncio
    async def test_output_within_token_cap(self, top_faculty_tool):
        result_str = await top_faculty_tool.ainvoke({})
        assert len(result_str) <= 3000 + 5


class TestTopFacultyStructuredPatterns:
    @pytest.mark.parametrize("query,expected_handler", [
        ("top 10 professors by h-index", "top_faculty_hindex"),
        ("top professors by h index", "top_faculty_hindex"),
        ("who has highest h-index at IIT Delhi", "top_faculty_hindex"),
        ("best h-index faculty", "top_faculty_hindex"),
        ("top 5 researchers by citations", "top_faculty_citations"),
        ("most cited professors at IIT Delhi", "top_faculty_citations"),
        ("most cited faculty", "top_faculty_citations"),
        ("top faculty by citation count", "top_faculty_citations"),
        ("ranked by h-index", "top_faculty_hindex"),
    ])
    def test_pattern_matches(self, query: str, expected_handler: str):
        match = match_structured(query)
        assert match is not None, f"Expected match for: '{query}'"
        assert match.handler == expected_handler

    @pytest.mark.parametrize("query", [
        "What research is done at IIT Delhi?",
        "Tell me about Prof. Kumar",
        "papers on machine learning",
    ])
    def test_non_top_queries_dont_match(self, query: str):
        match = match_structured(query)
        if match:
            assert match.handler not in ("top_faculty_hindex", "top_faculty_citations")


class TestTopFacultyStructuredExecution:
    @pytest.fixture
    def faculty_repo(self):
        return FakeFacultyRepo()

    @pytest.fixture
    def research_repo(self):
        return FakeResearchRepo()

    @pytest.mark.asyncio
    async def test_top_hindex_returns_ranked_text(self, faculty_repo, research_repo):
        route = RouteMatch(handler="top_faculty_hindex", capture="")
        result = await execute_structured(route, faculty_repo, research_repo)
        assert "text" in result
        text = result["text"]
        assert "H-Index" in text
        assert "Amit" in text
        assert "@iitd.ac.in" in text

    @pytest.mark.asyncio
    async def test_top_citations_returns_ranked_text(self, faculty_repo, research_repo):
        route = RouteMatch(handler="top_faculty_citations", capture="")
        result = await execute_structured(route, faculty_repo, research_repo)
        assert "text" in result
        text = result["text"]
        assert "Citations" in text
        assert "3,000" in text or "3000" in text

    @pytest.mark.asyncio
    async def test_get_total_faculty_count_returns_number(self, faculty_repo, research_repo):
        route = RouteMatch(handler="get_total_faculty_count", capture="")
        result = await execute_structured(route, faculty_repo, research_repo)
        assert "text" in result
        assert "2" in result["text"]
        assert "faculty" in result["text"].lower()

    @pytest.mark.asyncio
    async def test_get_faculty_by_dept_returns_names_and_emails(self, faculty_repo, research_repo):
        route = RouteMatch(handler="get_faculty_by_dept", capture="Computer Science")
        result = await execute_structured(route, faculty_repo, research_repo)
        assert "text" in result
        text = result["text"]
        assert "@iitd.ac.in" in text
        assert "Amit" in text or "Sunita" in text


class TestFakeRepoNewMethods:
    @pytest.mark.asyncio
    async def test_find_top_faculty_global_sorted(self):
        repo = FakeFacultyRepo()
        results = await repo.find_top_faculty_global(sort_by="h_index", limit=10)
        assert results[0]["h_index"] == 25

    @pytest.mark.asyncio
    async def test_count_all_faculty(self):
        repo = FakeFacultyRepo()
        count = await repo.count_all_faculty()
        assert count == 2

    @pytest.mark.asyncio
    async def test_count_faculty_by_dept(self):
        repo = FakeFacultyRepo()
        count = await repo.count_all_faculty(department_name="Computer Science")
        assert count == 2

    @pytest.mark.asyncio
    async def test_count_faculty_unknown_dept(self):
        repo = FakeFacultyRepo()
        count = await repo.count_all_faculty(department_name="Underwater Basket Weaving")
        assert count == 0
