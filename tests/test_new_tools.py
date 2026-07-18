"""Unit tests for department_profile, list_departments,
faculty_by_expertise, interdisciplinary_papers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from tests.conftest import SAMPLE_DEPARTMENTS, make_tool_deps


@pytest.fixture
def base_deps(fake_faculty_repo, fake_research_repo):
    return make_tool_deps(faculty_repo=fake_faculty_repo, research_repo=fake_research_repo)


class TestGetDepartmentProfile:
    @pytest.mark.asyncio
    async def test_known_department_returns_profile(self, base_deps):
        from agent.tools.department_profile import build_tool

        tool = build_tool(base_deps)
        result_str = await tool.ainvoke({"department": "Computer Science"})
        data = json.loads(result_str)

        assert "department" in data
        assert data["department"]["name"] == "Computer Science and Engineering"
        assert "faculty_count" in data
        assert "top_faculty" in data

    @pytest.mark.asyncio
    async def test_unknown_department_returns_error(self, base_deps):
        from agent.tools.department_profile import build_tool

        tool = build_tool(base_deps)
        result_str = await tool.ainvoke({"department": "Underwater Basket Weaving"})
        data = json.loads(result_str)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_code_lookup(self, base_deps):
        from agent.tools.department_profile import build_tool

        tool = build_tool(base_deps)
        result_str = await tool.ainvoke({"department": "CSE"})
        data = json.loads(result_str)
        assert "department" in data

    @pytest.mark.asyncio
    async def test_output_within_token_cap(self, base_deps):
        from agent.tools.department_profile import build_tool

        tool = build_tool(base_deps)
        result_str = await tool.ainvoke({"department": "Computer Science"})
        assert len(result_str) <= 2500 + 5


class TestListDepartments:
    @pytest.mark.asyncio
    async def test_no_filter_returns_all(self, base_deps):
        from agent.tools.list_departments import build_tool

        tool = build_tool(base_deps)
        result_str = await tool.ainvoke({})
        data = json.loads(result_str)

        assert data["total"] == len(SAMPLE_DEPARTMENTS)
        assert "departments" in data
        assert isinstance(data["departments"], dict)

    @pytest.mark.asyncio
    async def test_category_filter_school(self, base_deps):
        from agent.tools.list_departments import build_tool

        tool = build_tool(base_deps)
        result_str = await tool.ainvoke({"category": "School"})
        data = json.loads(result_str)

        depts = data["departments"]
        assert data["filter_category"] == "School"
        for cat_name, items in depts.items():
            assert "School" in cat_name or cat_name == "School"

    @pytest.mark.asyncio
    async def test_grouped_structure(self, base_deps):
        from agent.tools.list_departments import build_tool

        tool = build_tool(base_deps)
        result_str = await tool.ainvoke({})
        data = json.loads(result_str)

        grouped = data["departments"]
        assert isinstance(grouped, dict)
        assert len(grouped) > 0

    @pytest.mark.asyncio
    async def test_output_within_token_cap(self, base_deps):
        from agent.tools.list_departments import build_tool

        tool = build_tool(base_deps)
        result_str = await tool.ainvoke({})
        assert len(result_str) <= 3000 + 5


class TestFindFacultyByExpertise:
    @pytest.mark.asyncio
    async def test_finds_faculty_with_matching_expertise(self, base_deps):
        from agent.tools.faculty_by_expertise import build_tool

        tool = build_tool(base_deps)
        result_str = await tool.ainvoke({"expertise": "Machine Learning"})
        data = json.loads(result_str)

        assert data["count"] > 0
        names = [f["name"] for f in data["faculty"]]
        assert any("Amit" in n for n in names)

    @pytest.mark.asyncio
    async def test_vlsi_expertise_match(self, base_deps):
        from agent.tools.faculty_by_expertise import build_tool

        tool = build_tool(base_deps)
        result_str = await tool.ainvoke({"expertise": "VLSI"})
        data = json.loads(result_str)

        assert data["count"] > 0
        assert any("Sunita" in f["name"] for f in data["faculty"])

    @pytest.mark.asyncio
    async def test_no_match_returns_empty_list(self, base_deps):
        from agent.tools.faculty_by_expertise import build_tool

        tool = build_tool(base_deps)
        result_str = await tool.ainvoke({"expertise": "Quantum Gravity"})
        data = json.loads(result_str)

        assert data["count"] == 0
        assert data["faculty"] == []
        assert "message" in data

    @pytest.mark.asyncio
    async def test_results_within_limit(self, base_deps):
        from agent.tools.faculty_by_expertise import build_tool

        tool = build_tool(base_deps)
        result_str = await tool.ainvoke({"expertise": "research", "limit": 1})
        data = json.loads(result_str)

        assert len(data["faculty"]) <= 1

    @pytest.mark.asyncio
    async def test_output_within_token_cap(self, base_deps):
        from agent.tools.faculty_by_expertise import build_tool

        tool = build_tool(base_deps)
        result_str = await tool.ainvoke({"expertise": "Machine Learning"})
        assert len(result_str) <= 2000 + 5


class TestFindInterdisciplinaryPapers:
    @pytest.fixture
    def interdisciplinary_tool(self, fake_faculty_repo, fake_research_repo):
        from agent.tools.interdisciplinary_papers import build_tool

        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock(return_value=[
            {"id": "paper1", "title": "Deep Learning for NLP"},
            {"id": "paper2", "title": "Renewable Energy"},
        ])
        deps = make_tool_deps(
            faculty_repo=fake_faculty_repo,
            research_repo=fake_research_repo,
            retriever=fake_retriever,
        )
        return build_tool(deps)

    @pytest.mark.asyncio
    async def test_two_fields_returns_papers(self, interdisciplinary_tool):
        result_str = await interdisciplinary_tool.ainvoke({
            "fields": ["machine learning", "healthcare"]
        })
        data = json.loads(result_str)

        assert "papers" in data
        assert data["fields"] == ["machine learning", "healthcare"]

    @pytest.mark.asyncio
    async def test_requires_at_least_two_fields(self, interdisciplinary_tool):
        import pydantic

        with pytest.raises((pydantic.ValidationError, Exception)):
            await interdisciplinary_tool.ainvoke({"fields": ["only one"]})

    @pytest.mark.asyncio
    async def test_year_filter_applied(self, interdisciplinary_tool):
        result_str = await interdisciplinary_tool.ainvoke({
            "fields": ["machine learning", "energy"],
            "year_from": 2020,
            "year_to": 2024,
        })
        data = json.loads(result_str)

        assert data["year_range"]["from"] == 2020
        assert data["year_range"]["to"] == 2024

    @pytest.mark.asyncio
    async def test_limit_respected(self, interdisciplinary_tool):
        result_str = await interdisciplinary_tool.ainvoke({
            "fields": ["machine learning", "energy"],
            "limit": 1,
        })
        data = json.loads(result_str)

        assert len(data["papers"]) <= 1

    @pytest.mark.asyncio
    async def test_year_from_gt_year_to_raises(self, interdisciplinary_tool):
        import pydantic

        with pytest.raises((pydantic.ValidationError, Exception)):
            await interdisciplinary_tool.ainvoke({
                "fields": ["ml", "health"],
                "year_from": 2024,
                "year_to": 2020,
            })
