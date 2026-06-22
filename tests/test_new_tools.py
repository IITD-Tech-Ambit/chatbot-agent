"""Unit tests for the 4 new tools: department_profile, list_departments,
faculty_by_expertise, interdisciplinary_papers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import FakeFacultyRepo, FakeResearchRepo, SAMPLE_FACULTY, SAMPLE_DEPARTMENTS


# ── Helpers ──

def _init_registry(fake_faculty_repo, fake_research_repo):
    """Patch the tool registry with fake repos and a fake config."""
    from unittest.mock import MagicMock
    from agent.tools import _registry

    fake_cfg = MagicMock()
    fake_cfg.TOKEN_CAP_DEPARTMENT_PROFILE = 2500
    fake_cfg.TOKEN_CAP_LIST_DEPARTMENTS = 3000
    fake_cfg.TOKEN_CAP_FACULTY_EXPERTISE = 2000
    fake_cfg.TOKEN_CAP_INTERDISCIPLINARY = 2000

    _registry._faculty_repo = fake_faculty_repo
    _registry._research_repo = fake_research_repo
    _registry._config = fake_cfg
    _registry._retriever = None  # not needed for most tests


# ── get_department_profile ──

class TestGetDepartmentProfile:
    @pytest.fixture(autouse=True)
    def setup(self, fake_faculty_repo, fake_research_repo):
        _init_registry(fake_faculty_repo, fake_research_repo)

    @pytest.mark.asyncio
    async def test_known_department_returns_profile(self):
        from agent.tools.department_profile import get_department_profile

        result_str = await get_department_profile.ainvoke({"department": "Computer Science"})
        data = json.loads(result_str)

        assert "department" in data
        assert data["department"]["name"] == "Computer Science and Engineering"
        assert "faculty_count" in data
        assert "top_faculty" in data
        assert "publication_stats" in data

    @pytest.mark.asyncio
    async def test_unknown_department_returns_error(self):
        from agent.tools.department_profile import get_department_profile

        result_str = await get_department_profile.ainvoke({"department": "Underwater Basket Weaving"})
        data = json.loads(result_str)

        assert "error" in data

    @pytest.mark.asyncio
    async def test_top_faculty_included(self):
        from agent.tools.department_profile import get_department_profile

        result_str = await get_department_profile.ainvoke({"department": "CSE"})
        data = json.loads(result_str)

        top = data.get("top_faculty", [])
        assert isinstance(top, list)

    @pytest.mark.asyncio
    async def test_output_within_token_cap(self):
        from agent.tools.department_profile import get_department_profile

        result_str = await get_department_profile.ainvoke({"department": "Computer Science"})
        assert len(result_str) <= 2500 + 5  # +5 for truncation suffix


# ── list_departments ──

class TestListDepartments:
    @pytest.fixture(autouse=True)
    def setup(self, fake_faculty_repo, fake_research_repo):
        _init_registry(fake_faculty_repo, fake_research_repo)

    @pytest.mark.asyncio
    async def test_no_filter_returns_all(self):
        from agent.tools.list_departments import list_departments

        result_str = await list_departments.ainvoke({})
        data = json.loads(result_str)

        assert data["total"] == len(SAMPLE_DEPARTMENTS)
        assert "departments" in data
        assert isinstance(data["departments"], dict)

    @pytest.mark.asyncio
    async def test_category_filter_school(self):
        from agent.tools.list_departments import list_departments

        result_str = await list_departments.ainvoke({"category": "School"})
        data = json.loads(result_str)

        depts = data["departments"]
        assert data["filter_category"] == "School"
        # Only School-category departments should appear
        for cat_name, items in depts.items():
            assert "School" in cat_name or cat_name == "School"

    @pytest.mark.asyncio
    async def test_grouped_structure(self):
        from agent.tools.list_departments import list_departments

        result_str = await list_departments.ainvoke({})
        data = json.loads(result_str)

        grouped = data["departments"]
        assert isinstance(grouped, dict)
        # At least one group should exist
        assert len(grouped) > 0

    @pytest.mark.asyncio
    async def test_output_within_token_cap(self):
        from agent.tools.list_departments import list_departments

        result_str = await list_departments.ainvoke({})
        assert len(result_str) <= 3000 + 5


# ── find_faculty_by_expertise ──

class TestFindFacultyByExpertise:
    @pytest.fixture(autouse=True)
    def setup(self, fake_faculty_repo, fake_research_repo):
        _init_registry(fake_faculty_repo, fake_research_repo)

    @pytest.mark.asyncio
    async def test_finds_faculty_with_matching_expertise(self):
        from agent.tools.faculty_by_expertise import find_faculty_by_expertise

        result_str = await find_faculty_by_expertise.ainvoke({"expertise": "Machine Learning"})
        data = json.loads(result_str)

        assert data["count"] > 0
        names = [f["name"] for f in data["faculty"]]
        assert any("Amit" in n for n in names)

    @pytest.mark.asyncio
    async def test_vlsi_expertise_match(self):
        from agent.tools.faculty_by_expertise import find_faculty_by_expertise

        result_str = await find_faculty_by_expertise.ainvoke({"expertise": "VLSI"})
        data = json.loads(result_str)

        assert data["count"] > 0
        assert any("Sunita" in f["name"] for f in data["faculty"])

    @pytest.mark.asyncio
    async def test_no_match_returns_empty_list(self):
        from agent.tools.faculty_by_expertise import find_faculty_by_expertise

        result_str = await find_faculty_by_expertise.ainvoke({"expertise": "Quantum Gravity"})
        data = json.loads(result_str)

        assert data["count"] == 0
        assert data["faculty"] == []
        assert "message" in data

    @pytest.mark.asyncio
    async def test_results_within_limit(self):
        from agent.tools.faculty_by_expertise import find_faculty_by_expertise

        result_str = await find_faculty_by_expertise.ainvoke({"expertise": "research", "limit": 1})
        data = json.loads(result_str)

        assert len(data["faculty"]) <= 1

    @pytest.mark.asyncio
    async def test_output_within_token_cap(self):
        from agent.tools.faculty_by_expertise import find_faculty_by_expertise

        result_str = await find_faculty_by_expertise.ainvoke({"expertise": "Machine Learning"})
        assert len(result_str) <= 2000 + 5


# ── find_interdisciplinary_papers ──

class TestFindInterdisciplinaryPapers:
    @pytest.fixture(autouse=True)
    def setup(self, fake_faculty_repo, fake_research_repo):
        from unittest.mock import MagicMock
        from agent.tools import _registry

        fake_cfg = MagicMock()
        fake_cfg.TOKEN_CAP_INTERDISCIPLINARY = 2000
        _registry._faculty_repo = fake_faculty_repo
        _registry._research_repo = fake_research_repo
        _registry._config = fake_cfg

        # Fake retriever that returns papers per field
        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock(return_value=[
            {"id": "paper1", "title": "Deep Learning for NLP"},
            {"id": "paper2", "title": "Renewable Energy"},
        ])
        _registry._retriever = fake_retriever

    @pytest.mark.asyncio
    async def test_two_fields_returns_papers(self):
        from agent.tools.interdisciplinary_papers import find_interdisciplinary_papers

        result_str = await find_interdisciplinary_papers.ainvoke({
            "fields": ["machine learning", "healthcare"]
        })
        data = json.loads(result_str)

        assert "papers" in data
        assert data["fields"] == ["machine learning", "healthcare"]

    @pytest.mark.asyncio
    async def test_requires_at_least_two_fields(self):
        from agent.tools.interdisciplinary_papers import find_interdisciplinary_papers
        import pydantic

        with pytest.raises((pydantic.ValidationError, Exception)):
            await find_interdisciplinary_papers.ainvoke({"fields": ["only one"]})

    @pytest.mark.asyncio
    async def test_year_filter_applied(self):
        from agent.tools.interdisciplinary_papers import find_interdisciplinary_papers

        result_str = await find_interdisciplinary_papers.ainvoke({
            "fields": ["machine learning", "energy"],
            "year_from": 2020,
            "year_to": 2024,
        })
        data = json.loads(result_str)

        assert data["year_range"]["from"] == 2020
        assert data["year_range"]["to"] == 2024

    @pytest.mark.asyncio
    async def test_limit_respected(self):
        from agent.tools.interdisciplinary_papers import find_interdisciplinary_papers

        result_str = await find_interdisciplinary_papers.ainvoke({
            "fields": ["machine learning", "energy"],
            "limit": 1,
        })
        data = json.loads(result_str)

        assert len(data["papers"]) <= 1

    @pytest.mark.asyncio
    async def test_year_from_gt_year_to_raises(self):
        from agent.tools.interdisciplinary_papers import find_interdisciplinary_papers
        import pydantic

        with pytest.raises((pydantic.ValidationError, Exception)):
            await find_interdisciplinary_papers.ainvoke({
                "fields": ["ml", "health"],
                "year_from": 2024,
                "year_to": 2020,
            })
