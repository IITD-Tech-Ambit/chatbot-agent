"""Tests for tools with mocked repositories (no real DB/OpenSearch)."""

import json

import pytest

from tests.conftest import FakeFacultyRepo, FakeResearchRepo, make_tool_deps


class FakeRetriever:
    async def retrieve(self, query, top_k=None, abstract_max_chars=150):
        return [
            {
                "index": 1,
                "id": "paper1",
                "title": "Deep Learning for NLP",
                "abstract": "Explores deep learning...",
                "authors": ["Amit Kumar"],
                "publication_year": 2023,
                "document_type": "Article",
                "field_associated": "Computer Science",
                "citation_count": 50,
                "link": "https://example.com",
            }
        ]


@pytest.fixture
def tool_deps():
    return make_tool_deps(
        faculty_repo=FakeFacultyRepo(),
        research_repo=FakeResearchRepo(),
        retriever=FakeRetriever(),
    )


class TestSearchPapers:
    @pytest.mark.asyncio
    async def test_returns_papers(self, tool_deps):
        from agent.tools.search_papers import build_tool

        tool = build_tool(tool_deps)
        result = await tool.ainvoke({"query": "machine learning"})
        data = json.loads(result)
        assert "papers" in data
        assert len(data["papers"]) >= 1
        assert data["papers"][0]["title"] == "Deep Learning for NLP"

    @pytest.mark.asyncio
    async def test_year_filter(self, tool_deps):
        from agent.tools.search_papers import build_tool

        tool = build_tool(tool_deps)
        result = await tool.ainvoke({"query": "ML", "year_from": 2025})
        data = json.loads(result)
        assert len(data["papers"]) == 0


class TestFindFacultyForTopic:
    @pytest.mark.asyncio
    async def test_returns_faculty(self, tool_deps):
        import httpx
        import respx

        from agent.tools.find_faculty import build_tool

        tool = build_tool(tool_deps)
        mock_response = {
            "departments": [
                {
                    "name": "Computer Science",
                    "faculty": [
                        {"author_id": "EXP001", "name": "Amit Kumar", "paper_count": 10, "relevance_score": 0.9}
                    ],
                }
            ],
            "total_matching_papers": 42,
        }

        with respx.mock:
            respx.get(url__regex=r".*/api/v1/search/faculty-for-query.*").mock(
                return_value=httpx.Response(200, json=mock_response)
            )
            result = await tool.ainvoke({"topic": "machine learning"})

        data = json.loads(result)
        assert data["topic"] == "machine learning"
        assert len(data["faculty"]) >= 1


class TestGetFacultyProfile:
    @pytest.mark.asyncio
    async def test_valid_name(self, tool_deps):
        from agent.tools.faculty_profile import build_tool

        tool = build_tool(tool_deps)
        result = await tool.ainvoke({"name": "Amit Kumar"})
        data = json.loads(result)
        assert "profile" in data
        assert data["profile"]["name"] == "Prof. Amit Kumar"
        assert data["profile"]["email"] == "amitkumar@iitd.ac.in"

    @pytest.mark.asyncio
    async def test_meta_name_rejected(self, tool_deps):
        from agent.tools.faculty_profile import build_tool

        tool = build_tool(tool_deps)
        result = await tool.ainvoke({"name": "yourself"})
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_unknown_name(self, tool_deps):
        from agent.tools.faculty_profile import build_tool

        tool = build_tool(tool_deps)
        result = await tool.ainvoke({"name": "Nonexistent Person"})
        data = json.loads(result)
        assert "error" in data


class TestGetPublicationStats:
    @pytest.mark.asyncio
    async def test_global_stats(self, tool_deps):
        from agent.tools.publication_stats import build_tool

        tool = build_tool(tool_deps)
        result = await tool.ainvoke({})
        data = json.loads(result)
        assert "total_papers" in data
        assert data["grouped_by"] == "department"
        assert isinstance(data["groups"], list)

    @pytest.mark.asyncio
    async def test_department_stats(self, tool_deps):
        from agent.tools.publication_stats import build_tool

        tool = build_tool(tool_deps)
        result = await tool.ainvoke({"department": "Computer Science"})
        data = json.loads(result)
        assert "department" in data
        assert "Computer Science" in data["department"]


class TestCompareFaculty:
    @pytest.mark.asyncio
    async def test_compare_known(self, tool_deps):
        from agent.tools.compare_faculty import build_tool

        tool = build_tool(tool_deps)
        result = await tool.ainvoke({"name_a": "Amit Kumar", "name_b": "Amit Kumar"})
        data = json.loads(result)
        assert "comparison" in data

    @pytest.mark.asyncio
    async def test_compare_unknown(self, tool_deps):
        from agent.tools.compare_faculty import build_tool

        tool = build_tool(tool_deps)
        result = await tool.ainvoke({"name_a": "Nobody Known", "name_b": "Also Nobody"})
        data = json.loads(result)
        assert "error" in data


class TestResearchTrends:
    @pytest.mark.asyncio
    async def test_trends(self, tool_deps):
        from agent.tools.research_trends import build_tool

        tool = build_tool(tool_deps)
        result = await tool.ainvoke({"topic": "Computer Science"})
        data = json.loads(result)
        assert "trend" in data
        assert len(data["trend"]) >= 1


class TestSimilarPapers:
    @pytest.mark.asyncio
    async def test_similar(self, tool_deps):
        from agent.tools.similar_papers import build_tool

        tool = build_tool(tool_deps)
        result = await tool.ainvoke({"title": "Some other paper", "abstract": "about NLP"})
        data = json.loads(result)
        assert "similar_papers" in data
