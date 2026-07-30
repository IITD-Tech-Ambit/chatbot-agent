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
    """get_faculty_profile resolves a name via the Directory search (top result)
    and assembles the profile from the /profile + /research-summary + IP-search
    endpoints, so the HTTP calls are mocked with respx (no live infra)."""

    @staticmethod
    def _mock_directory(respx, httpx, *, faculties, summary=None, ip=None):
        respx.get(url__regex=r".*/api/directory/search.*").mock(
            return_value=httpx.Response(200, json={"data": {"faculties": faculties}})
        )
        respx.get(url__regex=r".*/api/directory/faculty/[^/]+/profile$").mock(
            return_value=httpx.Response(200, json={"data": (faculties[0] if faculties else {})})
        )
        respx.get(url__regex=r".*/api/directory/faculty/[^/]+/research-summary.*").mock(
            return_value=httpx.Response(200, json={"data": summary or {"stats": {"totalPapers": 0}, "timeline": []}})
        )
        respx.post(url__regex=r".*/api/v1/ip/search.*").mock(
            return_value=httpx.Response(200, json=ip or {"results": [], "pagination": {"total": 0}})
        )

    @pytest.mark.asyncio
    async def test_valid_name(self, tool_deps):
        import httpx
        import respx
        from agent.tools.faculty_profile import build_tool

        faculty = {
            "name": "Prof Amit Kumar", "email": "amitkumar@iitd.ac.in",
            "hIndex": 25, "citationCount": 3000, "research_areas": ["Machine Learning"],
            "scopusId": "SCOP001", "googleScholarId": "GS1",
            "designation": "Professor", "workingFromYear": 2018,
            "department": {"name": "Computer Science", "code": "cse", "category": "Department"},
        }
        summary = {"stats": {"totalPapers": 42, "totalYears": 6}, "timeline": [
            {"year": 2024, "count": 2, "papers": [
                {"title": "Paper A", "type": "Article", "citations": 10, "link": "a"},
                {"title": "Paper B", "type": "Article", "citations": 5, "link": "b"},
            ]},
        ]}
        tool = build_tool(tool_deps)
        with respx.mock:
            self._mock_directory(respx, httpx, faculties=[faculty], summary=summary)
            result = await tool.ainvoke({"name": "Amit Kumar"})
        data = json.loads(result)
        assert data["resolved_from_query"] == "Amit Kumar"
        assert data["profile"]["name"] == "Prof Amit Kumar"
        assert data["profile"]["email"] == "amitkumar@iitd.ac.in"
        assert data["profile"]["kerberos"] == "amitkumar"
        assert data["profile"]["profile_url"] == "/faculty/amitkumar"
        assert data["profile"]["h_index"] == 25
        assert data["papers"]["total"] == 42
        assert len(data["papers"]["latest"]) == 2
        assert data["papers"]["latest"][0]["title"] == "Paper A"

    @pytest.mark.asyncio
    async def test_unknown_name(self, tool_deps):
        import httpx
        import respx
        from agent.tools.faculty_profile import build_tool

        tool = build_tool(tool_deps)
        with respx.mock:
            self._mock_directory(respx, httpx, faculties=[])
            result = await tool.ainvoke({"name": "Nonexistent Person"})
        data = json.loads(result)
        assert "error" in data
        assert not data.get("profile")

    @pytest.mark.asyncio
    async def test_blank_name(self, tool_deps):
        from agent.tools.faculty_profile import build_tool

        tool = build_tool(tool_deps)
        result = await tool.ainvoke({"name": "   "})
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
