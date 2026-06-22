"""Shared test fixtures: fake LLMs, repositories, Redis, and FastAPI test client."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage


# ── Fake LLM (returns deterministic tool calls or text) ──

class FakeToolCallingLLM:
    """Simulates a ChatModel that returns tool_calls on first invoke, text on second."""

    def __init__(self, tool_calls: list[dict] | None = None, answer: str = "Test answer."):
        self._tool_calls = tool_calls
        self._answer = answer
        self._call_count = 0
        self.tags: list[str] = []

    def bind_tools(self, tools: list) -> "FakeToolCallingLLM":
        return self

    async def ainvoke(self, messages: list, **kwargs) -> AIMessage:
        self._call_count += 1
        if self._call_count == 1 and self._tool_calls:
            return AIMessage(content="", tool_calls=self._tool_calls)
        return AIMessage(content=self._answer)

    def with_config(self, **kwargs) -> "FakeToolCallingLLM":
        if "tags" in kwargs:
            self.tags = kwargs["tags"]
        return self


class FakeNoToolLLM:
    """LLM that never returns tool_calls (tests the force-tool fallback)."""

    def __init__(self, answer: str = "I don't know."):
        self._answer = answer
        self.tags: list[str] = []

    def bind_tools(self, tools: list) -> "FakeNoToolLLM":
        return self

    async def ainvoke(self, messages: list, **kwargs) -> AIMessage:
        return AIMessage(content=self._answer)

    def with_config(self, **kwargs) -> "FakeNoToolLLM":
        if "tags" in kwargs:
            self.tags = kwargs["tags"]
        return self


# ── Fake Redis ──

class FakeRedis:
    def __init__(self):
        self._store: dict[str, str] = {}
        self._ttls: dict[str, int] = {}
        self._counters: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value
        self._ttls[key] = ttl

    async def incr(self, key: str) -> int:
        self._counters[key] = self._counters.get(key, 0) + 1
        return self._counters[key]

    async def expire(self, key: str, seconds: int) -> None:
        self._ttls[key] = seconds

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


# ── Sample data ──

SAMPLE_FACULTY = {
    "_id": "fac1",
    "expert_id": "EXP001",
    "title": "Prof.",
    "firstName": "Amit",
    "lastName": "Kumar",
    "email": "amitkumar@iitd.ac.in",
    "designation": "Professor",
    "department": {"_id": "dept1", "name": "Computer Science"},
    "expertise": ["Machine Learning", "Deep Learning", "Computer Vision"],
    "brief_expertise": ["ML", "DL"],
    "subjects": ["AI", "Data Science"],
    "h_index": 25,
    "citation_count": 3000,
    "scopus_id": ["SCOP001"],
}

SAMPLE_FACULTY_2 = {
    "_id": "fac2",
    "expert_id": "EXP002",
    "title": "Prof.",
    "firstName": "Sunita",
    "lastName": "Singh",
    "email": "sunitasingh@iitd.ac.in",
    "designation": "Associate Professor",
    "department": {"_id": "dept1", "name": "Computer Science"},
    "expertise": ["VLSI Design", "Embedded Systems"],
    "brief_expertise": ["VLSI", "Embedded"],
    "subjects": ["Electronics"],
    "h_index": 18,
    "citation_count": 1500,
    "scopus_id": ["SCOP002"],
}

SAMPLE_PAPER = {
    "_id": "paper1",
    "title": "Deep Learning for NLP",
    "abstract": "This paper explores deep learning approaches for NLP tasks.",
    "authors": [{"author_name": "Amit Kumar", "author_id": "SCOP001"}],
    "publication_year": 2023,
    "document_type": "Article",
    "field_associated": "Computer Science",
    "citation_count": 50,
    "link": "https://example.com/paper1",
}

SAMPLE_PAPER_2 = {
    "_id": "paper2",
    "title": "Renewable Energy Research",
    "abstract": "Solar cell efficiency improvements.",
    "authors": [{"author_name": "Sunita Singh", "author_id": "SCOP002"}],
    "publication_year": 2022,
    "document_type": "Article",
    "field_associated": "Electrical Engineering",
    "citation_count": 30,
    "link": "https://example.com/paper2",
}

SAMPLE_DEPARTMENTS = [
    {"_id": "dept1", "name": "Computer Science and Engineering", "code": "CSE", "category": "Department"},
    {"_id": "dept2", "name": "Electrical Engineering", "code": "EE", "category": "Department"},
    {"_id": "dept3", "name": "School of Biological Sciences", "code": "SBS", "category": "School"},
    {"_id": "dept4", "name": "Centre for Applied Research in Electronics", "code": "CARE", "category": "Centre"},
]


# ── Fake repositories ──

class FakeFacultyRepo:
    def __init__(self):
        self.faculty = [SAMPLE_FACULTY, SAMPLE_FACULTY_2]
        self.departments = SAMPLE_DEPARTMENTS

    async def text_search(self, query: str, limit: int = 5) -> list[dict]:
        return [
            f for f in self.faculty
            if query.lower() in f"{f['firstName']} {f['lastName']}".lower()
        ][:limit]

    async def regex_search(self, tokens: list[str], limit: int = 5) -> list[dict]:
        return await self.text_search(" ".join(tokens), limit)

    async def find_by_expert_ids(self, ids: list[str]) -> list[dict]:
        return [f for f in self.faculty if f.get("expert_id") in ids]

    async def find_department(self, name: str) -> dict | None:
        for d in self.departments:
            if name.lower() in d["name"].lower() or name.lower() == d["code"].lower():
                return d
        return None

    async def find_faculty_by_department_id(self, dept_id) -> list[dict]:
        return [
            {"email": f["email"], "scopus_id": f.get("scopus_id", [])}
            for f in self.faculty
            if str((f.get("department") or {}).get("_id", "")) == str(dept_id)
        ]

    async def find_top_faculty_by_department(self, department_name: str, limit: int = 10) -> list[dict]:
        dept = await self.find_department(department_name)
        if not dept:
            return []
        return sorted(self.faculty, key=lambda f: f.get("h_index", 0) or 0, reverse=True)[:limit]

    async def list_all_departments(self, category: str | None = None) -> list[dict]:
        if category:
            return [d for d in self.departments if d.get("category", "").lower() == category.lower()]
        return self.departments

    async def find_faculty_by_expertise(self, expertise_terms: list[str], limit: int = 15) -> list[dict]:
        result = []
        for f in self.faculty:
            expertise_str = " ".join((f.get("expertise") or []) + (f.get("brief_expertise") or [])).lower()
            if any(t.lower() in expertise_str for t in expertise_terms):
                result.append(f)
        return sorted(result, key=lambda f: f.get("h_index", 0) or 0, reverse=True)[:limit]

    async def find_top_faculty_global(
        self,
        sort_by: str = "h_index",
        limit: int = 10,
        department_name: str | None = None,
    ) -> list[dict]:
        candidates = self.faculty
        if department_name:
            dept = await self.find_department(department_name)
            if not dept:
                return []
            dept_id = str(dept["_id"])
            candidates = [
                f for f in candidates
                if str((f.get("department") or {}).get("_id", "")) == dept_id
            ]
        return sorted(candidates, key=lambda f: f.get(sort_by, 0) or 0, reverse=True)[:limit]

    async def count_all_faculty(self, department_name: str | None = None) -> int:
        if department_name:
            dept = await self.find_department(department_name)
            if not dept:
                return 0
            dept_id = str(dept["_id"])
            return sum(
                1 for f in self.faculty
                if str((f.get("department") or {}).get("_id", "")) == dept_id
            )
        return len(self.faculty)

    async def get_kerberos_to_dept_map(self) -> dict[str, str]:
        result = {}
        for f in self.faculty:
            kerberos = (f.get("email") or "").split("@")[0].lower().strip()
            dept = f.get("department") or {}
            dept_name = dept.get("name", "") if isinstance(dept, dict) else ""
            if kerberos and dept_name:
                result[kerberos] = dept_name
        return result


class FakeResearchRepo:
    def __init__(self):
        self.papers = [SAMPLE_PAPER, SAMPLE_PAPER_2]

    async def find_by_ids(self, ids: list[str], fields: dict | None = None) -> list[dict]:
        return [p for p in self.papers if str(p["_id"]) in ids]

    async def count_documents(self, match: dict) -> int:
        return len(self.papers)

    async def aggregate(self, pipeline: list[dict]) -> list[dict]:
        return [{"_id": 2023, "count": 10}]

    async def find_top_cited(self, match: dict, limit: int = 5) -> list[dict]:
        return self.papers[:limit]

    async def faculty_publication_stats(self, match: dict) -> dict | None:
        return {
            "total_papers": 42,
            "papers_by_recent_year": [{"year": 2023, "count": 10}, {"year": 2022, "count": 8}],
            "top_subject_areas": [{"subject": "CS", "papers": 30}],
            "top_fields": [{"field": "Computer Science", "papers": 30}],
            "most_cited_papers": [{"title": "Deep Learning for NLP", "year": 2023, "citations": 50}],
        }

    async def department_stats(self, match: dict) -> dict:
        return {
            "total_papers": 100,
            "papers_by_recent_year": [{"year": 2023, "count": 20}, {"year": 2022, "count": 15}],
            "papers_by_type": [{"type": "Article", "count": 80}],
        }

    async def global_stats(self, base_match: dict, dimension: str, sort_field: str, limit: int = 25) -> tuple[int, list[dict]]:
        return 500, [{"_id": "Computer Science", "count": 100}]

    async def trend_by_ids(self, paper_ids: list[str], year_from: int | None, year_to: int | None) -> list[dict]:
        return [{"_id": 2021, "count": 5}, {"_id": 2022, "count": 8}, {"_id": 2023, "count": 12}]

    async def kerberos_counts_for_ids(self, paper_ids: list[str], base_match: dict) -> list[dict]:
        return [{"_id": "testfaculty", "count": 10}]

    async def find_interdisciplinary_papers(self, fields: list[str], limit: int = 10) -> list[dict]:
        return self.papers[:limit]

    async def papers_by_kerberos(self, base_match: dict) -> list[dict]:
        # Return fake kerberos-grouped counts for testing
        return [
            {"_id": "testfaculty", "count": 25},
            {"_id": "anshulk", "count": 15},
        ]


# ── Fixtures ──

@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def fake_faculty_repo():
    return FakeFacultyRepo()


@pytest.fixture
def fake_research_repo():
    return FakeResearchRepo()


@pytest.fixture
def fake_tool_llm():
    return FakeToolCallingLLM(
        tool_calls=[{
            "id": "call_1",
            "name": "search_papers",
            "args": {"query": "machine learning"},
        }],
        answer="Based on the search results, IIT Delhi has extensive ML research.",
    )


@pytest.fixture
def fake_no_tool_llm():
    return FakeNoToolLLM()


@pytest.fixture
def fake_answer_llm():
    return FakeToolCallingLLM(answer="Here is a detailed answer about research.")
