"""Tests for get_ip_stats + IpRepository.grouped_counts — department-filtered
classification breakdowns (the "most filed IPC domain in department X" case).
"""

from __future__ import annotations

import json

import pytest
from bson import ObjectId
from mongomock_motor import AsyncMongoMockClient

from agent.repositories.faculty_repo import FacultyRepository
from agent.repositories.ip_repo import IpRepository
from agent.services.ipc.cache import RedisIpcCache
from agent.services.ipc.service import IpcClassificationService
from agent.services.ipc.static_table import StaticIpcTable
from agent.tools.deps import ToolDeps
from agent.tools.get_ip_stats import build_tool as build_get_ip_stats

CHEM_DEPT_ID = ObjectId()
EE_DEPT_ID = ObjectId()


class _NoopIpcCache(RedisIpcCache):
    """Skips Redis entirely; static-table lookups are all these tests need."""

    def __init__(self) -> None:
        pass

    async def get(self, code: str) -> str | None:
        return None

    async def set(self, code: str, meaning: str) -> None:
        return None


@pytest.fixture
async def seeded_db():
    client = AsyncMongoMockClient()
    db = client["testdb"]
    await db["departments"].insert_many([
        {"_id": CHEM_DEPT_ID, "name": "Chemical Engineering", "code": "chemical", "category": "Department"},
        {"_id": EE_DEPT_ID, "name": "Electrical Engineering", "code": "ee", "category": "Department"},
    ])
    await db["ipmetadatas"].insert_many([
        {"department": CHEM_DEPT_ID, "classification": ["B01J0037000000", "B01J0020320000"], "publication_year": 2023},
        {"department": CHEM_DEPT_ID, "classification": ["B01J0035000000"], "publication_year": 2024},
        {"department": CHEM_DEPT_ID, "classification": ["H01M0004900000"], "publication_year": 2024},
        {"department": EE_DEPT_ID, "classification": ["H02J0007350000"], "publication_year": 2023},
    ])
    return db


@pytest.fixture
def ip_stats_deps(seeded_db):
    from agent.config import settings

    ip_repo = IpRepository(seeded_db)
    faculty_repo = FacultyRepository(seeded_db)
    ipc_service = IpcClassificationService(StaticIpcTable(), _NoopIpcCache(), wipo_client=None)
    return ToolDeps(
        retriever=None,
        faculty_repo=faculty_repo,
        research_repo=None,
        config=settings,
        ip_repo=ip_repo,
        ipc_service=ipc_service,
    )


class TestGroupedCountsClassification:
    @pytest.mark.asyncio
    async def test_department_match_ranks_classification_at_subclass_level(self, seeded_db):
        ip_repo = IpRepository(seeded_db)
        groups = await ip_repo.grouped_counts({"department": CHEM_DEPT_ID}, ["classification"])
        assert groups[0] == {"count": 3, "classification": "B01J"}
        assert {"count": 1, "classification": "H01M"} in groups
        assert all(g["department"] if "department" in g else True for g in groups)
        assert sum(g["count"] for g in groups) == 4

    @pytest.mark.asyncio
    async def test_department_filter_does_not_leak_other_departments(self, seeded_db):
        ip_repo = IpRepository(seeded_db)
        groups = await ip_repo.grouped_counts({"department": EE_DEPT_ID}, ["classification"])
        assert groups == [{"count": 1, "classification": "H02J"}]


class TestGetIpStatsTool:
    @pytest.mark.asyncio
    async def test_department_filter_combined_with_classification_group_by(self, ip_stats_deps):
        tool = build_get_ip_stats(ip_stats_deps)
        out = await tool.ainvoke({"department": "Chemical Engineering", "group_by": "classification"})
        data = json.loads(out)

        assert data["total"] == 3  # 3 documents (one has 2 classification codes)
        assert data["dimensions"] == ["classification"]
        top = data["groups"][0]
        assert top["count"] == 3
        assert top["classification_code"] == "B01J"
        assert top["classification"].startswith("B01J")

    @pytest.mark.asyncio
    async def test_misplaced_classification_word_in_prefix_arg_is_redirected_to_group_by(self, ip_stats_deps):
        """Models sometimes pass the dimension name into classification_prefix
        instead of group_by; that should still produce the classification
        breakdown rather than a doomed zero-result filter."""
        tool = build_get_ip_stats(ip_stats_deps)
        out = await tool.ainvoke({
            "department": "Chemical Engineering",
            "classification_prefix": "classification",
        })
        data = json.loads(out)

        assert data["total"] == 3
        assert "classification" in data["dimensions"]
        assert data["groups"], "expected a non-empty classification breakdown"

    @pytest.mark.asyncio
    async def test_real_classification_prefix_still_filters_normally(self, ip_stats_deps):
        tool = build_get_ip_stats(ip_stats_deps)
        out = await tool.ainvoke({"classification_prefix": "H01M"})
        data = json.loads(out)

        assert data["total"] == 1
        assert data["dimensions"] == ["year"]

    @pytest.mark.asyncio
    async def test_unknown_department_returns_error(self, ip_stats_deps):
        tool = build_get_ip_stats(ip_stats_deps)
        out = await tool.ainvoke({"department": "Underwater Basket Weaving", "group_by": "classification"})
        data = json.loads(out)
        assert "error" in data
        assert data["groups"] == []
