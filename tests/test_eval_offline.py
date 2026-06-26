"""Offline evaluation harness tests using mock retriever."""

from __future__ import annotations

import pytest

from eval.answer_quality import evaluate_answer, faithfulness_score, relevance_score
from eval.clients import MockRetriever
from eval.fixtures import load_corpus, load_golden_comprehensive
from eval.multi_hop import generate_edge_case_queries, generate_multi_hop_queries
from eval.retrieval_runner import run_retrieval_eval


@pytest.fixture
def corpus():
    return load_corpus()


@pytest.fixture
def mock_retriever(corpus):
    return MockRetriever(corpus["documents"], top_k=50)


@pytest.mark.asyncio
async def test_mock_retriever_exact_title_match(mock_retriever, corpus):
    doc = corpus["documents"][0]
    words = doc["title"].split()[:5]
    query = " ".join(words)
    results = await mock_retriever.retrieve(query)
    ids = [r["id"] for r in results]
    assert doc["mongo_id"] in ids


@pytest.mark.asyncio
async def test_comprehensive_golden_offline(mock_retriever):
    golden = load_golden_comprehensive()
    # Subset for speed
    subset = {**golden, "queries": golden["queries"][:20]}
    report = await run_retrieval_eval(mock_retriever, subset, label="comprehensive_subset")
    assert report["queries_evaluated"] == 20
    assert "average" in report
    assert report["average"]["mrr"] is not None


@pytest.mark.asyncio
async def test_edge_empty_query(mock_retriever):
    results = await mock_retriever.retrieve("   ")
    assert results == []


def test_multi_hop_generation(corpus):
    queries = generate_multi_hop_queries(corpus, count=5)
    assert len(queries) == 5
    assert all("relevant" in q for q in queries)


def test_edge_case_generation(corpus):
    edges = generate_edge_case_queries(corpus)
    assert any(q["type"] == "edge_ambiguous" for q in edges)


def test_answer_quality_faithful():
    sources = [{"title": "Solar Cell Efficiency", "abstract": "photovoltaic conversion efficiency improved"}]
    answer = "Research shows photovoltaic conversion efficiency improved in solar cells."
    score = faithfulness_score(answer, sources)
    assert score > 0.5


def test_answer_quality_relevance():
    assert relevance_score("machine learning neural networks", "neural networks for machine learning tasks") > 0.3


def test_hallucination_detection():
    sources = [{"title": "Paper A", "abstract": "published in 2020"}]
    result = evaluate_answer(
        "papers from 2020",
        "A groundbreaking paper from 1999 with 500 citations by John Smith.",
        sources,
    )
    assert result["hallucination_rate"] > 0
