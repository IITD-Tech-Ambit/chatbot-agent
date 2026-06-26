"""Unit tests for IR metrics."""

from __future__ import annotations

import pytest

from eval.metrics import (
    average_metrics,
    compute_all,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_perfect_rank1():
    retrieved = ["a", "b", "c"]
    relevant = {"a": 3}
    m = compute_all(retrieved, relevant)
    assert m["mrr"] == 1.0
    assert m["precision_1"] == 1.0
    assert m["recall_50"] == 1.0


def test_miss_returns_zero_mrr():
    retrieved = ["x", "y"]
    relevant = {"a": 3}
    assert mrr(retrieved, relevant) == 0.0
    assert recall_at_k(retrieved, relevant, 50) == 0.0


def test_graded_ndcg():
    retrieved = ["a", "b", "c"]
    relevant = {"a": 3, "b": 2, "c": 1}
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0, abs=0.01)


def test_precision_denominator_is_k():
    retrieved = ["a"]
    relevant = {"a": 3}
    assert precision_at_k(retrieved, relevant, 5) == 0.2


def test_average_metrics_skips_null_recall():
    per_query = [
        {"recall_50": 1.0, "precision_1": 1.0, "precision_5": 0.2,
         "precision_10": 0.1, "ndcg_10": 1.0, "mrr": 1.0, "total_relevant": 1},
        {"recall_50": None, "precision_1": 0.0, "precision_5": 0.0,
         "precision_10": 0.0, "ndcg_10": 0.0, "mrr": 0.0, "total_relevant": 0},
    ]
    avg = average_metrics(per_query)
    assert avg["queries_evaluated"] == 2
    assert avg["mrr"] == 0.5
