"""Information retrieval evaluation metrics (aligned with opensearch/tests/eval/metrics.mjs)."""

from __future__ import annotations

import math
from typing import Mapping


def recall_at_k(retrieved: list[str], relevant: Mapping[str, int], k: int) -> float | None:
    rel_set = set(relevant)
    if not rel_set:
        return None
    top_k = retrieved[:k]
    found = sum(1 for doc_id in top_k if doc_id in rel_set)
    return found / len(rel_set)


def precision_at_k(retrieved: list[str], relevant: Mapping[str, int], k: int) -> float:
    rel_set = set(relevant)
    top_k = retrieved[:k]
    found = sum(1 for doc_id in top_k if doc_id in rel_set)
    return found / k


def mrr(retrieved: list[str], relevant: Mapping[str, int]) -> float:
    rel_set = set(relevant)
    for i, doc_id in enumerate(retrieved):
        if doc_id in rel_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: Mapping[str, int], k: int) -> float:
    top_k = retrieved[:k]

    def _dcg(ids: list[str]) -> float:
        total = 0.0
        for i, doc_id in enumerate(ids):
            rel = relevant.get(doc_id, 0)
            total += (2**rel - 1) / math.log2(i + 2)
        return total

    dcg = _dcg(top_k)
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return 0.0 if idcg == 0 else dcg / idcg


def compute_all(retrieved: list[str], relevant: Mapping[str, int]) -> dict:
    return {
        "recall_50": recall_at_k(retrieved, relevant, 50),
        "precision_1": precision_at_k(retrieved, relevant, 1),
        "precision_5": precision_at_k(retrieved, relevant, 5),
        "precision_10": precision_at_k(retrieved, relevant, 10),
        "ndcg_10": ndcg_at_k(retrieved, relevant, 10),
        "mrr": mrr(retrieved, relevant),
        "total_retrieved": len(retrieved),
        "total_relevant": len(relevant),
    }


def average_metrics(per_query: list[dict]) -> dict:
    keys = ["recall_50", "precision_1", "precision_5", "precision_10", "ndcg_10", "mrr"]
    avgs: dict = {}
    for key in keys:
        vals = [m[key] for m in per_query if m.get(key) is not None]
        avgs[key] = sum(vals) / len(vals) if vals else None
    avgs["queries_evaluated"] = len(per_query)
    avgs["queries_with_judgments"] = sum(1 for m in per_query if m.get("total_relevant", 0) > 0)
    return avgs


def category_breakdown(per_query: list[dict]) -> dict:
    cats: dict[str, list[dict]] = {}
    for row in per_query:
        cats.setdefault(row.get("type", "unknown"), []).append(row)
    out: dict = {}
    for cat_type, rows in cats.items():
        n = len(rows)
        out[cat_type] = {
            "count": n,
            **{
                key: sum(r.get(key, 0) or 0 for r in rows) / n
                for key in ["recall_50", "precision_1", "precision_5", "precision_10", "ndcg_10", "mrr"]
            },
        }
    return out
