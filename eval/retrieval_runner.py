"""Retrieval evaluation runner against golden sets."""

from __future__ import annotations

import time
from typing import Any, Protocol

from eval.metrics import average_metrics, category_breakdown, compute_all
from eval.thresholds import COMPREHENSIVE_CATEGORY, HARD_CATEGORY, RETRIEVAL_GLOBAL


class RetrieverLike(Protocol):
    async def retrieve(self, query: str, top_k: int | None = None, **kwargs: Any) -> list[dict]: ...


async def run_retrieval_eval(
    retriever: RetrieverLike,
    golden_set: dict[str, Any],
    *,
    top_k_retrieve: int = 50,
    label: str = "retrieval",
) -> dict[str, Any]:
    """Evaluate retriever against a golden set; returns full report dict."""
    per_query: list[dict] = []
    latencies_ms: list[float] = []
    errors: list[dict] = []

    for entry in golden_set.get("queries", []):
        query = entry.get("query", "")
        if entry.get("expect_error") or not query.strip():
            continue

        t0 = time.perf_counter()
        try:
            results = await retriever.retrieve(query, top_k=top_k_retrieve)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed)

            retrieved_ids = [str(r.get("id", "")) for r in results]
            relevant = entry.get("relevant") or {}
            metrics = compute_all(retrieved_ids, relevant)
            per_query.append({
                "id": entry.get("id"),
                "query": query,
                "type": entry.get("type", "unknown"),
                "difficulty": entry.get("difficulty"),
                "latency_ms": round(elapsed, 1),
                "retrieved_ids": retrieved_ids[:10],
                **metrics,
            })
        except Exception as exc:
            errors.append({"id": entry.get("id"), "query": query, "error": str(exc)})

    avg = average_metrics(per_query)
    by_cat = category_breakdown(per_query)

    thresholds = HARD_CATEGORY if "hard_" in label or any(
        k.startswith("hard_") for k in by_cat
    ) else COMPREHENSIVE_CATEGORY

    failures = _check_thresholds(avg, by_cat, thresholds)

    lat_sorted = sorted(latencies_ms)
    latency = _latency_stats(lat_sorted)

    return {
        "label": label,
        "golden_set": golden_set.get("description", label),
        "queries_total": len(golden_set.get("queries", [])),
        "queries_evaluated": len(per_query),
        "errors": errors,
        "average": avg,
        "by_category": by_cat,
        "latency": latency,
        "threshold_failures": failures,
        "per_query": per_query,
        "passed": len(failures) == 0 and not errors,
    }


def _latency_stats(sorted_ms: list[float]) -> dict[str, float | int]:
    if not sorted_ms:
        return {"count": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "max_ms": 0}
    n = len(sorted_ms)

    def pct(p: float) -> float:
        idx = min(n - 1, int(n * p))
        return round(sorted_ms[idx], 1)

    return {
        "count": n,
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "max_ms": round(sorted_ms[-1], 1),
        "mean_ms": round(sum(sorted_ms) / n, 1),
    }


def _check_thresholds(
    avg: dict,
    by_cat: dict,
    category_thresholds: dict,
) -> list[str]:
    failures: list[str] = []
    for key, min_val in RETRIEVAL_GLOBAL.items():
        actual = avg.get(key)
        if actual is not None and actual < min_val:
            failures.append(f"global.{key}: {actual:.3f} < {min_val}")

    for cat, mins in category_thresholds.items():
        if cat not in by_cat:
            continue
        for key, min_val in mins.items():
            actual = by_cat[cat].get(key, 0)
            if actual < min_val:
                failures.append(f"{cat}.{key}: {actual:.3f} < {min_val}")
    return failures
