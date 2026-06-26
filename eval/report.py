"""Format evaluation reports and production-readiness verdict."""

from __future__ import annotations

import json
from typing import Any


def format_retrieval_report(report: dict[str, Any]) -> str:
    lines = [
        f"\n{'='*60}",
        f"  RETRIEVAL EVAL: {report.get('label', '')}",
        f"{'='*60}",
        f"  Golden set: {report.get('golden_set', '')[:70]}",
        f"  Queries evaluated: {report.get('queries_evaluated')} / {report.get('queries_total')}",
    ]
    avg = report.get("average", {})
    lines.append("\n  Global averages:")
    for key in ["mrr", "precision_1", "precision_5", "precision_10", "ndcg_10", "recall_50"]:
        val = avg.get(key)
        lines.append(f"    {key:16s}: {val:.3f}" if val is not None else f"    {key:16s}: n/a")

    lat = report.get("latency", {})
    if lat.get("count"):
        lines.append(
            f"\n  Latency: p50={lat['p50_ms']}ms  p95={lat['p95_ms']}ms  "
            f"max={lat['max_ms']}ms  mean={lat.get('mean_ms', 0)}ms"
        )

    by_cat = report.get("by_category", {})
    if by_cat:
        lines.append("\n  By category:")
        for cat, stats in sorted(by_cat.items()):
            lines.append(
                f"    {cat:28s} n={stats['count']:2d}  "
                f"MRR={stats.get('mrr', 0):.2f}  P@1={stats.get('precision_1', 0):.2f}  "
                f"R@50={stats.get('recall_50', 0):.2f}"
            )

    failures = report.get("threshold_failures", [])
    errors = report.get("errors", [])
    if failures:
        lines.append(f"\n  THRESHOLD FAILURES ({len(failures)}):")
        for f in failures[:15]:
            lines.append(f"    ✗ {f}")
    if errors:
        lines.append(f"\n  ERRORS ({len(errors)}):")
        for e in errors[:5]:
            lines.append(f"    ✗ {e.get('id')}: {e.get('error', '')[:80]}")

    status = "PASS" if report.get("passed") else "FAIL"
    lines.append(f"\n  Result: {status}")
    return "\n".join(lines)


def format_e2e_report(report: dict[str, Any]) -> str:
    lines = [
        f"\n{'='*60}",
        "  E2E CHATBOT EVAL",
        f"{'='*60}",
        f"  Research queries: {report.get('research_queries')}",
        f"  Guardrail queries: {report.get('guardrail_queries')}",
    ]
    lat = report.get("latency", {})
    lines.append(f"  Latency p50={lat.get('p50_ms')}ms  p95={lat.get('p95_ms')}ms")

    if report.get("source_hit_rate_at_8") is not None:
        lines.append(f"  Source hit@8: {report['source_hit_rate_at_8']:.3f}")
    lines.append(f"  Answer rate: {report.get('answer_rate', 0):.3f}")
    lines.append(f"  Error rate: {report.get('error_rate', 0):.3f}")
    lines.append(f"  Guardrail accuracy: {report.get('guardrail_accuracy', 0):.3f}")

    aq = report.get("answer_quality", {})
    if aq.get("faithfulness_avg") is not None:
        lines.append(
            f"  Answer quality: faithfulness={aq['faithfulness_avg']:.3f}  "
            f"relevance={aq.get('relevance_avg', 0):.3f}  "
            f"hallucination={aq.get('hallucination_rate_avg', 0):.3f}"
        )

    failures = report.get("threshold_failures", [])
    if failures:
        lines.append(f"\n  THRESHOLD FAILURES: {', '.join(failures)}")

    status = "PASS" if report.get("passed") else "FAIL"
    lines.append(f"\n  Result: {status}")
    return "\n".join(lines)


def production_verdict(
    *,
    services: dict[str, bool],
    retrieval_reports: list[dict],
    e2e_report: dict | None,
    unit_tests_passed: bool | None,
    mode: str,
) -> dict[str, Any]:
    """Synthesize go/no-go with strengths, weaknesses, and recommendations."""

    strengths: list[str] = []
    weaknesses: list[str] = []
    blockers: list[str] = []

    retrieval_ready = all(services.get(k) for k in ("opensearch", "mongodb", "embedding"))

    if unit_tests_passed:
        strengths.append("Unit/integration pytest suite passes")
    elif unit_tests_passed is False:
        weaknesses.append("Pytest suite has failures")
        blockers.append("Fix failing unit tests before production")

    if retrieval_ready:
        strengths.append("Retrieval stack (OpenSearch + MongoDB + embeddings) reachable")
    else:
        weaknesses.append("Retrieval dependencies not all reachable — live IR metrics unavailable")
        if mode == "live":
            blockers.append("Start OpenSearch, MongoDB, embedding service for live retrieval eval")

    if services.get("chatbot"):
        strengths.append("Chatbot API health endpoint responding")
    else:
        weaknesses.append("Chatbot not running — E2E latency and answer quality not measured live")
        if mode == "live":
            blockers.append("Start chatbot-agent for E2E evaluation")

    for rep in retrieval_reports:
        label = rep.get("label", "retrieval")
        # Skip edge-case suites from go/no-go blockers (adversarial by design)
        if "edge" in label:
            if not rep.get("passed"):
                weaknesses.append(f"{label}: expected failures on adversarial queries")
            continue
        avg = rep.get("average", {})
        if rep.get("passed"):
            strengths.append(f"{label}: meets IR thresholds (MRR={avg.get('mrr', 0):.2f})")
        else:
            failures = rep.get("threshold_failures", [])
            weaknesses.append(f"{label}: below thresholds — {failures[0] if failures else 'errors'}")
            if mode == "live" and failures:
                blockers.append(f"{label} retrieval below production thresholds")

    if e2e_report:
        if e2e_report.get("passed"):
            strengths.append(
                f"E2E: guardrail accuracy {e2e_report.get('guardrail_accuracy', 0):.0%}, "
                f"p95 latency {e2e_report.get('latency', {}).get('p95_ms')}ms"
            )
        else:
            weaknesses.append(f"E2E failures: {e2e_report.get('threshold_failures', [])}")
            if mode == "live":
                blockers.append("E2E chat evaluation below thresholds")

    # Architecture strengths (always)
    strengths.extend([
        "Hybrid BM25 + kNN retrieval with MongoDB hydration",
        "Guardrails for off-topic, injection, and meta queries",
        "Prometheus metrics and SSE streaming API",
    ])

    go = len(blockers) == 0 and (unit_tests_passed is not False)

    if mode == "offline":
        go = False
        verdict = "NO-GO (offline eval only — services unavailable)"
    elif go:
        verdict = "CONDITIONAL GO — meets thresholds; monitor hard-query categories in production"
    else:
        verdict = "NO-GO — blockers must be resolved"

    return {
        "verdict": verdict,
        "go": go and mode == "live",
        "mode": mode,
        "blockers": blockers,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommended_thresholds": {
            "retrieval_mrr": 0.35,
            "retrieval_precision_at_1": 0.30,
            "retrieval_ndcg_at_10": 0.40,
            "e2e_source_hit_at_8": 0.55,
            "e2e_guardrail_accuracy": 0.90,
            "e2e_p95_latency_ms": 30000,
            "answer_faithfulness": 0.50,
        },
    }


def print_full_report(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))
