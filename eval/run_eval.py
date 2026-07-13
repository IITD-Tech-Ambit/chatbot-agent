#!/usr/bin/env python3
"""
Rigorous evaluation harness for chatbot-agent.

Usage:
  cd chatbot-agent
  python -m eval.run_eval                    # auto-detect live vs offline
  python -m eval.run_eval --mode live        # require live services
  python -m eval.run_eval --mode offline     # mock retriever on corpus
  python -m eval.run_eval --suite hard       # hard golden set only
  python -m eval.run_eval --verbose          # per-query details
  python -m eval.run_eval --output report.json

Requires eval/fixtures/*.json by default (corpus_v3, golden_*_v3).
Use --legacy-v2 or --legacy for older fixture sets.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from eval.clients import MockRetriever, RetrieverSession, check_services
from eval.e2e_runner import run_e2e_eval
from eval.fixtures import (
    load_corpus,
    load_golden_comprehensive,
    load_golden_hard,
    set_fixture_version,
)
from eval.multi_hop import generate_edge_case_queries, generate_multi_hop_queries
from eval.report import format_e2e_report, format_retrieval_report, production_verdict
from eval.retrieval_runner import run_retrieval_eval


async def main_async(args: argparse.Namespace) -> int:
    if args.legacy:
        set_fixture_version("v1")
    elif args.legacy_v2:
        set_fixture_version("v2")

    fixture_label = "opensearch legacy" if args.legacy else ("eval v2" if args.legacy_v2 else "eval v3")

    print(f"\nChatbot-Agent Evaluation Harness")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Mode: {args.mode}")
    print(f"Fixtures: {fixture_label}")

    services = await check_services(args.chatbot_url)
    print(f"Services: {services.as_dict()}")

    mode = args.mode
    if mode == "auto":
        mode = "live" if services.retrieval_ready else "offline"

    corpus = load_corpus()
    print(f"Corpus: {corpus['total_documents']} docs from {corpus.get('exported_at', 'unknown')}")

    retrieval_reports: list[dict] = []

    # ── Retrieval eval ──
    suites = []
    if args.suite in ("all", "comprehensive"):
        suites.append(("comprehensive", load_golden_comprehensive()))
    if args.suite in ("all", "hard"):
        suites.append(("hard", load_golden_hard()))

    # Multi-hop and edge cases appended to comprehensive run
    extra_queries = generate_multi_hop_queries(corpus, count=args.multihop_count)
    edge_queries = generate_edge_case_queries(corpus)

    if mode == "live":
        async with RetrieverSession(top_k=50) as retriever:
            for label, golden in suites:
                combined = {**golden, "queries": list(golden["queries"]) + extra_queries}
                rep = await run_retrieval_eval(
                    retriever, combined, top_k_retrieve=50, label=label,
                )
                retrieval_reports.append(rep)
                print(format_retrieval_report(rep))

            # Edge cases (separate — many have no judgments)
            edge_set = {"queries": edge_queries, "description": "Edge case queries"}
            edge_rep = await run_retrieval_eval(retriever, edge_set, label="edge_cases")
            retrieval_reports.append(edge_rep)
            print(format_retrieval_report(edge_rep))
    else:
        print("\n[OFFLINE] Using MockRetriever (token-overlap on corpus sample)")
        mock = MockRetriever(corpus["documents"], top_k=50)
        for label, golden in suites:
            combined = {**golden, "queries": list(golden["queries"]) + extra_queries}
            rep = await run_retrieval_eval(mock, combined, top_k_retrieve=50, label=f"{label}_mock")
            retrieval_reports.append(rep)
            print(format_retrieval_report(rep))

        edge_set = {"queries": edge_queries, "description": "Edge case queries (mock)"}
        edge_rep = await run_retrieval_eval(mock, edge_set, label="edge_cases_mock")
        retrieval_reports.append(edge_rep)
        print(format_retrieval_report(edge_rep))

    # ── E2E eval ──
    e2e_report = None
    if mode == "live" and services.chatbot and not args.skip_e2e:
        # Sample golden queries with judgments for source hit rate
        golden = load_golden_comprehensive()
        e2e_queries = [
            {"id": q["id"], "query": q["query"], "relevant": q.get("relevant", {})}
            for q in golden["queries"]
            if q.get("type") in ("exact_title", "semantic_paraphrase", "semantic", "partial_title")
        ][:args.e2e_limit]

        e2e_report = await run_e2e_eval(args.chatbot_url, e2e_queries)
        print(format_e2e_report(e2e_report))
    else:
        print("\n[E2E] Skipped — chatbot not running or --skip-e2e")

    # ── Unit tests ──
    unit_passed = None
    if not args.skip_unit:
        print("\nRunning pytest suite...")
        unit_passed = _run_pytest()
        print(f"Pytest: {'PASSED' if unit_passed else 'FAILED'}")

    # ── Verdict ──
    verdict = production_verdict(
        services=services.as_dict(),
        retrieval_reports=retrieval_reports,
        e2e_report=e2e_report,
        unit_tests_passed=unit_passed,
        mode=mode,
    )

    print(f"\n{'='*60}")
    print("  PRODUCTION READINESS VERDICT")
    print(f"{'='*60}")
    print(f"  {verdict['verdict']}")
    print("\n  Strengths:")
    for s in verdict["strengths"]:
        print(f"    + {s}")
    print("\n  Weaknesses:")
    for w in verdict["weaknesses"]:
        print(f"    - {w}")
    if verdict["blockers"]:
        print("\n  Blockers:")
        for b in verdict["blockers"]:
            print(f"    ! {b}")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "services": services.as_dict(),
        "corpus": {
            "total_documents": corpus["total_documents"],
            "exported_at": corpus.get("exported_at"),
        },
        "retrieval": retrieval_reports,
        "e2e": e2e_report,
        "unit_tests_passed": unit_passed,
        "verdict": verdict,
    }

    if args.verbose:
        print("\n--- Verbose per-query (first 5 per suite) ---")
        for rep in retrieval_reports:
            for row in rep.get("per_query", [])[:5]:
                print(f"  [{row.get('id')}] MRR={row.get('mrr', 0):.2f} "
                      f"P@1={row.get('precision_1', 0):.2f} "
                      f"lat={row.get('latency_ms')}ms — {row.get('query', '')[:50]}")

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nReport written to {out}")

    return 0 if verdict.get("go") or mode == "offline" else 1


def _run_pytest() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        print(result.stderr[-1000:] if result.stderr else "")
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Chatbot-agent evaluation harness")
    parser.add_argument("--legacy", action="store_true", help="Use opensearch/tests/fixtures (v1)")
    parser.add_argument("--legacy-v2", action="store_true", help="Use eval v2 fixtures (corpus_v2, golden_*_v2)")
    parser.add_argument("--mode", choices=["auto", "live", "offline"], default="auto")
    parser.add_argument("--suite", choices=["all", "comprehensive", "hard"], default="all")
    parser.add_argument("--chatbot-url", default="http://localhost:3003")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--skip-unit", action="store_true")
    parser.add_argument("--e2e-limit", type=int, default=20)
    parser.add_argument("--multihop-count", type=int, default=15)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default="eval/results/latest.json")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
