"""
Live golden-set generator + retrieval evaluator.

Fetches random faculty, departments, and papers from the production MongoDB/OpenSearch,
builds ~500 grounded test cases, runs them through the retriever, and reports metrics.

Methodology
───────────
The only reliable paper→faculty linkage is kerberos (set by the indexer on every
attributed paper).  All golden queries are built around this chain:

  faculty.email prefix  →  kerberos  →  paper.kerberos

Query categories generated
  faculty_fullname  : "FirstName LastName"  → papers with kerberos = that faculty
  faculty_lastname  : "LastName"            → papers with kerberos = that faculty (harder)
  dept_broad        : "Department Name"     → papers with kerberos ∈ dept faculty set
  dept_topic        : "keyword dept_name"   → a paper from that dept containing keyword
  topic_title       : 3–5 words from title  → that specific paper
  topic_abstract    : 3–5 words from abstract → that specific paper (hardest)

Usage
  cd chatbot-agent
  python -m eval.live_golden_eval                    # run with defaults
  python -m eval.live_golden_eval --n-faculty 50 --n-dept 30 --n-topic 100
  python -m eval.live_golden_eval --output eval/results/live_golden.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from agent.config import settings
from agent.data import mongo, opensearch as os_mod, redis as redis_mod
from agent.rag.embeddings import EmbeddingClient
from agent.rag.retriever import Retriever
from agent.repositories.faculty_repo import FacultyRepository
from agent.repositories.research_repo import ResearchRepository
from eval.metrics import average_metrics, category_breakdown, compute_all

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# ── Stopwords for keyword extraction ──────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with",
    "by", "at", "from", "is", "are", "was", "were", "be", "been", "being",
    "its", "it", "this", "that", "these", "those", "as", "into", "not",
    "but", "also", "has", "have", "had", "do", "does", "did", "will",
    "would", "could", "can", "may", "might", "via", "between", "through",
    "based", "using", "new", "novel", "study", "analysis", "paper",
    "results", "approach", "method", "system", "model", "high", "low",
    "two", "three", "multi", "non", "under", "over", "our", "their",
})

_MIN_WORD_LEN = 4
_KEYWORD_WINDOW = 5   # words to include in a topic query


def _extract_keywords(text: str, n: int = _KEYWORD_WINDOW) -> list[str]:
    """Extract `n` non-stop words from text (preserving order)."""
    words = re.findall(r"[a-zA-Z]{%d,}" % _MIN_WORD_LEN, text)
    filtered = [w for w in words if w.lower() not in _STOPWORDS]
    # Pick a random start window from the first 30 candidates so queries vary
    pool = filtered[:30] if len(filtered) >= n else filtered
    if len(pool) < n:
        return pool
    start = random.randint(0, max(0, len(pool) - n))
    return pool[start: start + n]


def _kerberos_from_email(email: str) -> str | None:
    if not email or "@" not in email:
        return None
    return email.split("@")[0].strip().lower() or None


# ── Golden set builder ─────────────────────────────────────────────────────────

class LiveGoldenBuilder:
    def __init__(
        self,
        faculty_repo: FacultyRepository,
        research_repo: ResearchRepository,
        opensearch_client: Any,
        index_name: str,
    ) -> None:
        self._faculty = faculty_repo
        self._papers = research_repo
        self._os = opensearch_client
        self._index = index_name

    async def _papers_for_kerberos(self, kerberos: str, max_papers: int = 5) -> list[str]:
        """Return mongo_ids of the top-cited papers attributed to this kerberos."""
        body = {
            "size": max_papers,
            "_source": ["mongo_id"],
            "query": {"term": {"kerberos": kerberos}},
            "sort": [{"citation_count": "desc"}, "_score"],
        }
        try:
            resp = await self._os.search(index=self._index, body=body)
            return [
                h["_source"]["mongo_id"]
                for h in resp.get("hits", {}).get("hits", [])
                if h.get("_source", {}).get("mongo_id")
            ]
        except Exception as exc:
            logger.debug("papers_for_kerberos(%s) failed: %s", kerberos, exc)
            return []

    async def _all_kerberoses_for_dept(self, dept_id: Any) -> list[str]:
        """Return all kerberos values for faculty in a department."""
        faculty_docs = await self._faculty.find_faculty_by_department_id(dept_id)
        kerberoses = []
        for f in faculty_docs:
            k = _kerberos_from_email(f.get("email", ""))
            if k:
                kerberoses.append(k)
        return kerberoses

    async def _papers_for_dept_kerberos(
        self, kerberoses: list[str], max_papers: int = 10
    ) -> list[str]:
        """Return mongo_ids of top-cited papers from a list of kerberos IDs."""
        if not kerberoses:
            return []
        body = {
            "size": max_papers,
            "_source": ["mongo_id"],
            "query": {"terms": {"kerberos": kerberoses}},
            "sort": [{"citation_count": "desc"}, "_score"],
        }
        try:
            resp = await self._os.search(index=self._index, body=body)
            return [
                h["_source"]["mongo_id"]
                for h in resp.get("hits", {}).get("hits", [])
                if h.get("_source", {}).get("mongo_id")
            ]
        except Exception as exc:
            logger.debug("papers_for_dept failed: %s", exc)
            return []

    async def _mongo_doc(self, mongo_id: str) -> dict | None:
        from bson import ObjectId
        if not ObjectId.is_valid(mongo_id):
            return None
        try:
            doc = await self._papers._coll.find_one(
                {"_id": ObjectId(mongo_id)},
                {"title": 1, "abstract": 1, "kerberos": 1, "citation_count": 1},
            )
            return doc
        except Exception:
            return None

    # ── Query generators ──────────────────────────────────────────────────────

    async def build_faculty_queries(
        self, n_faculty: int, seed: int = 42
    ) -> list[dict[str, Any]]:
        """Sample random faculty and generate name-based queries."""
        cursor = self._faculty._faculty.aggregate([
            {"$match": {"email": {"$regex": "@iitd.ac.in", "$options": "i"}}},
            {"$sample": {"size": n_faculty * 2}},  # over-sample for filtering
            {"$project": {"firstName": 1, "lastName": 1, "email": 1}},
        ])
        faculty_docs = await cursor.to_list(length=n_faculty * 2)

        queries: list[dict] = []
        seen_kerb: set[str] = set()
        rng = random.Random(seed)

        for f in faculty_docs:
            if len(queries) >= n_faculty:
                break
            first = (f.get("firstName") or "").strip()
            last = (f.get("lastName") or "").strip()
            email = f.get("email", "")
            kerberos = _kerberos_from_email(email)
            if not kerberos or not last or kerberos in seen_kerb:
                continue

            paper_ids = await self._papers_for_kerberos(kerberos, max_papers=5)
            if not paper_ids:
                continue  # no attributed papers found
            seen_kerb.add(kerberos)

            relevant = {pid: 3 for pid in paper_ids}  # grade 3 = exact faculty match

            # Full name query
            if first:
                queries.append({
                    "id": f"live_faculty_full_{kerberos}",
                    "query": f"{first} {last}",
                    "type": "faculty_fullname",
                    "difficulty": "medium",
                    "kerberos": kerberos,
                    "relevant": relevant,
                })

            # Last-name-only query (harder — more ambiguous)
            if rng.random() < 0.5 and len(last) >= 4:
                queries.append({
                    "id": f"live_faculty_last_{kerberos}",
                    "query": last,
                    "type": "faculty_lastname",
                    "difficulty": "hard",
                    "kerberos": kerberos,
                    "relevant": relevant,
                })

        return queries

    async def build_dept_queries(
        self, n_dept: int, seed: int = 43
    ) -> list[dict[str, Any]]:
        """Sample random departments and generate department-name queries."""
        cursor = self._faculty._departments.aggregate([
            {"$sample": {"size": n_dept * 2}},
            {"$project": {"name": 1, "code": 1}},
        ])
        dept_docs = await cursor.to_list(length=n_dept * 2)

        queries: list[dict] = []
        rng = random.Random(seed)

        for dept in dept_docs:
            if len(queries) >= n_dept:
                break
            dept_name = (dept.get("name") or "").strip()
            dept_id = dept.get("_id")
            if not dept_name or not dept_id:
                continue

            kerberoses = await self._all_kerberoses_for_dept(dept_id)
            if not kerberoses:
                continue

            paper_ids = await self._papers_for_dept_kerberos(kerberoses, max_papers=10)
            if len(paper_ids) < 2:  # need at least 2 relevant papers to be meaningful
                continue

            relevant = {pid: 2 for pid in paper_ids}  # grade 2 = dept match

            queries.append({
                "id": f"live_dept_{dept_id}",
                "query": dept_name,
                "type": "dept_broad",
                "difficulty": "hard",
                "dept_name": dept_name,
                "n_faculty": len(kerberoses),
                "relevant": relevant,
            })

            # Also make a partial / abbreviation query when dept name is long
            words = dept_name.split()
            if len(words) >= 3 and rng.random() < 0.4:
                short = " ".join(words[-2:])  # last two words (e.g., "Electrical Engineering")
                queries.append({
                    "id": f"live_dept_short_{dept_id}",
                    "query": short,
                    "type": "dept_broad",
                    "difficulty": "hard",
                    "dept_name": dept_name,
                    "n_faculty": len(kerberoses),
                    "relevant": relevant,
                })

        return queries

    async def build_topic_queries(
        self, n_topic: int, from_abstract: bool = False, seed: int = 44
    ) -> list[dict[str, Any]]:
        """
        Sample random papers from the live index and build keyword queries.
        If from_abstract=True, extract keywords from the abstract (harder).
        """
        body = {
            "size": n_topic * 3,
            "_source": ["mongo_id", "title", "abstract"],
            "query": {
                "function_score": {
                    "query": {"match_all": {}},
                    "random_score": {"seed": seed, "field": "mongo_id"},
                }
            },
        }
        try:
            resp = await self._os.search(index=self._index, body=body)
            hits = resp.get("hits", {}).get("hits", [])
        except Exception as exc:
            logger.warning("topic sample failed: %s", exc)
            return []

        queries: list[dict] = []
        rng = random.Random(seed)

        for h in hits:
            if len(queries) >= n_topic:
                break
            src = h.get("_source", {})
            mongo_id = src.get("mongo_id", "")
            title = (src.get("title") or "").strip()
            abstract = (src.get("abstract") or "").strip()
            if not mongo_id or not title:
                continue

            text = abstract if (from_abstract and len(abstract) > 50) else title
            keywords = _extract_keywords(text, n=rng.randint(3, 5))
            if len(keywords) < 3:
                continue

            query_str = " ".join(keywords)
            qtype = "topic_abstract" if from_abstract and text is abstract else "topic_title"

            queries.append({
                "id": f"live_topic_{mongo_id[:12]}",
                "query": query_str,
                "type": qtype,
                "difficulty": "hard" if from_abstract else "medium",
                "source_mongo_id": mongo_id,
                "relevant": {mongo_id: 3},
            })

        return queries

    async def build_dept_topic_queries(
        self, n: int, seed: int = 45
    ) -> list[dict[str, Any]]:
        """
        Build queries like "keyword department_name" where the keyword is from a paper
        in that department (via kerberos).  Tests combined content+field retrieval.
        """
        cursor = self._faculty._departments.aggregate([
            {"$sample": {"size": n * 3}},
            {"$project": {"name": 1}},
        ])
        dept_docs = await cursor.to_list(length=n * 3)

        queries: list[dict] = []
        rng = random.Random(seed)

        for dept in dept_docs:
            if len(queries) >= n:
                break
            dept_name = (dept.get("name") or "").strip()
            dept_id = dept.get("_id")
            if not dept_name or not dept_id:
                continue

            kerberoses = await self._all_kerberoses_for_dept(dept_id)
            if not kerberoses:
                continue

            # Pick one random paper from this dept
            paper_ids = await self._papers_for_dept_kerberos(kerberoses, max_papers=20)
            if not paper_ids:
                continue

            mongo_id = rng.choice(paper_ids)
            doc = await self._mongo_doc(mongo_id)
            if not doc:
                continue

            title = (doc.get("title") or "").strip()
            keywords = _extract_keywords(title, n=rng.randint(2, 3))
            if not keywords:
                continue

            query_str = " ".join(keywords) + " " + dept_name

            queries.append({
                "id": f"live_depttopic_{mongo_id[:12]}",
                "query": query_str,
                "type": "dept_topic",
                "difficulty": "hard",
                "dept_name": dept_name,
                "relevant": {mongo_id: 3},
            })

        return queries


# ── Retrieval evaluation ───────────────────────────────────────────────────────

async def run_live_golden_eval(
    retriever: Retriever,
    queries: list[dict[str, Any]],
    top_k: int = 50,
    label: str = "live_golden",
) -> dict[str, Any]:
    """Run retrieval eval against the generated live queries."""
    import time

    per_query: list[dict] = []
    errors: list[dict] = []
    latencies_ms: list[float] = []

    for entry in queries:
        query = entry.get("query", "").strip()
        if not query:
            continue
        t0 = time.perf_counter()
        try:
            results = await retriever.retrieve(query, top_k=top_k)
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

    lat_sorted = sorted(latencies_ms)
    n = len(lat_sorted)
    latency = {
        "count": n,
        "p50_ms": round(lat_sorted[int(n * 0.50)] if n else 0, 1),
        "p95_ms": round(lat_sorted[min(n - 1, int(n * 0.95))] if n else 0, 1),
        "max_ms": round(max(lat_sorted, default=0), 1),
        "mean_ms": round(sum(lat_sorted) / n if n else 0, 1),
    }

    avg = average_metrics(per_query)
    by_cat = category_breakdown(per_query)

    return {
        "label": label,
        "queries_total": len(queries),
        "queries_evaluated": len(per_query),
        "errors": errors,
        "average": avg,
        "by_category": by_cat,
        "latency": latency,
        "per_query": per_query,
    }


# ── Pretty printer ─────────────────────────────────────────────────────────────

def _print_report(report: dict) -> None:
    avg = report["average"]
    by_cat = report["by_category"]
    lat = report["latency"]

    print(f"\n{'='*65}")
    print(f"  LIVE GOLDEN EVAL — {report['label']}")
    print(f"  {report['queries_evaluated']}/{report['queries_total']} queries evaluated")
    print(f"  Errors: {len(report['errors'])}")
    print(f"{'='*65}")
    print(f"  {'Metric':<22} {'Value':>8}  Threshold")
    print(f"  {'-'*40}")

    thresholds = {
        "mrr": 0.35, "precision_1": 0.30, "precision_5": 0.15,
        "precision_10": 0.12, "ndcg_10": 0.40, "recall_50": 0.25,
    }
    labels = {
        "mrr": "MRR", "precision_1": "P@1", "precision_5": "P@5",
        "precision_10": "P@10", "ndcg_10": "nDCG@10", "recall_50": "Recall@50",
    }
    for key, label in labels.items():
        val = avg.get(key, 0)
        thresh = thresholds.get(key, 0)
        flag = "✓" if val >= thresh else "✗"
        print(f"  {label:<22} {val:>8.3f}  ≥{thresh:.2f} {flag}")

    print(f"\n  Latency  p50={lat['p50_ms']}ms  p95={lat['p95_ms']}ms")

    print(f"\n  By category:")
    for cat, m in sorted(by_cat.items(), key=lambda x: x[1].get("mrr", 0), reverse=True):
        count = m.get("queries_evaluated", "?")
        print(f"    {cat:<25} MRR={m.get('mrr',0):.3f}  P@1={m.get('precision_1',0):.3f}"
              f"  R@50={m.get('recall_50',0):.3f}  (n={count})")

    if report["errors"]:
        print(f"\n  Errors:")
        for e in report["errors"][:5]:
            print(f"    [{e.get('id')}] {e.get('error','')[:80]}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> int:
    print(f"\nLive Golden Eval — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    db = await mongo.connect(settings.MONGODB_URI)
    os_client = await os_mod.connect(
        settings.OPENSEARCH_NODE,
        settings.OPENSEARCH_USER,
        settings.OPENSEARCH_PASSWORD,
        verify_certs=settings.OPENSEARCH_VERIFY_CERTS,
        use_ssl=settings.OPENSEARCH_USE_SSL,
    )
    redis_client = await redis_mod.connect(settings.REDIS_URL)

    faculty_repo = FacultyRepository(db)
    research_repo = ResearchRepository(db)
    embed_client = EmbeddingClient(
        base_url=settings.EMBEDDING_SERVICE_URL,
        redis_client=redis_client,
        timeout_ms=settings.EMBEDDING_TIMEOUT_MS,
        cache_ttl=settings.EMBEDDING_CACHE_TTL,
    )
    retriever = Retriever(
        opensearch=os_client,
        index_name=settings.OPENSEARCH_INDEX,
        research_repo=research_repo,
        embedding_client=embed_client,
        top_k=50,
        faculty_repo=faculty_repo,
    )

    builder = LiveGoldenBuilder(faculty_repo, research_repo, os_client, settings.OPENSEARCH_INDEX)

    print(f"\nBuilding live golden set (seed={args.seed})...")
    print(f"  Generating {args.n_faculty} faculty queries...")
    faculty_qs = await builder.build_faculty_queries(args.n_faculty, seed=args.seed)
    print(f"    → {len(faculty_qs)} queries generated")

    print(f"  Generating {args.n_dept} department queries...")
    dept_qs = await builder.build_dept_queries(args.n_dept, seed=args.seed + 1)
    print(f"    → {len(dept_qs)} queries generated")

    print(f"  Generating {args.n_topic} title-keyword queries...")
    topic_title_qs = await builder.build_topic_queries(
        args.n_topic, from_abstract=False, seed=args.seed + 2
    )
    print(f"    → {len(topic_title_qs)} queries generated")

    print(f"  Generating {args.n_abstract} abstract-keyword queries (hard)...")
    topic_abs_qs = await builder.build_topic_queries(
        args.n_abstract, from_abstract=True, seed=args.seed + 3
    )
    print(f"    → {len(topic_abs_qs)} queries generated")

    print(f"  Generating {args.n_dept_topic} department+topic queries...")
    dept_topic_qs = await builder.build_dept_topic_queries(args.n_dept_topic, seed=args.seed + 4)
    print(f"    → {len(dept_topic_qs)} queries generated")

    all_queries = faculty_qs + dept_qs + topic_title_qs + topic_abs_qs + dept_topic_qs
    print(f"\nTotal queries: {len(all_queries)}")

    # Optionally save the generated golden set
    if args.save_golden:
        golden_path = Path(args.save_golden)
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps({
            "description": f"Live golden set generated {datetime.now(timezone.utc).isoformat()}",
            "total_queries": len(all_queries),
            "queries": all_queries,
        }, indent=2, default=str))
        print(f"Golden set saved to {golden_path}")

    print(f"\nRunning retrieval eval (top_k=50)...")
    report = await run_live_golden_eval(retriever, all_queries, top_k=50)
    _print_report(report)

    # Save full report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": "live_golden_eval",
        "config": {
            "n_faculty": args.n_faculty,
            "n_dept": args.n_dept,
            "n_topic": args.n_topic,
            "n_abstract": args.n_abstract,
            "n_dept_topic": args.n_dept_topic,
            "seed": args.seed,
            "total_queries": len(all_queries),
        },
        "report": report,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nFull report → {out_path}")

    await os_mod.close()
    await redis_mod.close()
    await mongo.close()
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Live golden-set retrieval eval")
    p.add_argument("--n-faculty",    type=int, default=100,
                   help="Faculty name queries to generate (default 100)")
    p.add_argument("--n-dept",       type=int, default=60,
                   help="Department broad queries (default 60)")
    p.add_argument("--n-topic",      type=int, default=150,
                   help="Title keyword queries (default 150)")
    p.add_argument("--n-abstract",   type=int, default=100,
                   help="Abstract keyword queries — harder (default 100)")
    p.add_argument("--n-dept-topic", type=int, default=90,
                   help="Dept+topic combined queries (default 90)")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--save-golden",  type=str, default=None,
                   help="Path to save the generated golden set JSON")
    p.add_argument("--output",       type=str, default="eval/results/live_golden.json",
                   help="Path for the full results JSON")
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
