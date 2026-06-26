"""Production-readiness thresholds for retrieval and E2E evaluation."""

from __future__ import annotations

# Global retrieval thresholds (chatbot retriever uses top_k=8 by default)
RETRIEVAL_GLOBAL = {
    "mrr": 0.35,
    "precision_1": 0.30,
    "precision_5": 0.15,
    "precision_10": 0.12,
    "ndcg_10": 0.40,
    "recall_50": 0.25,
}

# Per-category minimums for comprehensive golden set
COMPREHENSIVE_CATEGORY = {
    "exact_title": {"mrr": 0.70, "precision_1": 0.60, "ndcg_10": 0.70},
    "partial_title": {"mrr": 0.30, "recall_50": 0.40},
    "abstract_keyword": {"mrr": 0.15, "recall_50": 0.25},
    "semantic": {"mrr": 0.10, "recall_50": 0.20},
    "author": {"mrr": 0.20, "precision_5": 0.10},
    "field_broad": {"recall_50": 0.10},
    "cross_field": {"mrr": 0.15, "precision_10": 0.05},
    "multi_relevant": {"ndcg_10": 0.25, "recall_50": 0.20},
}

# Hard golden set (ported from opensearch/tests/eval/eval_hard.mjs)
HARD_CATEGORY = {
    "hard_exact_rank1": {"mrr": 0.5, "precision_1": 0.4, "ndcg_10": 0.5},
    "hard_partial_common": {"mrr": 0.2, "recall_50": 0.5, "precision_10": 0.05},
    "hard_graded_cluster": {"ndcg_10": 0.3, "recall_50": 0.25},
    "hard_ambiguous_recall": {"recall_50": 0.15},
    "hard_abstract_gap": {"mrr": 0.1, "recall_50": 0.3},
    "hard_paraphrase": {"mrr": 0.05, "recall_50": 0.2},
    "hard_author_disambiguation": {"mrr": 0.2, "precision_5": 0.05},
    "hard_distractor_ranking": {"ndcg_10": 0.4, "mrr": 0.25},
    "hard_cross_field": {"mrr": 0.15, "precision_10": 0.05},
}

# E2E chatbot thresholds
E2E_THRESHOLDS = {
    "source_hit_rate_at_8": 0.55,  # fraction of queries where a relevant doc appears in streamed sources
    "answer_min_length": 40,
    "guardrail_accuracy": 0.90,
    "error_rate_max": 0.05,
    "p95_latency_ms": 30_000,
    "p50_latency_ms": 12_000,
}

# Answer quality heuristics (no LLM judge)
ANSWER_QUALITY = {
    "faithfulness_min": 0.50,  # fraction of answer key terms found in source abstracts
    "relevance_min": 0.35,     # query term overlap with answer
    "hallucination_rate_max": 0.25,  # unsupported numeric claims / max fraction
}

# Retriever top-k used by chatbot (for source_hit_rate)
CHAT_TOP_K = 8
