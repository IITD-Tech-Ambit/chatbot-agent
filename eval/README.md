# Chatbot-Agent Retrieval Evaluation

Evaluates hybrid retrieval (OpenSearch BM25 + kNN + kerberos/department boosts) against a stratified paper corpus sampled from MongoDB.

## Fixture versions

| Version | Files | Flag |
|---------|-------|------|
| **v3** (default) | `corpus_v3.json`, `golden_comprehensive_v3.json`, `golden_hard_v3.json` | — |
| v2 | `corpus_v2.json`, `golden_*_v2.json` | `--legacy-v2` |
| v1 | `opensearch/tests/fixtures/*` | `--legacy` |

## Regenerating fixtures

```bash
cd chatbot-agent
.venv/bin/python eval/scripts/sample_corpus_v3.py
.venv/bin/python eval/scripts/build_golden_v3.py
```

Requires `MONGODB_URI` in `.env` (papers in `researchmetadatascopus`, faculty in `faculties`, departments in `departments`).

## Query crafting rules

Queries and relevance judgments use **only** these dimensions (aligned with `agent/rag/retriever.py`):

| Dimension | Source |
|-----------|--------|
| `title` | Paper title |
| `abstract` | Paper abstract |
| `keywords` | Paper keywords or subject_area fallback |
| `department` | `faculties.department` → `departments.name` via `paper.kerberos` |
| Professor name | `faculties.firstName` + `lastName` via `paper.kerberos` |

**Never used for query text or judgments:** `field_associated`, `authors`, `author_names`.

Faculty display names use first + last name only (no honorific `title` field like "Prof"). When multiple faculty share a surname in the corpus, queries include department for disambiguation.

## Query tiers

### Comprehensive (`golden_comprehensive_v3.json`, ~150)

- **Standard:** `exact_title`, `partial_title`, `abstract_keyword`, `semantic_paraphrase`
- **Hard (in comprehensive):** `abstract_gap`, `faculty_kerberos`, `department_scoped`, `multi_hop_temporal`, `multi_hop_topic`, `multi_hop_faculty`

### Hard / extra-hard (`golden_hard_v3.json`, ~60)

- `hard_exact_rank1`, `hard_partial_common`, `hard_graded_cluster` (abstract bigrams)
- `hard_paraphrase`, `very_hard_abstract_only`
- `kerberos_disambiguation`, `cross_department_topic`

## Running evaluation

```bash
.venv/bin/python -m eval.run_eval                  # auto: live if services up, else offline mock
.venv/bin/python -m eval.run_eval --mode live     # require OpenSearch, MongoDB, embedding service
.venv/bin/python -m eval.run_eval --suite hard     # hard set only
.venv/bin/python -m eval.run_eval --legacy-v2      # use v2 fixtures
```

## What the chatbot actually searches

The agent retriever (`retriever.py`) uses:

1. BM25 on `title` (^6) and `abstract` (^2) with fuzziness
2. Phrase boost on title/abstract
3. kNN on `embedding`
4. Optional `kerberos` constant_score boost (faculty name match via MongoDB)
5. Optional department `kerberos` boost (department name match via MongoDB)

OpenSearch UI search additionally indexes `subject_area` and `field_associated`, but the **chatbot agent does not** use those fields for retrieval.
