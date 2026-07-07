# Research Ambit Chatbot Agent

Agentic RAG chatbot for the [Research Ambit](https://github.com/IITD-Tech-Ambit/tech-ambit-explorer) portal at IIT Delhi. Answers natural-language questions about faculty, departments, and publications by retrieving live data from MongoDB and OpenSearch, then synthesizing a streamed response with **xAI Grok** via a **LangGraph** agent.

Built with **FastAPI**, **LangGraph**, and **LangChain**. Designed to power the floating research chat widget in `tech-ambit-explorer`.

## Role in the Research Ambit stack

```
┌─────────────────────────────────────────────────────────────────┐
│  tech-ambit-explorer (React/Vite)                    :8080      │
│  VITE_CHAT_API_URL → chatbot-agent /api/v1                      │
└───────────────────────────────┬─────────────────────────────────┘
                                │ SSE (POST /api/v1/chat)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  chatbot-agent (this repo)                           :3003      │
│  LangGraph agent → tools → MongoDB / OpenSearch / search-api  │
└───────┬─────────────┬──────────────┬──────────────┬───────────┘
        │             │              │              │
        ▼             ▼              ▼              ▼
   MongoDB      OpenSearch      Redis         Embedding svc
 (research_db) (research_docs)  (cache)         (BGE :8000)
                                    │
                                    └── search-api (Node, :3000)
                                        faculty-for-topic aggregation
```

| Service | Repo / path | Used for |
|---------|-------------|----------|
| Frontend | [`tech-ambit-explorer`](https://github.com/IITD-Tech-Ambit/tech-ambit-explorer) | Chat widget UI; calls `VITE_CHAT_API_URL` |
| Search API | [`opensearch`](https://github.com/IITD-Tech-Ambit) (local `opensearch/`) | Hybrid paper search index; faculty topic lookup via HTTP |
| Backend CMS | [`research-ambit-main`](https://github.com/IITD-Tech-Ambit/research-ambit-main) | Shared MongoDB research database (not called directly) |
| Embedding | BGE service (`opensearch` stack) | Query embeddings + cross-encoder reranking |

In production, nginx (see `opensearch/deploy/nginx/nginx.conf`) proxies the chat SSE endpoint at `/chat-api/api/v1/chat` → `chatbot:3003`.

## Features

- **Forced-tool agent graph** — always retrieves before answering; injects `search_papers` if the LLM returns no tool calls
- **12 auto-discovered tools** — papers, faculty, departments, stats, trends, comparisons, interdisciplinary search
- **Hybrid RAG retrieval** — BM25 + kNN over OpenSearch, BGE reranking, MongoDB hydration, kerberos/department boosts
- **Fast paths** — regex guardrails (greeting/identity/injection) and structured queries (h-index, citations, dept) bypass the LLM
- **Redis caching** — LLM response cache + embedding cache
- **SSE streaming** — `thinking`, `status`, `sources`, `chart`, `token`, `done` events for the frontend
- **Prometheus metrics** — exposed on the FastAPI app
- **295 pytest tests** — guardrails, routing, tools, graph, SSE endpoint (fully mocked, no live infra)
- **Retrieval eval suite** — offline/live benchmarks under `eval/` (see [`eval/README.md`](eval/README.md))

## Quick start (local)

### Prerequisites

- Python 3.11+
- Running instances of:
  - **MongoDB** — shared research database (`research_db`)
  - **OpenSearch** — paper index (`research_documents`)
  - **Redis** — rate limiting + caches
  - **BGE embedding service** — typically `http://localhost:8000`
  - **Search API** — Node.js service from the `opensearch` repo, typically `http://localhost:3000`
- **xAI API key** — [console.x.ai](https://console.x.ai/) (stored in `GROQ_API_KEY`; legacy env name)

Connection details for MongoDB, OpenSearch, and Redis are usually shared with the rest of the stack via `opensearch/.env`.

### Install and configure

```bash
cd chatbot-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Set GROQ_API_KEY (xAI key) and adjust service URLs
```

### Run

```bash
python run.py
```

Server starts at `http://localhost:3003`. Health check: `GET /health`.

Point the frontend at the agent:

```bash
# tech-ambit-explorer/.env
VITE_CHAT_API_URL=http://localhost:3003/api/v1
```

### Run tests

```bash
python -m pytest tests/ -v
```

CI excludes offline eval fixtures and one known graph test (see `.github/workflows/ci.yml`). Tests mock LLM, MongoDB, OpenSearch, and Redis — no external services required.

### Docker

```bash
docker build -t chatbot-agent .
docker run --env-file .env -p 3003:3003 chatbot-agent
```

The image runs **gunicorn** + **UvicornWorker** (2 workers by default, override with `WEB_CONCURRENCY`).

## API

### `POST /api/v1/chat`

```json
{
  "message": "Who works on machine learning in CSE?",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

**Response:** Server-Sent Events stream.

### `GET /health`

Returns `healthy` or `degraded` with per-backend checks (MongoDB, OpenSearch, Redis, embedding service).

### SSE event contract

| Event | Data | When |
|-------|------|------|
| `thinking` | `{"text": "Searching indexed publications"}` | Tool selection (user-friendly label) |
| `status` | `{"text": "Searching publications..."}` | Tool execution starts |
| `sources` | `[{citation_index, title, authors, year, ...}]` | After paper search completes |
| `chart` | chart payload | When a tool returns chartable stats |
| `token` | `{"text": "..."}` | Each token of the streamed answer |
| `done` | `{"took_ms": 1234}` | Response complete |
| `error` | `{"message": "..."}` | On failure |

## Architecture

```
Request
  → sanitize + validate
  → guardrails (greeting / identity / capability / injection → canned reply, no LLM)
  → structured router (h-index / citations / dept → MongoDB direct, no LLM)
  → LLM cache check (Redis)
  → LangGraph agent:
       agent node (Grok, temp=0, bind_tools)
         → forces ≥1 tool call (injects search_papers if LLM returns none)
       ToolNode (parallel execution, single round)
         → context budget guard (truncate per-tool, cap total)
       answer node (Grok, temp=0.3, tagged ["answer"] for SSE filtering)
  → SSE stream (astream_events v2 → thinking/status/sources/chart/token/done)
  → cache answer in Redis
```

### Key design decisions

- **Forced tool calling** — prevents hallucinated papers, faculty, or statistics
- **Single tool round** — `MAX_TOOL_ROUNDS=1` enforced via `tool_rounds` in `AgentState`
- **Two LLM instances** — `make_tool_llm()` (temp=0) for tool selection; `make_answer_llm()` (temp=0.3) for streaming answers
- **xAI Grok via OpenAI-compatible API** — `https://api.x.ai/v1`; main model `grok-4.3`, cheap `grok-3-mini` for query parsing
- **Repository layer** — tools depend on `FacultyRepository` / `ResearchRepository`; tests mock at this seam
- **Kerberos linkage** — paper→faculty attribution uses indexed `kerberos` (email prefix), not Scopus `field_associated`

## Tools

Tools are auto-discovered from `src/agent/tools/*.py` at startup.

| Tool | Primary source | Example query |
|------|----------------|---------------|
| `search_papers` | OpenSearch (BM25 + kNN) + MongoDB | "research on perovskite solar cells" |
| `find_faculty_for_topic` | search-api HTTP + MongoDB | "who works on ML" |
| `find_faculty_by_expertise` | MongoDB | "faculty with expertise in robotics" |
| `get_faculty_profile` | MongoDB (kerberos + scopus_id) | "Prof Kumar's publications" |
| `get_publication_stats` | MongoDB aggregations | "papers by Civil Engineering" |
| `get_department_profile` | MongoDB | "overview of CSE department" |
| `list_departments` | MongoDB | "list all departments" |
| `compare_faculty` | MongoDB | "compare Prof A vs Prof B" |
| `find_similar_papers` | re-embed + kNN | "papers similar to this one" |
| `get_research_trends` | MongoDB aggregation | "ML paper trends 2018–2023" |
| `find_interdisciplinary_papers` | OpenSearch + MongoDB | "cross-department work on AI" |
| `get_top_faculty` | MongoDB | "top cited faculty in EE" |

## Environment variables

See [`.env.example`](.env.example) for the full list. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3003` | HTTP listen port |
| `MONGODB_URI` | `mongodb://localhost:27017/research_db` | Shared research database |
| `OPENSEARCH_NODE` | `http://localhost:9200` | OpenSearch cluster |
| `OPENSEARCH_INDEX` | `research_documents` | Paper index name |
| `REDIS_URL` | `redis://localhost:6379` | Caches + rate limiting |
| `EMBEDDING_SERVICE_URL` | `http://localhost:8000` | BGE embed + rerank service |
| `SEARCH_API_URL` | `http://localhost:3001` | Faculty-for-topic search-api (live stack uses `:3000`) |
| `GROQ_API_KEY` | — | **Required.** xAI API key (legacy env name) |
| `GROQ_MODEL` | `grok-4.3` | Main LLM for tool selection + answers |
| `GROQ_EXTRACT_MODEL` | `grok-3-mini` | Cheap model for query parsing |
| `MAX_TOOL_ROUNDS` | `1` | Agent tool-call limit |
| `CHAT_TOP_K` | `8` | Papers retrieved per search |
| `LLM_CACHE_TTL` | `90` | Response cache TTL (seconds; `0` = off) |
| `ALLOWED_ORIGINS` | `*` | CORS origins (comma-separated or JSON array) |

## Project structure

```
chatbot-agent/
├── run.py                          # uvicorn entrypoint
├── Dockerfile                      # gunicorn + UvicornWorker production image
├── eval/                           # retrieval + E2E evaluation (see eval/README.md)
├── src/agent/
│   ├── config.py                   # pydantic-settings
│   ├── main.py                     # FastAPI app + lifespan
│   ├── api/
│   │   ├── routes_chat.py          # SSE chat endpoint
│   │   ├── routes_health.py        # /health
│   │   └── sse_events.py           # typed SSE payloads
│   ├── graph/                      # LangGraph state, nodes, builder
│   ├── llm/
│   │   ├── groq_client.py          # xAI Grok factory (tool + answer LLMs)
│   │   └── prompts.py              # system prompt
│   ├── tools/                      # auto-discovered @tool modules
│   ├── repositories/               # FacultyRepository, ResearchRepository
│   ├── rag/
│   │   ├── retriever.py            # hybrid BM25+kNN + rerank + hydrate
│   │   ├── embeddings.py           # embedding client + Redis cache
│   │   └── query_parser.py         # LLM faculty/dept extraction
│   ├── data/                       # mongo, opensearch, redis clients
│   ├── guardrails/                 # sanitize, meta-classify, injection detect
│   ├── routing/                    # structured fast-path router
│   └── services/                   # LLM cache, rate limiter
└── tests/                          # pytest suite (mocked backends)
```

## Related repositories

- [`tech-ambit-explorer`](https://github.com/IITD-Tech-Ambit/tech-ambit-explorer) — frontend that consumes this API
- [`research-ambit-main`](https://github.com/IITD-Tech-Ambit/research-ambit-main) — CMS / directory / knowledge-graph backend
- [`classification-pipeline`](https://github.com/IITD-Tech-Ambit/classification-pipeline) — paper classification for the research corpus

## License

Part of the IIT Delhi Tech Ambit / Research Ambit project.
