# Research Ambit Chatbot Agent

Agentic RAG chatbot for the [Research Ambit](https://researchambit.iitd.ac.in/) portal at IIT Delhi. Answers natural-language questions about faculty, departments, and publications by retrieving live data from MongoDB and OpenSearch, then synthesizing a streamed response with **Claude Haiku 4.5** (Anthropic) via a **LangGraph** agent.

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

- **LangGraph agent** — picks a tool, runs it, then answers grounded in the result (up to `MAX_TOOL_ROUNDS` tool rounds)
- **10 allowlisted tools** — mirroring the three portal tabs: **Explore** (paper + IP search), **Research Areas** (experts by thematic area/domain), **Directory** (a unit's faculty, and one person's full profile), plus five chart/analytics tools (trends, publication & IP stats, department profile, faculty comparison). Tool files are auto-discovered but gated by an allowlist in `_registry.py`
- **Dynamic query knobs** — the search/list tools take `sort` / `year_from` / `year_to` / `group_by_department` / `author` so the bot honors custom requirements (e.g. "most-cited papers on X since 2020 by Prof Y") instead of a single fixed ordering
- **Every browseable list capped at 10** — each list tool returns a 10-item subset and a deep-link button; the full list lives on the corresponding page
- **Grounded answers** — answer LLM runs at temperature 0 with strict anti-hallucination rules (never invent counts, entities, or rankings; say "I don't have that" instead of guessing)
- **Hybrid RAG retrieval** — BM25 + kNN over OpenSearch, BGE reranking, MongoDB hydration, kerberos/department boosts (used by the paper-search tools)
- **Fast paths** — regex guardrails (greeting/identity/injection) and structured queries (h-index, citations, faculty counts, department lists) bypass the LLM
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
- **Anthropic API key** — [console.anthropic.com](https://console.anthropic.com/) (stored in `ANTHROPIC_API_KEY`)

Connection details for MongoDB, OpenSearch, and Redis are usually shared with the rest of the stack via `opensearch/.env`.

### Install and configure

```bash
cd chatbot-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Set ANTHROPIC_API_KEY and adjust service URLs
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
       agent node (Claude Haiku 4.5, temp=0, bind_tools)
         → empty tool_calls routes straight to the answer node
       ToolNode (parallel execution, up to MAX_TOOL_ROUNDS)
         → context budget guard (truncate per-tool, cap total)
       answer node (Claude Haiku 4.5, temp=0, tagged ["answer"] for SSE filtering)
  → SSE stream (astream_events v2 → thinking/status/sources/chart/token/done)
  → cache answer in Redis
```

### Key design decisions

- **No forced tool call** — the agent node may answer directly (structural/naming questions are served from the inlined reference); empty `tool_calls` routes straight to the answer node
- **Up to 2 tool rounds** — `MAX_TOOL_ROUNDS=2` (via `tool_rounds` in `AgentState`) lets the agent chain e.g. an IPC lookup into a refined patent query; it still stops as soon as the LLM emits no further tool call
- **Two LLM instances** — `make_tool_llm()` (temp=0) for tool selection; `make_answer_llm()` (temp=0, tagged) for streaming answers
- **Claude Haiku 4.5 via the Anthropic API** — `langchain-anthropic` `ChatAnthropic`; main + query-parser model `claude-haiku-4-5-20251001` (the parser calls the raw `anthropic` SDK directly)
- **Repository layer** — tools depend on `FacultyRepository` / `ResearchRepository`; tests mock at this seam
- **Kerberos linkage** — paper→faculty attribution uses indexed `kerberos` (email prefix), not Scopus `field_associated`

## Design: inform **and** navigate

The bot is both an information assistant and a navigation aid for the portal. It
answers briefly (a headline + a preview of up to 10 items), then hands the user
off to the full page for the complete data. Its whole surface deliberately mirrors
three portal tabs — **Explore** (papers/IP), **Research Areas** (the classification
taxonomy), and **Directory** (departments/centres/schools) — so a chat answer and
the corresponding page always agree.

Two mechanisms make this work without a tool call:

- **Static structural reference** — built at startup
  ([`agent/llm/reference.py`](src/agent/llm/reference.py)) and inlined into the
  system prompt: the 9 thematic areas → their domains, and the departments /
  centres / schools **exactly as the Directory page lists them** (pulled from the
  backend's `/directory/grouped` so the sets match), with each department's
  current **Head of Department** (mirrors the Directory page's HoD map). Structural
  questions ("what themes exist", "which theme is CFD under", "list the centres",
  "who is the HoD of Civil") are answered from this — no tool, no embeddings.
- **Deep-link buttons** — when a search/experts tool runs, the answer carries a
  button that reopens the matching page (`/explore`, `/explore/ip`,
  `/research-areas`) with the **same query + filters** already applied. Other
  pages are linked inline.

## Tools

Tools are auto-discovered from `src/agent/tools/*.py`, gated by an allowlist in
[`_registry.py`](src/agent/tools/_registry.py). **10 tools** are active:

| Tool | Mirrors | What it does |
|------|---------|--------------|
| `search_research` | Explore (papers) | Same search-api + client-side transforms as the Explore page. Knobs: `year_from/to`, `sort` (relevance/citations), `group_by_department`, `author` (professor drill-down). Returns the top 10 papers **in the same order Explore shows**, plus the **People list** (`faculty.top_faculty` — top researchers by matching-paper count across the WHOLE result set; omitted on an author drill-down). Author resolution runs off the topic People sidebar (spelling-tolerant, highlightable id) with a validated directory fallback. |
| `search_ip` | ExploreIP (patents) | Advanced patent/IP search. Knobs: `year_from/to`, `type_of_ip`, `field_of_invention`, `country`, `search_in` (incl. `inventor`), `primary_inventor_only`, `sort`. Top 10 filings + related inventors. *(Only works where the `ip_documents` index exists — absent in local dev.)* |
| `experts_by_research_area` | Research Areas | Experts in a thematic area (required) → optional domain → optional department, with the area's paper/faculty counts. Top 10 experts. Wraps the `/taxonomy/*` endpoints. |
| `list_department_faculty` | Directory | Faculty of a named **department / centre / school** (`category` arg). Returns the top 10 by h-index — the Directory page's order — plus total count and a button that opens that unit's expanded list on `/directory`. |
| `get_faculty_profile` | Directory (one person) | One named faculty member's profile, assembled from the SAME endpoints the `/faculty/<kerberos>` page uses: resolves the name via `/directory/search` (first result), then returns designation, department, email, h-index, citations, research areas, Scopus/Scholar IDs, **latest 10 papers** (research-summary timeline) and **latest 10 patents/IP** (inventor-scoped IP search), each with totals. The professor's hyperlinked name IS the profile button. |
| `get_research_trends` | — | A topic's publications over years → line chart (semantic retrieval) |
| `get_publication_stats` | — | Paper counts by year / department / type → bar chart |
| `get_department_profile` | — | One department's overview + publication chart |
| `compare_faculty` | — | Two professors side by side → chart |
| `get_ip_stats` | — | Patent/IP counts by year / department / type / country / IPC → chart |

### Page coverage (what a human can do vs. the bot)

- **Explore (papers):** covered — free-text/semantic search, relevance/citations sort, year range, group-by-department, and the click-a-professor author drill-down + People list. **Minor gaps:** no `document_type` filter (Article vs Conference Paper) and no field-restricted search (title/abstract-only) — facets are returned but not applied as filters.
- **Explore (IP):** covered — topic/inventor search with year, IP type, field-of-invention, country, inventor-scope and primary-inventor filters. Live only where the IP index exists.
- **Research Areas:** covered — the Thematic Area → Domain → (department) → experts picker and the area's paper/faculty counts. **Gap:** the per-expert "papers in this research area" drill-down (`/taxonomy/faculty/:kerberos/papers`) — the bot lists experts but can't yet scope one expert's papers to a theme/domain.
- **Directory:** fully covered — unit lists (departments/centres/schools) and each department's HoD from the inlined reference; a unit's faculty via `list_department_faculty`; and one person's full profile via `get_faculty_profile`. The page's full paginated publication/patent timelines are surfaced as the latest 10 + the profile button.

The 5 chart tools return structured data the frontend auto-renders. Everything a
prior release did via ~20 finer-grained tools (faculty lookups, classification
browse, department lists, IP details) is now served by these 9 plus the static
reference and the structured fast-path (`agent/routing/structured.py`).

> **Directory coverage note:** there is not yet a consolidated *individual
> faculty profile* tool (email, designation, expertise, publications) — the
> fast-path answers h-index / citations / papers-by piecemeal, and
> `list_department_faculty` covers a unit's roster.

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
| `SEARCH_API_URL` | `http://localhost:3001` | search-api: paper/IP search, faculty-for-query, taxonomy (`search-api:3001` in Docker) |
| `BACKEND_API_URL` | `http://localhost:3002` | Main backend: Directory `/directory/grouped` (used at startup to build the department/centre/school reference) |
| `ANTHROPIC_API_KEY` | — | **Required.** Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Main LLM for tool selection + answers |
| `ANTHROPIC_EXTRACT_MODEL` | `claude-haiku-4-5-20251001` | Cheap model for query parsing |
| `MAX_TOOL_ROUNDS` | `2` | Agent tool-call rounds per query |
| `CHAT_TOP_K` | `8` | Papers retrieved per search |
| `CHAT_MAX_HISTORY_TURNS` | `5` | Recent conversation turns fed to the model (trimmed to `HISTORY_TOKEN_BUDGET`) |
| `LLM_CACHE_TTL` | `90` | Response cache TTL (seconds; `0` = off) |

> Answer LLM temperature is fixed at **0** (in `llm/anthropic_client.py`) for faithful, grounded replies — not env-configurable.
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
│   │   ├── anthropic_client.py     # Claude Haiku 4.5 factory (tool + answer LLMs)
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
- [`research-ambit-main`](https://github.com/IITD-Tech-Ambit/research-ambit-main) — CMS / directory / Atlas backend
- [`classification-pipeline`](https://github.com/IITD-Tech-Ambit/classification-pipeline) — paper classification for the research corpus

## License

Part of the IIT Delhi Tech Ambit / Research Ambit project.
