# Research Chatbot Agent

Production-grade agentic RAG chatbot for the IIT Delhi research portal.
Built with **LangGraph** + **FastAPI**, powered by **Llama 3.3 70B** via **Groq**.

## Features

- A forced-tool agent graph (no hallucination — always retrieves before answering)
- 7 tools: `search_papers`, `find_faculty_for_topic`, `get_faculty_profile`,
  `get_publication_stats`, `compare_faculty`, `find_similar_papers`, `get_research_trends`
- Regex guardrails (identity/capability/greeting/injection short-circuits without LLM)
- Structured-query fast paths (h-index/citations/dept lookups bypass the LLM entirely)
- Redis LLM-response cache + embedding cache
- SSE streaming compatible with the existing `tech-ambit-explorer` frontend
- 99 pytest tests (guardrails, routing, tools, graph, SSE endpoint)

## Quick Start (local)

### 1. Prerequisites

- Python 3.11+
- MongoDB, OpenSearch, Redis (see `opensearch/.env` for connection details)
- The BGE embedding service running on `localhost:8000`
- A Groq API key ([console.groq.com/keys](https://console.groq.com/keys))

### 2. Install and configure

```bash
cd chatbot-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env — set GROQ_API_KEY and adjust service URLs
```

### 3. Run

```bash
python run.py
```

The server starts on `http://localhost:3003`.

### 4. Run tests

```bash
python -m pytest tests/ -v
```

All 99 tests run without any external services (fully mocked).

## API

```
POST /api/v1/chat
Body: { "message": "...", "history": [{ "role": "user"|"assistant", "content": "..." }] }
Response: Server-Sent Events stream

GET /health
Response: JSON health check
```

### SSE Event Contract

| Event | Data | When |
|-------|------|------|
| `status` | `{"text": "Searching publications..."}` | Tool execution starts |
| `sources` | `[{citation_index, title, authors, year, ...}]` | After `search_papers` completes |
| `token` | `{"text": "..."}` | Each token of the streamed answer |
| `done` | `{"took_ms": 1234}` | Response complete |
| `error` | `{"message": "..."}` | On failure |

## Architecture

```
Request -> sanitize -> rate limit (Redis, fail-open)
       -> guardrails (greeting/identity/capability/injection -> canned reply, no LLM)
       -> structured router (h-index/citations/dept -> MongoDB direct, no LLM)
       -> LLM cache check (Redis)
       -> LangGraph agent:
            agent node (Llama 3.3 70B, temp=0, bind_tools)
              -> forces >=1 tool call (injects search_papers if LLM returns none)
            ToolNode (parallel execution, single round)
              -> context budget guard (truncate per-tool, cap total)
            answer node (Llama 3.3 70B, temp=0.3, tagged ["answer"] for SSE filtering)
       -> SSE stream (astream_events v2 -> status/sources/token/done)
       -> cache the answer in Redis
```

### Key Design Decisions

- **Forced tool calling**: The agent always calls at least one tool. If the LLM returns no
  tool_calls, a `search_papers` call is injected from the user's query. This prevents
  hallucination of papers, faculty, and statistics.
- **Single tool round**: `MAX_TOOL_ROUNDS=1` enforced via `tool_rounds` counter in `AgentState`.
- **Two LLM instances**: `make_tool_llm()` (temp=0) for deterministic tool selection,
  `make_answer_llm()` (temp=0.3, tagged `["answer"]`) for the final streamed answer.
- **Groq inference**: Llama 3.3 70B at ~394 tok/s, sub-second TTFT, 128K context window.
  No local GPU required.
- **Repository layer**: Tools depend on `FacultyRepository` / `ResearchRepository`, not the
  raw Motor client. Tests mock at the repository layer.

## Tools

| Tool | Source | Example Query |
|------|--------|---------------|
| `search_papers` | OpenSearch (kNN+BM25) + MongoDB | "research on solar cells" |
| `find_faculty_for_topic` | search-api HTTP + MongoDB | "who works on ML" |
| `get_faculty_profile` | MongoDB (dual kerberos+scopus_id) | "Prof Kumar's publications" |
| `get_publication_stats` | MongoDB aggregations | "papers by Civil Engineering" |
| `compare_faculty` | MongoDB | "compare Prof A vs Prof B" |
| `find_similar_papers` | re-embed + kNN | "papers similar to this one" |
| `get_research_trends` | MongoDB aggregation | "ML paper trends 2018-2023" |

## Project Structure

```
chatbot-agent/
├── run.py                          # uvicorn entrypoint
├── src/agent/
│   ├── config.py                   # pydantic-settings
│   ├── main.py                     # FastAPI app + lifespan
│   ├── api/
│   │   ├── routes_chat.py          # SSE chat endpoint (astream_events mapping)
│   │   ├── routes_health.py        # /health
│   │   └── schemas.py              # Pydantic request models
│   ├── graph/
│   │   ├── state.py                # AgentState (messages + tool_rounds)
│   │   ├── nodes.py                # agent, answer, route_after_agent, budget guard
│   │   └── builder.py              # StateGraph construction
│   ├── llm/
│   │   ├── groq_client.py          # make_tool_llm / make_answer_llm (ChatOpenAI → Groq)
│   │   └── prompts.py              # system prompt
│   ├── tools/                      # @tool with Pydantic args_schema
│   │   ├── _registry.py            # dependency injection for tools
│   │   ├── search_papers.py
│   │   ├── find_faculty.py
│   │   ├── faculty_profile.py
│   │   ├── publication_stats.py
│   │   ├── compare_faculty.py
│   │   ├── similar_papers.py
│   │   └── research_trends.py
│   ├── repositories/               # mock seam for tests
│   │   ├── faculty_repo.py
│   │   └── research_repo.py
│   ├── rag/
│   │   ├── retriever.py            # hybrid BM25+kNN + hydrate + truncation
│   │   └── embeddings.py           # embedding client + Redis cache
│   ├── data/                       # connection managers
│   │   ├── mongo.py
│   │   ├── opensearch.py
│   │   └── redis.py
│   ├── guardrails/
│   │   └── guardrails.py           # sanitize, classify_meta, injection detection
│   ├── routing/
│   │   └── structured.py           # fast-path regex router
│   └── services/
│       ├── cache.py                # Redis LLM-response cache
│       └── ratelimit.py            # Redis fixed-window rate limiter
└── tests/                          # 99 tests, fully mocked
    ├── conftest.py                 # fake LLMs, repos, Redis
    ├── test_guardrails.py
    ├── test_structured_routing.py
    ├── test_tools.py
    ├── test_agent_graph.py
    └── test_chat_endpoint.py
```

## Dependencies

- MongoDB, OpenSearch, Redis, the BGE embedding service
- Groq API key (free tier available)
- See `.env.example` for all configuration
