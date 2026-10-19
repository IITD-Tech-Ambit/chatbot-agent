# Research Ambit Chatbot Agent

Conversational research assistant over IIT Delhi research data: papers, patents and other IP, faculty, departments, centres, schools, and the research-area taxonomy. The service is a FastAPI application. Answers are produced by a LangGraph agent (LangChain tools) that calls Claude Haiku 4.5 through the Anthropic API (`ANTHROPIC_MODEL`, default `claude-haiku-4-5-20251001`).

The HTTP service is implemented and starts on its own. The Research Ambit web app (`tech-ambit-explorer`) does not expose it until `VITE_CHATBOT_ENABLED=true`.

## System design

### Role

The agent answers questions about the IIT Delhi research corpus and points at the matching portal page. List tools return a capped preview (10 items for papers, IP filings, experts, a unit's faculty, and a profile's latest papers and patents). A static structure reference, built once at startup, answers naming questions (thematic areas, domains, departments, centres, schools, and each unit's head) without a tool call.

### Request path

`POST /api/v1/chat` accepts JSON:

```json
{
  "message": "Who works on machine learning in CSE?",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

`message` is 1–2000 characters. `history` is optional, at most 12 turns, each `content` at most 4000 characters, roles `user` or `assistant`. The optional body field `user_id` is not used for identity.

Identity comes from the API gateway:

- `GATEWAY_SHARED_SECRET` empty: `x-user-id`, `x-user-kerberos`, and `x-user-category` are trusted (local development).
- Secret set: the request counts as gateway-originated only when `x-gateway-secret` matches that secret via `hmac.compare_digest`. A mismatch clears the trusted identity.
- `require_user_id` then requires a non-empty `x-user-id`. Otherwise the handler returns HTTP 401.

`GET /api/v1/quota` uses the same identity check and returns the caller's daily quota (or `{"unlimited": true}`).

`GET /health` reports `healthy` or `degraded` with boolean checks for MongoDB, OpenSearch, Redis, and the embedding client. `GET /metrics` exposes Prometheus metrics.

### Pipeline

`handle_chat` in `src/agent/api/chat_pipeline.py` runs, in order:

1. **Sanitize.** Strip control characters and truncate to `CHAT_MAX_MESSAGE_LENGTH` (2000). An empty result is HTTP 400.
2. **Guardrails.** Regex classification (`greeting`, `identity`, `capabilities`, prompt-reveal, injection, off-topic) returns a canned SSE reply. No quota increment and no LLM call.
3. **Injection block.** `detect_injection` returns the canned refusal if the first classifier did not already short-circuit.
4. **Quota.** Unless the caller is exempt, Redis increments `chat:quota:{user_id}:{YYYY-MM-DD}` (date in `Asia/Kolkata`). Over `CHAT_QUOTA_DAILY` (default 5) the handler returns HTTP 429 JSON, not an SSE stream. A Redis error fails open.
5. **Structured router.** Regex matches for h-index, citation count, top faculty by h-index or citations, papers by a named author, a department list, and total faculty count run against MongoDB through `FacultyRepository` / `ResearchRepository`. No LLM. A successful answer is written to the LLM cache.
6. **LLM response cache.** Redis key `llm:v2:{sha256}` of the normalized message text (TTL `LLM_CACHE_TTL`, default 90 seconds; `0` disables). A hit replays the stored answer, sources, chart, and explore payload.
7. **LangGraph agent.** Otherwise the graph runs and the answer is cached when non-empty.

Graph (`src/agent/graph/builder.py`):

```
agent (tool-selection LLM, tools bound, temperature 0)
  → tool_calls present → ToolNode → bump tool_rounds → agent
  → no tool_calls      → answer LLM (temperature 0, tag "answer") → END
```

`MAX_TOOL_ROUNDS` (default 2) is checked at the start of the agent node. At the cap the node emits an empty assistant message with no tool calls, which routes to the answer node. An empty tool-call list on an earlier round also routes straight to the answer node. The answer node drops that tool-selection message and regenerates the reply. Tool outputs are truncated to per-tool character caps; if the combined context exceeds `CONTEXT_ANSWER_RESERVE * 4` characters, later tool messages are replaced with `{"truncated": true}`.

Two `ChatAnthropic` clients share `ANTHROPIC_MODEL` and `MAX_ANSWER_TOKENS` (default 1024). Temperature is fixed at 0 in `src/agent/llm/anthropic_client.py`. `LLM_HTTP_PROXY_URL`, when set, is passed only as `anthropic_proxy`.

Retrieval inside `get_research_trends` and `get_publication_stats` uses `Retriever` (`src/agent/rag/retriever.py`):

- `QueryParser` calls the Anthropic Messages API (`ANTHROPIC_EXTRACT_MODEL`, temperature 0, max 128 tokens) to extract a faculty name and department names. Results sit in an in-process map of at most 512 entries.
- Those names resolve to kerberos ids (email local-part). A non-empty set is a hard OpenSearch `terms` filter on `kerberos`, with BM25 (`most_fields` over title and abstract) plus kNN inside the filter. Zero hits fall back to the unfiltered hybrid query. The retriever does not query `field_associated`.
- Query vectors come from `EmbeddingClient` (Redis cache, TTL `EMBEDDING_CACHE_TTL`, default 86400 seconds). After MongoDB hydration, the BGE cross-encoder reranker reorders the hits. A rerank failure returns the unreranked list.

`search_research`, `search_ip`, `get_faculty_profile` (patents), and `experts_by_research_area` do not use this retriever. They call search-api over HTTP at `SEARCH_API_URL`. Directory reads go to `BACKEND_API_URL`. `get_ip_stats`, `get_department_profile`, `compare_faculty`, and the structured router read MongoDB.

Startup (`src/agent/main.py`) refuses to boot when `ANTHROPIC_API_KEY` is empty. It connects MongoDB, OpenSearch, and Redis, builds the structure reference, constructs the embedding and faculty-search transports, builds the ten registered tools, compiles the graph, and, when `GRPC_ENABLED` is true, listens for `chat.v1`.

### gRPC

`MESH_TRANSPORT` selects the east-west client (`http` by default, `grpc` in the mesh). The gRPC target is `ENVOY_GRPC_TARGET` (default `envoy:10000`), an insecure channel.

| Stub | Used for |
|------|----------|
| `embedding.v1.EmbeddingService` | `Embed` and `Rerank` on `GrpcEmbeddingTransport` when `MESH_TRANSPORT=grpc`. `EmbeddingClient` uses these for query vectors and BGE cross-encoder reranking inside `Retriever`. Health probes `Embed` with the text `ping`. |
| `embedding.v1` over HTTP | When `MESH_TRANSPORT` is not `grpc`: `POST {EMBEDDING_SERVICE_URL}/embed` and `POST /rerank`. A 404 on `/rerank` is treated as "no reranker". |
| `search.v1.SearchService.FacultyForQuery` | `GrpcFacultySearchClient` when `MESH_TRANSPORT=grpc` (HTTP `GET /api/v1/search/faculty-for-query` otherwise). The client is injected as `ToolDeps.search_client`. The only caller is `find_faculty`, which is not in `ENABLED_TOOL_MODULES`, so registered tools do not invoke this stub. |
| Registered search and taxonomy calls | HTTP to `SEARCH_API_URL` regardless of `MESH_TRANSPORT`: `POST /api/v1/search`, `POST /api/v1/search/author-scope`, `GET /api/v1/search/faculty-for-query`, `POST /api/v1/ip/search`, and `GET /api/v1/taxonomy/*`. Those clients set `trust_env=False`. |

Generated `search.v1` methods other than `FacultyForQuery` (including `TaxonomyService`) are present as stubs and are not called by this process.

This process also **serves** `chat.v1.ChatService.CheckQuota` on `GRPC_PORT` (default 50054) when `GRPC_ENABLED` is true. `CheckQuota` peeks the Redis quota for `user_id` and does not consume a message.

### CORS

`ALLOWED_ORIGINS` defaults to an empty list. A string value is parsed as a JSON array when that parse succeeds, otherwise as a comma-separated list. `CORSMiddleware` allows those origins, methods `GET`, `POST`, and `OPTIONS`, and all request headers. `allow_credentials` is true except when the allow-list is exactly `["*"]`. An empty allow-list adds no `Access-Control-Allow-Origin` entry.

### Session and memory

There is no server-side conversation store. The client sends `history`. The agent keeps the last `CHAT_MAX_HISTORY_TURNS` turns (Settings default 5; `.env.example` sets 6), then drops older turns until the remainder fits `HISTORY_TOKEN_BUDGET` (default 800, counted as characters / 4).

The LLM response cache is keyed only on the current message text. It does not include `history` or `user_id`. Embedding vectors and the IPC label cache are also in Redis. The query-parser cache is process-local and is not shared across workers.

### Boundaries

Coded limits:

- Scope is IIT Delhi research, publications, faculty, and patents. Greeting, identity, capability, prompt-reveal, injection, and off-topic patterns (code generation, homework, general knowledge, and role-play among them) receive a fixed reply.
- Untrusted or missing `x-user-id` is HTTP 401. Categories matching `CHAT_QUOTA_LIMITED_CATEGORIES` (default substring `student`) share a daily cap. Other non-empty categories are exempt. An empty category is counted against the cap. `CHAT_QUOTA_WHITELIST_KERBEROS` is exempt. `ENABLE_AUTH=false` exempts every caller from the quota; it does not skip the `x-user-id` check.
- Only the modules in `ENABLED_TOOL_MODULES` (`src/agent/tools/_registry.py`) are registered. Other files under `src/agent/tools/` stay on disk and are not loaded. That includes `lookup_ipc_classification`, `find_faculty`, `search_papers`, and `search_ips`. `IpRetriever` is constructed at startup and is only referenced by the unregistered `search_ips` tool.
- `search_research` accepts `query`, `year_from`, `year_to`, `sort` (`relevance` or `citations`), `group_by_department`, and `author`. It has no department, document-type, or field-restricted search arguments. Facet buckets are returned as a summary.
- `experts_by_research_area` calls taxonomy faculty, counts, themes, domains, and departments. It does not call a per-faculty papers-in-theme endpoint.
- The system prompt states that IITD Verse, Magazines, and Contributors have no tools: the model may describe those pages and must not list magazine issues, name contributors, or query the atlas. It also states that institute Director and Dean names are absent. Unit heads in the prompt come from the hand-maintained `_UNIT_HEADS` map. Research labs and the directory "Other" bucket are omitted from the reference.
- Prompt rules require tool-grounded facts and forbid invented counts. Temperature 0 and the context cap are the code-side controls; they do not validate the generated text.
- `get_ip_stats` labels IPC classifications through `IpcClassificationService` (static table, Redis, then WIPO HTTP at `IPC_WIPO_API_URL` on a cache miss). The separate IPC lookup tool is not registered, so the agent cannot chain a lookup tool into a later search round. Two tool rounds remain available for any registered tools the model chooses to call.

### Registered tools

| Tool | Module | Data source |
|------|--------|-------------|
| `search_research` | `search_research` | search-api paper search, author-scope, and faculty-for-query. Returns 10 papers from a 20-hit page, plus up to 10 people. |
| `search_ip` | `search_ip` | search-api `POST /api/v1/ip/search`. Up to 10 filings. Filters: year, IP type, field of invention, country, `search_in`, primary inventor, sort. |
| `experts_by_research_area` | `experts_by_research_area` | search-api taxonomy. Theme required; domain and department optional. At most 10 experts. |
| `list_department_faculty` | `list_department_faculty` | Backend directory API. Top 10 by h-index for a department, centre, or school. |
| `get_faculty_profile` | `faculty_profile` | Directory search and profile, research-summary papers, inventor-scoped IP search. Latest 10 papers and 10 filings. Name resolution uses the first directory hit. |
| `get_research_trends` | `research_trends` | `Retriever` (top 400) then MongoDB year counts. |
| `get_publication_stats` | `publication_stats` | `Retriever` when a topic is set; otherwise MongoDB aggregations. Group by department, year, or document type. |
| `get_department_profile` | `department_profile` | MongoDB department overview, top faculty, publication chart. |
| `compare_faculty` | `compare_faculty` | MongoDB side-by-side h-index, citations, and paper counts for two names. |
| `get_ip_stats` | `get_ip_stats` | MongoDB IP aggregations (year, department, type, country, classification, inventor) with IPC labels. |

On `on_tool_end`, paper and IP tools emit a `sources` event (once per request). Chart tools emit `chart` via `chart_builder`. `search_research`, `search_ip`, `experts_by_research_area`, and `list_department_faculty` can emit an `explore` deep-link payload (once per request).

### SSE contract

`POST /api/v1/chat` responds with `text/event-stream`.

| Event | Payload | When |
|-------|---------|------|
| `thinking` | `{"step": "...", "detail": null}` | Structured lookup, or a tool start (`step` is that tool's thinking label) |
| `sources` | list of paper or IP source objects | First paper or IP tool result that has items |
| `chart` | `{tool_name, chart}` | A chart-producing tool result |
| `explore` | deep-link object (`kind` of `research`, `ip`, `research_area`, or `directory`) | First matching search, experts, or department-faculty result |
| `token` | `{"text": "..."}` | Canned reply (one event), cached answer (one event), or each streamed answer token |
| `done` | `{"took_ms": N, "cached": false}` | Stream finished. `cached` is true on a cache hit |
| `error` | `{"message": "..."}` | Structured-path or agent-stream exception |

`StatusEvent` exists in `sse_events.py` and is not emitted. Quota rejection is HTTP 429 JSON, not an SSE event.

## Quick start

### Prerequisites

- Python 3.11+ (the Docker image is Python 3.13)
- MongoDB database `research_db` (`MONGODB_URI`)
- OpenSearch paper index `research_documents` (`OPENSEARCH_INDEX`). `OPENSEARCH_IP_INDEX` (`ip_documents`) is opened for `IpRetriever`, which no registered tool calls
- Redis (`REDIS_URL`) for the LLM cache, embedding cache, daily quota, and IPC label cache
- With `MESH_TRANSPORT=http` (the default): embedding service at `EMBEDDING_SERVICE_URL` (default `http://localhost:8000`) and search-api at `SEARCH_API_URL` (default `http://localhost:3001`)
- Backend directory API at `BACKEND_API_URL` (default `http://localhost:3002`), used at startup for the unit list and by the directory tools
- `ANTHROPIC_API_KEY`. The process raises `RuntimeError` at startup if it is unset

`MESH_TRANSPORT=grpc` sends embedding calls, and the unused faculty-for-query client, to `ENVOY_GRPC_TARGET`. Registered search-api calls stay on HTTP.

### Install and run

```bash
cd chatbot-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Set ANTHROPIC_API_KEY and service URLs
python run.py
```

`run.py` starts uvicorn at `http://localhost:3003` (override with `PORT` / `HOST`). Health: `GET /health`.

### Tests

```bash
python -m pytest tests/ -v
```

Tests mock the LLM, MongoDB, OpenSearch, and Redis. CI (`.github/workflows/ci.yml`) installs `requirements.txt` and ignores `tests/test_eval_offline.py` (its corpus fixture is gitignored). The live retrieval class in `tests/test_retrieval_quality.py` runs only when `RUN_E2E` is set. Offline and live benchmarks live under `eval/` (`eval/README.md`).

### Docker

```bash
docker build -t chatbot-agent .
docker run --env-file .env -p 3003:3003 chatbot-agent
```

The image runs gunicorn with `uvicorn.workers.UvicornWorker`. Worker count is `WEB_CONCURRENCY` (default 2). The image listens on port 3003.

## Environment variables

Settings are `pydantic-settings` fields in `src/agent/config.py` (`.env`, extra keys ignored). Defaults below are the Settings defaults. See [`.env.example`](.env.example) for a template. `.env.example` sets `CHAT_MAX_HISTORY_TURNS=6`; the code default is 5.

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3003` | HTTP listen port (`run.py`). The Docker process binds 3003 directly |
| `HOST` | `0.0.0.0` | Bind address for `run.py` |
| `ENABLE_AUTH` | `true` | When `false`, every caller is quota-exempt. Does not disable the `x-user-id` check |
| `GATEWAY_SHARED_SECRET` | empty | Shared with the API gateway. Empty trusts `x-user-*`. When set, `x-gateway-secret` must match (`hmac.compare_digest`) |
| `ALLOWED_ORIGINS` | empty | CORS allow-list: JSON array or comma-separated origins. `["*"]` disables credentials |
| `MONGODB_URI` | `mongodb://localhost:27017/research_db` | Research database |
| `OPENSEARCH_NODE` | `http://localhost:9200` | OpenSearch |
| `OPENSEARCH_INDEX` | `research_documents` | Paper index used by `Retriever` |
| `OPENSEARCH_IP_INDEX` | `ip_documents` | IP index used by the unregistered `IpRetriever` |
| `REDIS_URL` | `redis://localhost:6379` | LLM cache, embedding cache, quota, IPC cache |
| `MESH_TRANSPORT` | `http` | `grpc` uses Envoy for the embedding client (and the unused faculty-search client); anything else uses HTTP |
| `ENVOY_GRPC_TARGET` | `envoy:10000` | gRPC target for those clients |
| `GRPC_ENABLED` | `true` | Serve `chat.v1.CheckQuota` |
| `GRPC_PORT` | `50054` | That listener's port |
| `EMBEDDING_SERVICE_URL` | `http://localhost:8000` | HTTP embed and rerank base URL |
| `EMBEDDING_TIMEOUT_MS` | `10000` | Embedding and rerank timeout |
| `EMBEDDING_CACHE_TTL` | `86400` | Redis TTL for query embeddings (seconds) |
| `SEARCH_API_URL` | `http://localhost:3001` | search-api HTTP base URL |
| `BACKEND_API_URL` | `http://localhost:3002` | Directory API (`/api/directory/grouped` and faculty profile routes) |
| `CHAT_QUOTA_DAILY` | `5` | Messages per IST day for limited categories |
| `CHAT_QUOTA_LIMITED_CATEGORIES` | `student` | Comma-separated substrings matched against `x-user-category` |
| `CHAT_QUOTA_WHITELIST_KERBEROS` | empty | Comma-separated kerberos ids that skip the quota |
| `ANTHROPIC_API_KEY` | empty | Required at startup |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Tool-selection and answer model |
| `ANTHROPIC_EXTRACT_MODEL` | `claude-haiku-4-5-20251001` | Query-parser model |
| `MAX_ANSWER_TOKENS` | `1024` | `max_tokens` for both chat LLMs. Temperature is fixed at 0 |
| `LLM_HTTP_PROXY_URL` | empty | Proxy for Anthropic, the query parser, and the WIPO IPC client |
| `MAX_TOOL_ROUNDS` | `2` | Tool rounds before the graph forces the answer node |
| `CHAT_TOP_K` | `8` | Default `Retriever` size. Trend and publication-stat tools pass `top_k=400` |
| `CHAT_MAX_HISTORY_TURNS` | `5` | History turns passed into the graph |
| `CHAT_MAX_MESSAGE_LENGTH` | `2000` | Sanitize truncate length (also the schema max) |
| `HISTORY_TOKEN_BUDGET` | `800` | History character budget is this value times 4 |
| `LLM_CACHE_TTL` | `90` | Response cache TTL in seconds; `0` disables it |
| `IPC_WIPO_API_URL` | `https://ipcpub.wipo.int/rest-services/ipc` | IPC scheme fetch on cache miss |
| `IPC_LOOKUP_TIMEOUT_MS` | `4000` | That HTTP timeout |
| `IPC_CACHE_TTL` | `2592000` | Redis TTL for IPC entries (seconds) |

`RATE_LIMIT_WINDOW_SEC` and `RATE_LIMIT_MAX_REQUESTS` appear in `.env.example` and are not Settings fields (`extra="ignore"`).

## Project structure

```
chatbot-agent/
├── run.py                          # local uvicorn entrypoint
├── Dockerfile                      # gunicorn + UvicornWorker
├── proto/chat/v1/chat.proto
├── eval/                           # retrieval and end-to-end evaluation
├── src/agent/
│   ├── config.py                   # Settings
│   ├── main.py                     # FastAPI app, lifespan, CORS
│   ├── grpc_server.py              # chat.v1 CheckQuota server
│   ├── api/
│   │   ├── routes_chat.py          # POST /chat, GET /quota
│   │   ├── chat_pipeline.py        # guardrails, quota, cache, SSE
│   │   ├── routes_health.py
│   │   └── sse_events.py
│   ├── graph/                      # LangGraph state, nodes, builder
│   ├── llm/                        # Anthropic clients, prompt, structure reference
│   ├── tools/                      # tool modules; allowlist in _registry.py
│   ├── transports/                 # HTTP and gRPC adapters
│   ├── repositories/               # MongoDB faculty, research, IP, taxonomy
│   ├── rag/                        # retriever, embeddings, query parser, IP retriever
│   ├── data/                       # mongo, opensearch, redis clients
│   ├── guardrails/
│   ├── routing/structured.py
│   └── services/                   # LLM cache, quota, IPC lookup
├── src/embedding/v1/               # generated embedding.v1 stubs
├── src/search/v1/                  # generated search.v1 stubs
├── src/chat/v1/                    # generated chat.v1 stubs
└── tests/
```

## Related repositories

- [`tech-ambit-explorer`](https://github.com/IITD-Tech-Ambit/tech-ambit-explorer) — web app; chat stays off unless `VITE_CHATBOT_ENABLED=true`
- [`research-ambit-main`](https://github.com/IITD-Tech-Ambit/research-ambit-main) — directory and CMS backend (`BACKEND_API_URL`)
- [`classification-pipeline`](https://github.com/IITD-Tech-Ambit/classification-pipeline) — paper classification for the corpus

## License

Part of the IIT Delhi Tech Ambit / Research Ambit project.
