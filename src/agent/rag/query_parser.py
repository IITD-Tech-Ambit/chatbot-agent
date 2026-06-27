"""LLM-based query parser for extracting faculty name and department.

Uses a cheap xAI model (grok-3-mini) with JSON-mode to robustly identify
structured fields in natural language queries like:
  "papers by Bhim Singh from Electrical Engineering on bridge converter"
  "Apurba Das (Textile & Fibre Engineering) publications"
  "Papers from Computer Science & Engineering on what news"

Keeps an in-process LRU cache (up to 512 entries) so repeated queries are free.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from agent import metrics as _metrics

logger = logging.getLogger(__name__)

_XAI_BASE_URL = "https://api.x.ai/v1"
_MAX_CACHE = 512

_SYSTEM_PROMPT = """\
You are a query parser for the IIT Delhi research portal.
Given a user query, extract and return ONLY valid JSON with exactly these keys:
  "faculty_name": full name of an IIT Delhi faculty member if mentioned anywhere in the query, or null
  "departments": list of IIT Delhi department/centre names mentioned in the query (can be empty list [])

Rules for faculty_name:
- Extract the person's name regardless of preposition: "by NAME", "of NAME", "about NAME", "has NAME published", "NAME's papers", "papers of NAME", "work by NAME", "Prof NAME", "Professor NAME", "Dr NAME", "NAME (DEPT)".
- Strip title prefixes (Prof, Dr) — return just the name.
- If multiple names appear, return the one most likely to be an IIT Delhi faculty member.

Rules for departments:
- Return ALL department or centre names explicitly mentioned in the query.
- Extract from "from DEPT", "in DEPT", "(DEPT)", "across departments (A, B)", "DEPT and DEPT".
- IIT Delhi departments include: Electrical Engineering, Computer Science & Engineering, Mechanical Engineering, Chemical Engineering, Physics, Chemistry, Mathematics, Civil Engineering, Biochemical Engineering & Biotechnology, Textile & Fibre Engineering, Applied Mechanics, Centre for Energy Studies, Centre for Automotive Research and Tribology, Department of Management Studies, Industrial Tribology & Machine Dynamics Centre, etc.
- If two departments are named (e.g. cross-department queries), include both.

Do NOT extract:
- Research topics, keywords, or subject areas as a faculty name.
- Years, citation counts, or paper titles as names/departments.
- Values not explicitly present in the query.

If nothing found: {"faculty_name": null, "departments": []}\
"""


@dataclass(frozen=True)
class ParsedQuery:
    faculty_name: str | None
    departments: tuple[str, ...]  # immutable so the dataclass stays hashable


_NULL = ParsedQuery(faculty_name=None, departments=())


class QueryParser:
    """Async query parser backed by a cheap xAI model.

    Pass an instance to Retriever so kerberos resolution uses LLM extraction
    instead of brittle regex patterns.
    """

    def __init__(self, api_key: str, model: str = "grok-3-mini") -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=_XAI_BASE_URL)
        self._model = model
        self._cache: dict[str, ParsedQuery] = {}

    async def extract(self, query: str) -> ParsedQuery:
        """Extract faculty_name and department from query (cached)."""
        if query in self._cache:
            _metrics.CHATBOT_QUERY_PARSER_REQUESTS_TOTAL.labels(outcome="cache_hit").inc()
            return self._cache[query]

        result = await self._call(query)

        # Simple FIFO eviction when cache is full
        if len(self._cache) >= _MAX_CACHE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[query] = result
        return result

    async def _call(self, query: str) -> ParsedQuery:
        t_start = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=64,
            )
            _metrics.CHATBOT_QUERY_PARSER_REQUESTS_TOTAL.labels(outcome="success").inc()
            _metrics.CHATBOT_QUERY_PARSER_DURATION_SECONDS.observe(time.perf_counter() - t_start)
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            raw_depts = data.get("departments") or []
            if isinstance(raw_depts, str):
                raw_depts = [raw_depts]
            departments = tuple(d for d in raw_depts if d and isinstance(d, str))
            return ParsedQuery(
                faculty_name=data.get("faculty_name") or None,
                departments=departments,
            )
        except Exception as exc:
            _metrics.CHATBOT_QUERY_PARSER_REQUESTS_TOTAL.labels(outcome="error").inc()
            _metrics.CHATBOT_QUERY_PARSER_DURATION_SECONDS.observe(time.perf_counter() - t_start)
            logger.debug("QueryParser.extract failed for %r: %s", query, exc)
            return _NULL
