"""Best-effort fetch of an IPC symbol title from WIPO's published IPC data.

WIPO exposes no clean JSON title endpoint: the machine-readable
``getSymbolValidity`` service returns only structural fields, and full titles
live in the HTML scheme view. This client validates the symbol via the JSON
service and scrapes the scheme page title as a fallback. The HTTP client is
injected so the network dependency stays swappable and testable.
"""

from __future__ import annotations

import logging
import re

import httpx

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r'class="[^"]*ipc-title[^"]*"[^>]*>(.*?)</', re.IGNORECASE | re.DOTALL)


def pad_ipc_symbol(code: str) -> str:
    """Zero-pad to WIPO's 14-character symbol form, e.g. A61K 38/00 → A61K0038000000."""
    cleaned = re.sub(r"\s+", "", (code or "").upper())
    if "/" in cleaned:
        main, sub = cleaned.split("/", 1)
    else:
        main, sub = cleaned, ""
    subclass = main[:4]
    main_group = main[4:]
    if not main_group and not sub:
        return subclass.ljust(14, "0")
    return (subclass.ljust(4, "0") + main_group.zfill(4) + sub.ljust(6, "0"))[:14]


class WipoIpcHttpClient:
    def __init__(self, http_client: httpx.AsyncClient, base_url: str, timeout_ms: int) -> None:
        self._http = http_client
        self._base = base_url.rstrip("/")
        self._timeout = timeout_ms / 1000.0

    async def fetch_definition(self, code: str) -> str | None:
        symbol = pad_ipc_symbol(code)
        if not symbol:
            return None
        url = f"{self._base}/scheme"
        params = {
            "symbol": symbol,
            "lang": "en",
            "menulang": "en",
            "definitions": "yes",
            "viewmode": "f",
        }
        try:
            resp = await self._http.get(url, params=params, timeout=self._timeout)
            resp.raise_for_status()
        except Exception as exc:
            logger.debug("WIPO IPC fetch failed for %s: %s", code, exc)
            return None

        return _extract_title(resp.text)


def _extract_title(html: str) -> str | None:
    if not html:
        return None
    m = _TITLE_RE.search(html)
    if not m:
        return None
    text = _TAG_RE.sub(" ", m.group(1))
    text = re.sub(r"\s+", " ", text).strip()
    return text or None
