"""Load test corpus and golden sets (v3 fixtures by default)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_WORKSPACE = _ROOT.parent
_OPENSEARCH_FIXTURES = _WORKSPACE / "opensearch" / "tests" / "fixtures"

_FIXTURE_VERSION = "v3"  # v3 (default), v2, or v1 (legacy opensearch)


def set_fixture_version(version: str) -> None:
    """Select fixture set: 'v3' (default), 'v2', 'v1', or 'legacy' (alias for v1)."""
    global _FIXTURE_VERSION
    if version == "legacy":
        version = "v1"
    if version not in ("v3", "v2", "v1"):
        raise ValueError(f"Unknown fixture version: {version}")
    _FIXTURE_VERSION = version


def set_legacy_fixtures(enabled: bool = True) -> None:
    """Switch loader to opensearch/tests/fixtures (v1)."""
    set_fixture_version("v1" if enabled else "v3")


def set_legacy_v2_fixtures(enabled: bool = True) -> None:
    """Switch loader to eval v2 fixtures."""
    set_fixture_version("v2" if enabled else "v3")


def _fixture_dir() -> Path:
    return _OPENSEARCH_FIXTURES if _FIXTURE_VERSION == "v1" else _FIXTURES


def _corpus_name() -> str:
    if _FIXTURE_VERSION == "v1":
        return "test_corpus.json"
    if _FIXTURE_VERSION == "v2":
        return "corpus_v2.json"
    return "corpus_v3.json"


def _comprehensive_name() -> str:
    if _FIXTURE_VERSION == "v1":
        return "golden_set_comprehensive.json"
    if _FIXTURE_VERSION == "v2":
        return "golden_comprehensive_v2.json"
    return "golden_comprehensive_v3.json"


def _hard_name() -> str:
    if _FIXTURE_VERSION == "v1":
        return "golden_set_hard.json"
    if _FIXTURE_VERSION == "v2":
        return "golden_hard_v2.json"
    return "golden_hard_v3.json"


def fixture_path(name: str) -> Path:
    mapping = {
        "corpus_v3.json": _corpus_name(),
        "corpus_v2.json": _corpus_name(),
        "test_corpus.json": _corpus_name(),
        "golden_comprehensive_v3.json": _comprehensive_name(),
        "golden_comprehensive_v2.json": _comprehensive_name(),
        "golden_set_comprehensive.json": _comprehensive_name(),
        "golden_hard_v3.json": _hard_name(),
        "golden_hard_v2.json": _hard_name(),
        "golden_set_hard.json": _hard_name(),
    }
    return _fixture_dir() / mapping.get(name, name)


def load_json(name: str) -> dict[str, Any]:
    path = fixture_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_corpus() -> dict[str, Any]:
    return load_json(_corpus_name())


def load_golden_comprehensive() -> dict[str, Any]:
    return load_json(_comprehensive_name())


def load_golden_hard() -> dict[str, Any]:
    return load_json(_hard_name())


def corpus_by_id(corpus: dict[str, Any]) -> dict[str, dict]:
    return {doc["mongo_id"]: doc for doc in corpus.get("documents", [])}


def corpus_doc_ids(corpus: dict[str, Any]) -> set[str]:
    return {doc["mongo_id"] for doc in corpus.get("documents", [])}
