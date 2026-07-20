"""Taxonomy / classification repository — async methods over MongoDB.

Reads the two-axis paper classification (thematic area + domain) that powers the
Explore → Browse taxonomy. Mirrors ip_repo.py / research_repo.py: tools depend on
this layer (the mock seam), never the raw Motor client.

Data model (see SEO-Backend-iitd taxonomy models):
  - Each paper (researchmetadatascopus) carries a single-label
      classification.{thematic_area_id, domain_id, subdomain_id, topics[]}
    plus resolved iitd_authors[].{kerberos, department_ref}.
  - `thematicareas` (9 fixed themes) and `domains` (independent axis; domains are
    NOT children of themes) hold name/slug + rollup `stats`.
  - The v2.2 taxonomy is TWO-LEVEL — there is no subdomain layer populated, so
    this repo intentionally ignores subdomains.
  - `taxonomyfacetcounts` / `taxonomyfacetmembers` are precomputed rollup cubes,
    keyed by the nullable 4-tuple (thematic_area_id, domain_id, subdomain_id,
    department_id); null means "not part of this configuration".
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


def _as_object_id(value: Any) -> ObjectId | None:
    if isinstance(value, ObjectId):
        return value
    if isinstance(value, str) and ObjectId.is_valid(value):
        return ObjectId(value)
    return None


class TaxonomyRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._papers = db["researchmetadatascopus"]
        self._themes = db["thematicareas"]
        self._domains = db["domains"]
        self._counts = db["taxonomyfacetcounts"]
        self._members = db["taxonomyfacetmembers"]

    # ---- catalog ---------------------------------------------------------

    async def all_themes(self) -> list[dict[str, Any]]:
        cursor = self._themes.find(
            {}, {"name": 1, "slug": 1, "display_order": 1, "stats": 1}
        ).sort("display_order", 1)
        return await cursor.to_list(length=100)

    async def all_domains(self) -> list[dict[str, Any]]:
        cursor = self._domains.find(
            {}, {"name": 1, "slug": 1, "display_order": 1, "stats": 1}
        ).sort("display_order", 1)
        return await cursor.to_list(length=500)

    _STOPWORDS = {
        "and", "the", "of", "for", "in", "on", "to", "with", "a", "an",
        "research", "area", "areas", "theme", "themes", "domain", "domains",
        "field", "fields", "technology", "technologies", "science", "sciences",
    }

    async def _resolve(self, coll, term: str | None) -> dict[str, Any] | None:
        """Resolve a theme/domain from an approximate name the LLM supplies.

        The LLM rarely knows the exact catalog name, so match in three tiers:
        exact slug/name → substring → token overlap (the candidate whose name
        shares the most significant words with the query wins)."""
        if not term or not str(term).strip():
            return None
        t = str(term).strip()
        exact = re.compile(f"^{re.escape(t)}$", re.IGNORECASE)
        doc = await coll.find_one({"$or": [{"slug": exact}, {"name": exact}]})
        if doc:
            return doc
        rx = re.compile(re.escape(t), re.IGNORECASE)
        doc = await coll.find_one({"$or": [{"slug": rx}, {"name": rx}]})
        if doc:
            return doc

        tokens = [
            w for w in re.split(r"[^a-z0-9]+", t.lower())
            if w and w not in self._STOPWORDS and len(w) > 1
        ]
        if not tokens:
            return None
        candidates = await coll.find({}, {"name": 1, "slug": 1}).to_list(length=500)
        best, best_score = None, 0
        for c in candidates:
            name_l = (c.get("name") or "").lower()
            slug_l = (c.get("slug") or "").lower().replace("-", " ")
            score = sum(1 for tok in tokens if tok in name_l or tok in slug_l)
            if score > best_score:
                best, best_score = c, score
        return best if best_score >= 1 else None

    async def resolve_theme(self, term: str | None) -> dict[str, Any] | None:
        return await self._resolve(self._themes, term)

    async def resolve_domain(self, term: str | None) -> dict[str, Any] | None:
        return await self._resolve(self._domains, term)

    async def name_maps(self) -> tuple[dict[str, str], dict[str, str]]:
        """{theme_id_str: name}, {domain_id_str: name} for labelling aggregations."""
        themes, domains = await self.all_themes(), await self.all_domains()
        return (
            {str(t["_id"]): t.get("name", "") for t in themes},
            {str(d["_id"]): d.get("name", "") for d in domains},
        )

    # ---- facet cube (precomputed rollups) --------------------------------

    async def theme_counts_for_department(self, department_id: ObjectId) -> list[dict[str, Any]]:
        cursor = self._counts.find({
            "thematic_area_id": {"$ne": None},
            "domain_id": None,
            "subdomain_id": None,
            "department_id": department_id,
        })
        return await cursor.to_list(length=100)

    async def domain_counts(
        self, theme_id: ObjectId | None = None, department_id: ObjectId | None = None
    ) -> list[dict[str, Any]]:
        # theme_id None selects the domain-axis-alone rows (thematic_area_id null),
        # i.e. each domain's counts independent of any theme.
        cursor = self._counts.find({
            "thematic_area_id": theme_id,
            "domain_id": {"$ne": None},
            "subdomain_id": None,
            "department_id": department_id,
        })
        return await cursor.to_list(length=500)

    async def config_counts(
        self,
        theme_id: ObjectId | None = None,
        domain_id: ObjectId | None = None,
        department_id: ObjectId | None = None,
    ) -> dict[str, Any] | None:
        return await self._counts.find_one({
            "thematic_area_id": theme_id,
            "domain_id": domain_id,
            "subdomain_id": None,
            "department_id": department_id,
        })

    async def config_members(
        self,
        theme_id: ObjectId | None = None,
        domain_id: ObjectId | None = None,
        department_id: ObjectId | None = None,
    ) -> dict[str, Any] | None:
        return await self._members.find_one({
            "thematic_area_id": theme_id,
            "domain_id": domain_id,
            "subdomain_id": None,
            "department_id": department_id,
        })

    # ---- live paper queries ----------------------------------------------

    def _classification_match(
        self,
        theme_id: ObjectId | None = None,
        domain_id: ObjectId | None = None,
        department_ref: ObjectId | None = None,
        kerberos: str | None = None,
    ) -> dict[str, Any]:
        match: dict[str, Any] = {}
        if theme_id is not None:
            match["classification.thematic_area_id"] = theme_id
        if domain_id is not None:
            match["classification.domain_id"] = domain_id
        if department_ref is not None:
            match["iitd_authors.department_ref"] = department_ref
        if kerberos:
            match["iitd_authors.kerberos"] = kerberos.lower().strip()
        return match

    async def papers_in_context(
        self,
        theme_id: ObjectId | None = None,
        domain_id: ObjectId | None = None,
        department_ref: ObjectId | None = None,
        kerberos: str | None = None,
        limit: int = 10,
    ) -> tuple[list[dict[str, Any]], int]:
        match = self._classification_match(theme_id, domain_id, department_ref, kerberos)
        if not match:
            return [], 0
        projection = {
            "title": 1, "abstract": 1, "link": 1, "publication_year": 1,
            "document_type": 1, "citation_count": 1, "document_scopus_id": 1,
            "document_eid": 1, "classification.topics": 1,
        }
        cursor = (
            self._papers.find(match, projection)
            .sort([("publication_year", -1), ("citation_count", -1)])
            .limit(limit)
        )
        items = await cursor.to_list(length=limit)
        total = await self._papers.count_documents(match)
        return items, total

    async def _distribution(
        self, group_field: str, match: dict[str, Any], limit: int = 100
    ) -> list[dict[str, Any]]:
        pipeline = [
            {"$match": {**match, group_field: {"$ne": None}}},
            {"$group": {"_id": f"${group_field}", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        cursor = self._papers.aggregate(pipeline)
        return await cursor.to_list(length=limit)

    async def theme_distribution(
        self,
        department_ref: ObjectId | None = None,
        kerberos: str | None = None,
    ) -> list[dict[str, Any]]:
        match = self._classification_match(
            department_ref=department_ref, kerberos=kerberos
        )
        return await self._distribution("classification.thematic_area_id", match)

    async def domain_distribution(
        self,
        theme_id: ObjectId | None = None,
        department_ref: ObjectId | None = None,
        kerberos: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        match = self._classification_match(
            theme_id=theme_id, department_ref=department_ref, kerberos=kerberos
        )
        return await self._distribution("classification.domain_id", match, limit=limit)

    async def count_in_context(
        self,
        theme_id: ObjectId | None = None,
        domain_id: ObjectId | None = None,
        department_ref: ObjectId | None = None,
        kerberos: str | None = None,
    ) -> int:
        match = self._classification_match(theme_id, domain_id, department_ref, kerberos)
        if not match:
            return 0
        return await self._papers.count_documents(match)
