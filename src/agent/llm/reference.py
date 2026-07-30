"""Build the fixed IIT Delhi structural reference injected into the system prompt.

This is small, slow-changing data (9 thematic areas, their ~80 domains, and the
departments/centres/schools) — so instead of a tool call or an embedding lookup,
we generate it once at startup from MongoDB and inline the whole thing into the
system prompt. The bot answers structural/naming questions ("what themes exist",
"which theme is domain X under", "list the centres") directly from it, with no
tool call. Anything dynamic (faculty, papers, patents, counts) still uses tools.

Generated from the DB (never hand-maintained) so it can't drift from the data.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import httpx

logger = logging.getLogger(__name__)

# Only these three — they mirror the Directory page's category tabs exactly.
# Research labs and the misc "Other" bucket (Administration, Old Phd Data, …)
# are deliberately excluded: they are not academic units the bot should list.
_DIRECTORY_CATEGORIES = [("departments", "Departments"), ("centres", "Centres"), ("schools", "Schools")]

# Fallback only: raw departments.category values, used if the backend is
# unreachable at startup.
_FALLBACK_CATEGORY = {"Departments": "Department", "Centres": "Centre", "Schools": "School"}

# Current head of each academic unit — departments (labelled "HoD") AND
# centres/schools (labelled "Head") — with the head's kerberos for a profile
# link. Mirrors the Directory page's hardcoded DEPT_HODS map
# (tech-ambit-explorer DepartmentGroupAccordionItem.tsx); not in the DB (the
# /directory/grouped payload carries no head field), so kept in sync by hand.
_UNIT_HEADS: dict[str, tuple[str, str]] = {
    # Departments
    "Applied Mechanics": ("Sawan S. Sinha", "sawan"),
    "Biochemical Engineering & Biotechnology": ("Preeti Srivastava", "preeti"),
    "Chemical Engineering": ("Anurag S. Rathore", "arathore"),
    "Chemistry Department": ("S. Nagendran", "sisn"),
    "Civil Engineering": ("Vasant Matsagar", "matsagar"),
    "Computer Science & Engineering": ("Naveen Garg", "naveen"),
    "Department of Design": ("Sumer Singh", "sumer"),
    "Department of Energy Science & Engineering": ("Ramesh Narayanan", "rams"),
    "Department of Management Studies": ("Surya Prakash Singh", "sprsingh"),
    "Electrical Engineering": ("Shankar Prakriya", ""),
    "Humanities & Social Sciences": ("Abhijit Banerji", "a_banerji"),
    "Materials Science & Engineering": ("Leena Nebhani", "lnebhani"),
    "Mathematics Department": ("Mani Mehra", "mmehra"),
    "Mechanical Engineering": ("Subbarao P M V", "pmvs"),
    "Physics Department": ("Sujeet Chaudhary", "sujeetc"),
    "Textile & Fibre Engineering": ("Deepti Gupta", "deepti"),
    # Schools
    "Amar Nath and Shashi Khosla School of Information Technology": ("Srikanta Bedathur", "srikanta"),
    "Bharti School of Telecommunication Technology and Management": ("Manav R. Bhatnagar", "manav"),
    "Kusuma School of Biological Sciences": ("Vivekanandan Perumal", "vperumal"),
    "School of Artificial Intelligence": ("Parag Singla", "parags"),
    "School of Interdisciplinary Research": ("B. Premachandran", "prem"),
    "School of Public Policy": ("Ambuj D. Sagar", "asagar"),
    # Centres
    "Centre for Applied Research in Electronics": ("Samaresh Das", "samareshdas"),
    "Centre for Atmospheric Sciences": ("Sagnik Dey", "sagnik"),
    "Centre for Automotive Research and Tribology": ("Santanu Kumar Mishra", "skmishra"),
    "Centre for Biomedical Engineering": ("Tapan K. Gandhi", "tgandhi"),
    "Centre for Rural Development and Technology": ("Vivek Kumar", "vivekk"),
    "Centre for Sensors, Instrumentation and Cyber Physical System Engineering (SeNSE)": ("Kedar Khare", "kedark"),
    "Computer Centre": ("Smruti Ranjan Sarangi", "srsarangi"),
    "Educational Technology Services Centre": ("Sourabh B. Paul", "sbpaul"),
    "National Resource Centre for Value Education in Engineering": ("James Gomes", "jgomes"),
    "Optics and Photonics Centre": ("Kedar Khare", "kedark"),
    "Transportation Research and Injury Prevention Programme (TRIPP)": ("Girish Agrawal", "girish"),
}

# Directory label for the unit's head: departments say "HoD", schools/centres say "Head".
_HEAD_LABEL = {"Departments": "HoD", "Centres": "Head", "Schools": "Head"}


async def _fetch_directory_units(base_url: str) -> dict[str, list[str]]:
    """Pull each Directory tab's unit list from the SAME endpoint the Directory
    page calls, so the bot lists exactly what that page shows (the backend
    applies its own roster/has-faculty rules we must not re-implement)."""
    units: dict[str, list[str]] = {}
    async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
        for tag, label in _DIRECTORY_CATEGORIES:
            resp = await client.get(
                f"{base_url.rstrip('/')}/api/directory/grouped",
                params={"category": tag, "summaryOnly": "true"},
            )
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data", payload) or {}
            groups = data.get("departments") or data.get("groups") or []
            names = set()
            for g in groups:
                # Each group is {_id, department: {_id, name}, stats: {...}}
                dept = g.get("department") if isinstance(g.get("department"), dict) else None
                name = ((dept or {}).get("name") or g.get("name") or "").strip()
                if name:
                    names.add(name)
            units[label] = sorted(names)
    return units


async def build_static_reference(db, backend_url: str | None = None) -> str:
    """Assemble the thematic-area→domain map and the department/centre/school
    list from Mongo into a compact Markdown reference block."""
    themes = await db["thematicareas"].find(
        {}, {"name": 1, "slug": 1, "display_order": 1}
    ).to_list(50)
    themes.sort(key=lambda t: t.get("display_order", 999))

    domain_docs = await db["domains"].find({}, {"name": 1}).to_list(500)
    domain_name = {d["_id"]: d.get("name") for d in domain_docs}

    # Each domain belongs to exactly one theme; derive the map from the
    # precomputed (theme, domain) facet rows.
    pairs = await db["taxonomyfacetcounts"].find(
        {
            "thematic_area_id": {"$ne": None},
            "domain_id": {"$ne": None},
            "subdomain_id": None,
            "department_id": None,
        },
        {"thematic_area_id": 1, "domain_id": 1},
    ).to_list(2000)
    theme_domains: dict = defaultdict(list)
    for p in pairs:
        name = domain_name.get(p["domain_id"])
        if name:
            theme_domains[p["thematic_area_id"]].append(name)

    # Departments / centres / schools: exactly what the Directory page lists.
    units: dict[str, list[str]] = {}
    if backend_url:
        try:
            units = await _fetch_directory_units(backend_url)
        except Exception as exc:
            logger.warning("Directory unit fetch failed (%s); falling back to DB categories", exc)
    if not units:
        depts = await db["departments"].find({}, {"name": 1, "category": 1}).to_list(500)
        by_cat: dict = defaultdict(list)
        for d in depts:
            nm = (d.get("name") or "").strip()
            if nm:
                by_cat[d.get("category") or "Other"].append(nm)
        units = {
            label: sorted(by_cat.get(_FALLBACK_CATEGORY[label], []))
            for _, label in _DIRECTORY_CATEGORIES
        }

    lines: list[str] = []
    lines.append("### Thematic areas and their research domains")
    lines.append(
        f"IIT Delhi research is organised into {len(themes)} thematic areas. Each "
        "research domain belongs to exactly ONE thematic area, as listed here:"
    )
    for t in themes:
        doms = sorted(theme_domains.get(t["_id"], []))
        body = "; ".join(doms) if doms else "(no domains)"
        lines.append(f"- **{t.get('name')}** — {body}")

    lines.append("")
    lines.append("### Academic units at IIT Delhi (exactly as listed on the Directory page)")
    lines.append(
        "These three lists are complete. Do NOT add any other unit, and do not "
        "mention research labs or administrative offices as academic units."
    )
    # Every unit — departments, centres, AND schools — carries its current head
    # (departments call it "HoD", centres/schools call it "Head"), exactly as the
    # Directory page shows for each category.
    for _, label in _DIRECTORY_CATEGORIES:
        names = units.get(label) or []
        if not names:
            continue
        head_label = _HEAD_LABEL.get(label, "Head")
        lines.append(f"- **{label}** ({len(names)}) — with current {head_label}:")
        for nm in names:
            head = _UNIT_HEADS.get(nm)
            if head and head[0]:
                who = f"[{head[0]}](/faculty/{head[1]})" if head[1] else head[0]
                lines.append(f"  - {nm} — {head_label}: {who}")
            else:
                lines.append(f"  - {nm}")

    return "\n".join(lines)
