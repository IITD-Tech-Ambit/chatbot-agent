"""Bundled offline IPC reference: sections A–H plus classes/subclasses common in
an academic patent portfolio. Serves code→meaning and topic→prefix lookups
without any network dependency.
"""

from __future__ import annotations

import re
from typing import Any

_SECTIONS: dict[str, str] = {
    "A": "Human Necessities",
    "B": "Performing Operations; Transporting",
    "C": "Chemistry; Metallurgy",
    "D": "Textiles; Paper",
    "E": "Fixed Constructions",
    "F": "Mechanical Engineering; Lighting; Heating; Weapons; Blasting",
    "G": "Physics",
    "H": "Electricity",
}

_ENTRIES: dict[str, str] = {
    "A01": "Agriculture; forestry; animal husbandry; hunting; trapping; fishing",
    "A23": "Foods or foodstuffs; their treatment",
    "A61": "Medical or veterinary science; hygiene",
    "A61B": "Diagnosis; surgery; identification",
    "A61C": "Dentistry; oral or dental hygiene",
    "A61F": "Implants; devices for care of body passages; treatment of eyes/ears",
    "A61K": "Preparations for medical, dental, or toilet purposes (drugs/pharmaceuticals)",
    "A61L": "Methods or apparatus for sterilising materials; disinfecting; deodorising",
    "A61M": "Devices for introducing media into, or onto, the body",
    "A61N": "Electrotherapy; magnetotherapy; radiation therapy; ultrasound therapy",
    "A61P": "Specific therapeutic activity of chemical compounds or medicinal preparations",
    "B01": "Physical or chemical processes or apparatus in general",
    "B01D": "Separation (filtration, distillation, adsorption, membranes)",
    "B01J": "Chemical or physical processes (e.g. catalysis, colloid chemistry)",
    "B22": "Casting; powder metallurgy",
    "B23": "Machine tools; metal-working not otherwise provided for",
    "B29": "Working of plastics; working of substances in a plastic state",
    "B32": "Layered products (laminates)",
    "B33": "Additive manufacturing technology (3D printing)",
    "B60": "Vehicles in general",
    "B62": "Land vehicles for travelling otherwise than on rails",
    "B64": "Aircraft; aviation; cosmonautics",
    "B81": "Microstructural technology",
    "B82": "Nanotechnology",
    "B82Y": "Specific uses or applications of nanostructures",
    "C01": "Inorganic chemistry",
    "C02": "Treatment of water, waste water, sewage, or sludge",
    "C07": "Organic chemistry",
    "C07D": "Heterocyclic compounds",
    "C08": "Organic macromolecular compounds (polymers)",
    "C08L": "Compositions of macromolecular compounds",
    "C09": "Dyes; paints; polishes; adhesives; miscellaneous compositions",
    "C10": "Petroleum, gas or coke industries; fuels; lubricants",
    "C12": "Biochemistry; beer; spirits; wine; microbiology; enzymology; genetic engineering",
    "C12N": "Micro-organisms or enzymes; genetic engineering",
    "C12Q": "Measuring or testing processes involving enzymes, nucleic acids or micro-organisms",
    "C22": "Metallurgy; ferrous or non-ferrous alloys",
    "C25": "Electrolytic or electrophoretic processes",
    "D01": "Natural or man-made threads or fibres; spinning",
    "D06": "Treatment of textiles",
    "E02": "Hydraulic engineering; foundations; soil-shifting",
    "E04": "Building",
    "E21": "Earth or rock drilling; mining",
    "F01": "Machines or engines in general",
    "F02": "Combustion engines; hot-gas or combustion-product engine plants",
    "F03": "Machines or engines for liquids; wind, spring, or weight motors",
    "F03D": "Wind motors (wind turbines)",
    "F16": "Engineering elements or units; general measures for machines/apparatus",
    "F21": "Lighting",
    "F24": "Heating; ranges; ventilating",
    "F25": "Refrigeration or cooling; heating and cooling combined; heat pumps",
    "F28": "Heat exchange in general",
    "G01": "Measuring; testing",
    "G01N": "Investigating or analysing materials by determining their properties",
    "G02": "Optics",
    "G02B": "Optical elements, systems, or apparatus",
    "G03": "Photography; cinematography; electrography; holography",
    "G05": "Controlling; regulating",
    "G06": "Computing; calculating or counting",
    "G06F": "Electric digital data processing",
    "G06K": "Recognition/presentation of data; record carriers (e.g. graphical data)",
    "G06N": "Computing arrangements based on specific computational models (AI, machine learning)",
    "G06Q": "Data processing for administrative, commercial, financial, or managerial purposes",
    "G06T": "Image data processing or generation in general",
    "G06V": "Image or video recognition or understanding",
    "G08": "Signalling",
    "G09": "Educating; cryptography; display; advertising; seals",
    "G10": "Musical instruments; acoustics",
    "G10L": "Speech analysis or synthesis; speech or voice recognition",
    "G11": "Information storage",
    "G16": "Information and communication technology adapted for specific application fields",
    "G16H": "Healthcare informatics (ICT for handling medical or healthcare data)",
    "H01": "Basic electric elements",
    "H01L": "Semiconductor devices; electric solid-state devices",
    "H01M": "Processes or means for the direct conversion of chemical into electrical energy (batteries, fuel cells)",
    "H01Q": "Antennas (aerials)",
    "H02": "Generation, conversion, or distribution of electric power",
    "H02J": "Circuit arrangements for supply or distribution of electric power; grids",
    "H03": "Basic electronic circuitry",
    "H04": "Electric communication technique",
    "H04B": "Transmission",
    "H04L": "Transmission of digital information (e.g. telegraphic communication, networks)",
    "H04N": "Pictorial communication (e.g. television)",
    "H04W": "Wireless communication networks",
    "H05": "Electric techniques not otherwise provided for",
    "H10": "Semiconductor devices; electric solid-state devices not otherwise provided for",
}

_TOPIC_KEYWORDS: list[tuple[frozenset[str], list[str]]] = [
    (frozenset({"drug", "drugs", "pharma", "pharmaceutical", "medicine", "medicinal", "therapeutic"}), ["A61K", "A61P"]),
    (frozenset({"drug delivery", "delivery"}), ["A61K", "A61M"]),
    (frozenset({"surgery", "surgical", "diagnosis", "diagnostic"}), ["A61B"]),
    (frozenset({"implant", "prosthesis", "prosthetic"}), ["A61F"]),
    (frozenset({"vaccine", "antibody", "genetic", "gene", "dna", "enzyme", "microorganism", "biotech", "biotechnology"}), ["C12N", "C12Q", "A61K"]),
    (frozenset({"cancer", "oncology", "tumour", "tumor"}), ["A61P", "A61K"]),
    (frozenset({"battery", "batteries", "fuel cell", "energy storage"}), ["H01M"]),
    (frozenset({"solar", "photovoltaic", "pv"}), ["H01L", "H02J"]),
    (frozenset({"semiconductor", "transistor", "chip", "vlsi"}), ["H01L"]),
    (frozenset({"antenna", "antennas", "aerial"}), ["H01Q"]),
    (frozenset({"wireless", "5g", "cellular", "mobile network"}), ["H04W", "H04B"]),
    (frozenset({"network", "networking", "data transmission", "internet"}), ["H04L"]),
    (frozenset({"machine learning", "artificial intelligence", "ai", "neural", "deep learning"}), ["G06N"]),
    (frozenset({"image", "vision", "computer vision", "video"}), ["G06V", "G06T", "H04N"]),
    (frozenset({"speech", "voice", "audio recognition"}), ["G10L"]),
    (frozenset({"computing", "software", "data processing", "computer"}), ["G06F", "G06N"]),
    (frozenset({"blockchain", "fintech", "commerce", "payment"}), ["G06Q"]),
    (frozenset({"healthcare informatics", "medical records", "health data"}), ["G16H"]),
    (frozenset({"sensor", "measurement", "testing", "analysis"}), ["G01N", "G01"]),
    (frozenset({"optics", "optical", "lens", "photonics", "laser"}), ["G02B", "G02"]),
    (frozenset({"catalysis", "catalyst"}), ["B01J"]),
    (frozenset({"membrane", "filtration", "separation"}), ["B01D"]),
    (frozenset({"water", "wastewater", "sewage", "effluent"}), ["C02"]),
    (frozenset({"polymer", "polymers", "plastic", "macromolecular"}), ["C08", "C08L"]),
    (frozenset({"nanotechnology", "nanoparticle", "nanostructure", "nano"}), ["B82Y", "B82"]),
    (frozenset({"3d printing", "additive manufacturing"}), ["B33"]),
    (frozenset({"wind", "wind turbine"}), ["F03D"]),
    (frozenset({"heat exchanger", "heat exchange", "cooling", "refrigeration"}), ["F28", "F25"]),
    (frozenset({"engine", "combustion"}), ["F02"]),
    (frozenset({"robot", "control", "automation"}), ["G05"]),
    (frozenset({"alloy", "metallurgy", "metal"}), ["C22", "B22"]),
    (frozenset({"construction", "building", "concrete"}), ["E04", "E02"]),
    (frozenset({"agriculture", "farming", "crop"}), ["A01"]),
    (frozenset({"food", "foodstuff"}), ["A23"]),
    (frozenset({"power grid", "electric power", "distribution"}), ["H02J", "H02"]),
]

_SYMBOL_RE = re.compile(r"^([A-H])(\d{0,2})([A-Z]?)")


def _normalize(code: str) -> str:
    return re.sub(r"\s+", " ", (code or "").strip().upper())


def _prefixes_of(code: str) -> list[str]:
    """Section, class, and subclass prefixes of an IPC code, longest first."""
    m = _SYMBOL_RE.match(code.replace(" ", ""))
    if not m:
        return []
    section, klass, subclass = m.group(1), m.group(2), m.group(3)
    prefixes: list[str] = []
    if section and klass and subclass:
        prefixes.append(f"{section}{klass}{subclass}")
    if section and klass:
        prefixes.append(f"{section}{klass}")
    prefixes.append(section)
    return prefixes


class StaticIpcTable:
    def describe(self, code: str) -> dict[str, Any] | None:
        norm = _normalize(code)
        section = norm[:1]
        if section not in _SECTIONS:
            return None

        matched_prefix = None
        meaning = None
        for prefix in _prefixes_of(norm):
            if prefix in _ENTRIES:
                matched_prefix = prefix
                meaning = _ENTRIES[prefix]
                break

        breakdown = {
            "section": section,
            "section_title": _SECTIONS[section],
        }
        if matched_prefix is None:
            return {
                "matched_prefix": section,
                "meaning": _SECTIONS[section],
                "level": "section",
                "breakdown": breakdown,
            }

        level = {1: "section", 3: "class", 4: "subclass"}.get(len(matched_prefix), "subclass")
        return {
            "matched_prefix": matched_prefix,
            "meaning": meaning,
            "level": level,
            "breakdown": breakdown,
        }

    def suggest(self, topic: str, limit: int = 8) -> list[dict[str, Any]]:
        text = (topic or "").lower()
        if not text:
            return []
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for keywords, prefixes in _TOPIC_KEYWORDS:
            if any(kw in text for kw in keywords):
                for prefix in prefixes:
                    if prefix in seen:
                        continue
                    seen.add(prefix)
                    out.append({
                        "prefix": prefix,
                        "meaning": _ENTRIES.get(prefix, _SECTIONS.get(prefix[:1], "")),
                    })
        return out[:limit]
