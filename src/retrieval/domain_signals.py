"""Small, stateless scope rules for lite ranking and existing context expansion.

Only explicit metadata and source headings establish document scope. Mentions in
body text, crawl dates and the host's current year never establish source scope.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from src.ingestion.chunker import Chunk


def normalize(text: object) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", str(text).casefold())
        if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")


def codes(text: str) -> frozenset[str]:
    # Whole tokens only: CN1 != CN10, A00 != A001, 7480202 != 17480202.
    return frozenset(re.findall(r"(?<!\w)(?:[a-z]{1,6}\d{1,4}|\d{7})(?!\w)", text.casefold()))


@dataclass(frozen=True)
class Years:
    values: frozenset[int] = frozenset()
    kind: str = "missing"


def years(value: object) -> Years:
    text = str(value or "")
    found = frozenset(int(item) for item in re.findall(r"(?<!\d)(?:19|20|21)\d{2}(?!\d)", text))
    ranges = re.findall(r"((?:19|20|21)\d{2})\s*[-–—/]\s*((?:19|20|21)\d{2})", text)
    if len(found) == 2 and len(ranges) == 1:
        first, second = map(int, ranges[0])
        if second == first + 1:
            return Years(found, "academic_range")
    return Years(found, "single" if len(found) == 1 else "multi" if found else "missing")


def campuses(value: object) -> frozenset[str]:
    text = normalize(value)
    found = set()
    if re.search(r"\bbvh\b|\bha noi\b|\bco so (?:phia )?bac\b|\bphia bac\b", text):
        found.add("BVH")
    if re.search(r"\bbvs\b|\bho chi minh\b|\btp[.\s]*hcm\b|\bhcm\b|\bsai gon\b|\bco so (?:phia )?nam\b|\bphia nam\b", text):
        found.add("BVS")
    return frozenset(found)


def source_campuses(chunk: Chunk) -> frozenset[str]:
    metadata = chunk.metadata or {}
    explicit = " ".join(str(metadata.get(key) or "") for key in ("campus", "campus_code", "campuses"))
    local = frozenset()
    for key in ("heading", "heading_path", "section"):
        local = campuses(metadata.get(key) or "")
        if local:
            break
    if explicit.strip():
        # Unknown explicit labels remain unknown, rather than being overridden.
        known = campuses(explicit)
        return local if local and local <= known else known
    return local or campuses(metadata.get("title") or "")


def source_years(chunk: Chunk, requested: Years) -> Years:
    metadata = chunk.metadata or {}
    keys = ("academic_year", "admission_year") if requested.kind == "academic_range" else ("admission_year", "academic_year")
    for key in (*keys, "year", "title"):
        parsed = years(metadata.get(key))
        if parsed.values:
            return parsed
    return Years()


@dataclass(frozen=True)
class QueryScope:
    years: Years
    campuses: frozenset[str]
    codes: frozenset[str]
    intent: str

    @classmethod
    def parse(cls, query: str) -> QueryScope:
        text = normalize(query)
        intent = "cutoff" if "diem chuan" in text else "tuition" if "hoc phi" in text else "admissions"
        return cls(years(query), campuses(query), codes(query), intent)


@dataclass(frozen=True)
class DomainDecision:
    eligible: bool
    factors: tuple[tuple[str, float], ...]
    ambiguities: tuple[str, ...]
    exclusions: tuple[str, ...]


def assess(scope: QueryScope, chunk: Chunk) -> DomainDecision:
    factors = []
    ambiguities = []
    exclusions = []
    actual_years = source_years(chunk, scope.years)
    if scope.years.values:
        if not actual_years.values:
            ambiguities.append("year_missing")
            factors.append(("year_unknown", 0.8))
        elif scope.years.kind == "academic_range" and actual_years.kind == "academic_range":
            if actual_years.values != scope.years.values:
                exclusions.append("wrong_academic_range")
            else:
                factors.append(("academic_range", 1.25))
        elif scope.years.kind == "single" and actual_years.kind == "academic_range":
            # The end of an academic year is not the admission year.
            if min(actual_years.values) not in scope.years.values:
                exclusions.append("wrong_admission_year")
            else:
                factors.append(("academic_start", 1.1))
        elif scope.years.kind == "academic_range" and actual_years.kind == "single" and min(scope.years.values) not in actual_years.values:
            exclusions.append("wrong_academic_start")
        elif not scope.years.values.intersection(actual_years.values):
            exclusions.append("wrong_year")
        else:
            factors.append(("year_match", 1.25))
            if scope.years.kind == "academic_range":
                ambiguities.append("academic_range_unconfirmed")

    actual_campuses = source_campuses(chunk)
    if scope.campuses:
        if not actual_campuses:
            ambiguities.append("campus_missing")
            factors.append(("campus_unknown", 0.8))
            if scope.intent == "cutoff":
                exclusions.append("cutoff_campus_unconfirmed")
        elif not actual_campuses.intersection(scope.campuses):
            exclusions.append("wrong_campus")
        else:
            factors.append(("campus_match", 1.2))
    elif not actual_campuses and scope.intent == "cutoff":
        ambiguities.append("campus_unspecified")

    metadata = chunk.metadata or {}
    heading = normalize(" ".join(str(metadata.get(key) or "") for key in ("heading", "heading_path", "section", "title")))
    matched_section = (
        (scope.intent == "cutoff" and "diem chuan" in heading)
        or (scope.intent == "tuition" and "hoc phi" in heading)
        or (scope.intent == "admissions" and any(term in heading for term in ("xet tuyen", "tuyen sinh", "sat", "act")))
    )
    if matched_section:
        factors.append(("section_match", 1.5))
    elif scope.intent in {"cutoff", "tuition"} and any(term in heading for term in ("chuong trinh dao tao", "khung chuong trinh", "chuan dau ra", "curriculum")):
        factors.append(("curriculum_section", 0.5))

    entity_text = " ".join((str(chunk.content), heading, *(str(metadata.get(key) or "") for key in ("major_code", "program_code", "subject_combination"))))
    if scope.codes and scope.codes.intersection(codes(entity_text)):
        factors.append(("exact_entity_code", 1.8))
    return DomainDecision(not exclusions, tuple(factors), tuple(ambiguities), tuple(exclusions))


def expansion_allowed(scope: QueryScope, seed: Chunk, candidate: Chunk) -> bool:
    """Require proven document identity and identical known year/campus scope.

    Missing scope on both sides can remain ambiguous; missing scope on only one
    side cannot prove safe expansion. Conflicting source/version fields reject.
    """
    left, right = seed.metadata or {}, candidate.metadata or {}
    if not left.get("doc_id") or left.get("doc_id") != right.get("doc_id"):
        return False
    if scope.years.values and not source_years(seed, scope.years).values:
        return False
    if scope.campuses and not source_campuses(seed):
        return False
    for key in ("source_url", "source_file", "raw_file", "version", "content_hash", "admission_year", "academic_year"):
        if left.get(key) != right.get(key):
            return False
    if source_campuses(seed) != source_campuses(candidate):
        return False
    if source_years(seed, scope.years) != source_years(candidate, scope.years):
        return False
    return assess(scope, candidate).eligible


def has_named_program_evidence(query: str, chunks: list[Chunk]) -> bool:
    """Require the named program in context for program-existence questions.

    Generic admission vocabulary cannot establish that a specific major exists.
    This rule uses no list of unsupported programs and makes no negative claim.
    """
    text = normalize(query)
    match = re.search(
        r"\bco\s+(?:(?:tuyen|dao tao)\s+)?nganh\s+(.+?)"
        r"(?=\s+(?:nam|ma|khong|o|tai|hay|va)\b|[?!.]|$)", text,
    )
    if not match:
        return True
    name = re.sub(r"\s+", " ", match.group(1)).strip()
    if name in {"nao", "gi", "nhung nganh nao"}:
        return True
    pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)")
    return any(pattern.search(re.sub(r"\s+", " ", normalize(
        " ".join((chunk.content, *(str((chunk.metadata or {}).get(key) or "") for key in ("title", "section", "heading"))))
    ))) for chunk in chunks)


def requires_live_availability(query: str) -> bool:
    """The static admissions index cannot establish current inventory/capacity."""
    text = normalize(query)
    live = any(term in text for term in ("hien gio", "hien tai", "bay gio", "hom nay", "ngay mai", "dang con", "con chinh xac"))
    availability = any(term in text for term in ("phong trong", "cho trong", "con cho", "con suat", "so cho con", "con bao nhieu phong"))
    return live and availability
