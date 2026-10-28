"""Conservative citation checks for lite output before it reaches either transport.

These checks reject missing/invalid markers, unmatched figures/entities/campuses
and low-overlap claims. They are not a semantic entailment model.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
import re
from typing import Any

from src.retrieval.domain_signals import Years, campuses, codes, normalize, source_campuses, source_years, years
from src.retrieval.document import chunk_search_text

logger = logging.getLogger(__name__)
_MARKER = re.compile(r"\[(\d+)\]")
_LIST = re.compile(r"^(\s*)(?:[-+*]|\d+[.)])\s+(.*)$")


@dataclass(frozen=True)
class LogicalClaim:
    text: str
    markers: tuple[int, ...]
    kind: str
    context: tuple[str, ...] = ()
    wrapped: bool = False
    relation_text: str = ""


def _plain(text: str) -> str:
    # Persisted Markdown escapes punctuation (notably 2026\-2027). Compare
    # the visible text, so an academic range does not become competing years.
    # Normalize CommonMark punctuation except brackets, which belong to the
    # strict citation syntax. Never erase arbitrary backslashes or infer values.
    text = re.sub(r"\\([!\"#$%&'()*+,\-./:;<=>?@\\^_`{|}~])", r"\1", text)
    return re.sub(r"[*_`~]", "", text).strip()


def _label(text: str) -> bool:
    """Only structural noun headings may defer evidence to their child claims."""
    bare = _MARKER.sub("", _plain(text)).rstrip(": ")
    figures = numbers(bare)
    if any(not re.fullmatch(r"(?:19|20)\d{2}", value) for value in figures):
        return False
    folded = normalize(bare)
    if (re.search(r"\b(?:nguong diem|muc diem|diem toi thieu|dieu kien|thong tin)\b", folded)
            and re.search(r"\b(?:nhu sau|sau day|duoi day|diem sau|la|gom)\s*$", folded)
            and not re.search(r"\b(?:khong|tat ca|moi|bat buoc|duy nhat|bao dam|dam bao|chac chan|mien phi)\b", folded)):
        return True
    return len(bare.split()) <= 16 and not re.search(
        r"\b(?:la|co|duoc|phai|can|bao dam|dam bao|chap nhan|chac chan|mien phi)\b",
        normalize(bare).replace("co so", "campus"),
    )


def _table_separator(text: str) -> bool:
    cells = text.strip().strip("|").split("|")
    return bool(cells) and all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells)


def _method_introduction(text: str) -> bool:
    bare = normalize(_MARKER.sub("", _plain(text))).strip()
    return bool(re.search(r"\b\d+\s+phuong thuc\s+(?:tuyen sinh|xet tuyen)\b", bare)
                and re.search(r"\b(?:nhu sau|sau day|duoi day)\s*:\s*$", bare))


def _blocks(answer: str):
    """Join soft wraps; keep list items and table data rows in separate scopes."""
    lines = [re.sub(r"^\s*>\s?", "", line).rstrip() for line in answer.splitlines()]
    pending: list[str] = []
    kind = "paragraph"
    headings: list[tuple[int, str]] = []
    labels: list[tuple[int, str]] = []
    table_header = ""
    after_blank = False

    def flush():
        if not pending:
            return []
        result = [(" ".join(pending), kind, tuple(text for _, text in headings + labels), len(pending) > 1)]
        pending.clear()
        return result

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            yield from flush()
            # A blank line before/between list items does not detach the
            # referential introduction (including its year/campus qualifiers).
            after_blank = True
            table_header = ""
            index += 1
            continue
        heading = re.match(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$", line)
        emphasized_heading = re.fullmatch(r"\s*(\*\*|__)(.+?)\1\s*:?\s*", line)
        setext = index + 1 < len(lines) and re.fullmatch(r"\s*(?:={3,}|-{3,})\s*", lines[index + 1])
        if heading or emphasized_heading or (setext and "|" not in line):
            yield from flush()
            if heading:
                level, text = len(heading[1]), heading[2]
            elif emphasized_heading:
                level, text = 6, emphasized_heading[2]
            else:
                level, text = 2, stripped
            headings = [(depth, value) for depth, value in headings if depth < level]
            labels.clear()
            if _label(text):
                headings.append((level, text))
            else:
                yield text, "heading", tuple(value for _, value in headings), False
            index += 1 if heading or emphasized_heading else 2
            continue
        if "|" in line:
            yield from flush()
            if index + 1 < len(lines) and _table_separator(lines[index + 1]):
                table_header = stripped
                if not _label(table_header.replace("|", " ")):
                    yield table_header.replace("|", " "), "table_header", tuple(value for _, value in headings), False
                index += 2
                continue
            if not _table_separator(line):
                context = tuple(value for _, value in headings + labels)
                if table_header:
                    context += (table_header,)
                yield stripped, "table_row", context, False
            index += 1
            continue
        table_header = ""
        item = _LIST.match(line)
        if item:
            yield from flush()
            indent, text = len(item[1]), item[2]
            labels = [(depth, value) for depth, value in labels if depth < indent]
            kind = "list_item"
            if _plain(text).endswith(":") and _label(text):
                labels.append((indent, text))
            else:
                pending.append(text)
        elif _method_introduction(stripped):
            yield from flush()
            # A numeric total is a factual claim with its own citation, rather
            # than a label exempt from validation. Its year/campus also scopes
            # each item; the total itself is not an individual item's figure.
            yield stripped, "list_introduction", tuple(value for _, value in headings), False
            scope = normalize(_MARKER.sub("", _plain(stripped)))
            scope = re.sub(r"\b\d+\s+(?=phuong thuc\b)", "", scope)
            labels = [(-1, scope)]
        elif _plain(stripped).endswith(":") and _label(stripped):
            yield from flush()
            labels = [(-1, stripped)]
        elif re.fullmatch(r"(?:`{3,}|~{3,}).*|[-*_]{3,}", stripped):
            yield from flush()
        else:
            if not pending:
                if after_blank:
                    labels.clear()
                kind = "paragraph"
            pending.append(stripped)
        after_blank = False
        index += 1
    yield from flush()


def logical_claims(answer: str) -> list[LogicalClaim]:
    """A closing paragraph marker can cover each independently checked sentence.

    Markers never cross blank paragraphs, sibling list items or table rows.
    No marker is written into the answer or invented from retrieved sources.
    """
    claims = []
    for text, kind, context, wrapped in _blocks(answer):
        text = _plain(text)
        ending = re.search(r"((?:\[\d+\][,\s]*)+)[.!?;]*\s*$", text)
        closing = tuple(int(value) for value in _MARKER.findall(ending[1])) if ending else ()
        units = re.split(
            r"(?<!TP\.)(?<!Tp\.)(?<!tp\.)(?<=[.!?;])\s+(?!\[\d+\])|(?<=\])\s+(?=[\w])",
            text,
        )
        for unit in units:
            claim = _MARKER.sub("", unit).strip(" .!?;")
            if kind == "table_row":
                claim = claim.replace("|", " ").strip()
            if not claim:
                continue
            if normalize(claim).strip(" ,!.") in {"chao ban", "xin chao ban", "cam on ban"}:
                continue
            own = tuple(int(value) for value in _MARKER.findall(unit))
            claims.append(LogicalClaim(claim, own or closing, kind, context, wrapped,
                                       unit if kind == "table_row" else ""))
    return claims


class RAGGroundingError(RuntimeError):
    """Generation completed but its claimed source support could not be verified."""
    code = "GENERATION_UNGROUNDED"

    def __init__(self, message: str, *, reason: str = "unsupported_claim", claim_index: int = 0,
                 citation_present: bool = False, source_count: int = 0):
        super().__init__(message)
        self.reason = reason
        self.claim_index = claim_index
        self.citation_present = citation_present
        self.source_count = source_count


def _number_value(value: str) -> str:
    """Compare exact decimal values without rounding or guessing separators."""
    whole, _, fraction = value.replace(",", ".").partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return whole + ("." + fraction if fraction else "")


def numbers(text: str) -> set[str]:
    return {_number_value(value) for value in re.findall(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)", _plain(text))}


def _major_name(value: str) -> str:
    from src.query_transform.abbreviations import ABBREVIATIONS

    name = normalize(_MARKER.sub("", _plain(value))).strip(" ():,;")
    name = re.sub(r"^(?:nganh|chuong trinh|major|program)\s+", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return normalize(ABBREVIATIONS.get(name, name))


_FIGURE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
_YEAR = re.compile(r"(?<!\d)(?:19|20|21)\d{2}(?:\s*[-–—/]\s*(?:19|20|21)\d{2})?(?!\d)")
_ENTITY_PATTERNS = {
    "program": (
        ("standard", r"\b(?:dai tra|standard)\b"),
        ("high_quality", r"\b(?:chat luong cao|clc|high[- ]quality)\b"),
        ("international_joint", r"\b(?:lien ket quoc te|international joint)\b"),
        ("advanced", r"\b(?:tien tien|advanced)\b"),
    ),
    "campus": (
        ("BVH", r"\b(?:bvh|ha noi|co so (?:phia )?bac|phia bac)\b"),
        ("BVS", r"\b(?:bvs|ho chi minh|tp[.\s]*hcm|hcm|sai gon|co so (?:phia )?nam|phia nam)\b"),
    ),
    "metric": (
        ("tuition", r"\b(?:hoc phi|tuition)\b"),
        ("cutoff", r"\b(?:diem chuan|cutoff)\b"),
        ("validity", r"\b(?:thoi han|hieu luc|validity|valid for)\b"),
    ),
}


@dataclass(frozen=True)
class _NumericRelation:
    value: str
    scope: dict[str, frozenset[str]]


def _mentions(text: str) -> list[tuple[int, int, str, str]]:
    found = []
    for kind, patterns in _ENTITY_PATTERNS.items():
        for value, pattern in patterns:
            found.extend((m.start(), m.end(), kind, value) for m in re.finditer(pattern, text))
    found.extend((m.start(), m.end(), "exam", m[0])
                 for m in re.finditer(r"\b(?:sat|act|ielts|toefl)\b", text))
    found.extend((m.start(), m.end(), "code", m[0])
                 for m in re.finditer(r"(?<!\w)(?:[a-z]{1,6}\d{1,4}|\d{7})(?!\w)", text))
    found.extend((m.start(), m.end(), "year", re.sub(r"\s*[-–—/]\s*", "-", m[0]))
                 for m in _YEAR.finditer(text))
    # Named majors are kept distinct rather than reduced to generic admission
    # vocabulary. Unknown paraphrases remain unverified, not guessed synonyms.
    for match in re.finditer(
        r"\b(?:nganh|chuong trinh|program|major)\s+([^:;,|.\d]+?)"
        r"(?=\s*(?:\(\s*)?ma\b|\s+(?:nam|la|co|hoc phi|diem chuan|tu|toi thieu|at|in|is)\b|\s*[:;,|.\d]|$)", text,
    ):
        name = match[1].strip()
        if name and not any(re.search(pattern, name) for _, pattern in _ENTITY_PATTERNS["program"]):
            found.append((match.start(1), match.end(1), "major", _major_name(name)))
    return sorted(found)


def _scope(text: str) -> dict[str, frozenset[str]]:
    result: dict[str, set[str]] = {}
    for _, _, kind, value in _mentions(normalize(_MARKER.sub("", _plain(text)))):
        result.setdefault(kind, set()).add(value)
    return {kind: frozenset(values) for kind, values in result.items()}


def _payload_figures(text: str):
    year_spans = [(m.start(), m.end()) for m in _YEAR.finditer(text)]
    entity_codes = codes(text)
    return [m for m in _FIGURE.finditer(text)
            if not any(start <= m.start() < end for start, end in year_spans)
            and m[0] not in entity_codes]


def _numeric_relations(text: str, inherited: dict[str, frozenset[str]]) -> list[_NumericRelation]:
    text = normalize(_MARKER.sub("", _plain(text)))
    figures = _payload_figures(text)
    if not figures:
        scope = {**inherited, **_scope(text)}
        if _YEAR.search(text) and "year" in scope and any(
            kind in scope for kind in ("program", "major", "exam", "campus", "code")
        ) and all(len(bound) == 1 for bound in scope.values()):
            return [_NumericRelation("@scope", scope)]
        return []
    mentions = _mentions(text)
    # A new subject/measure following a value starts its own relationship, even
    # inside one sentence ("SAT 1130 or ACT 25", "standard 35.9, CLC 58.3").
    boundaries = []
    for start, _, kind, value in mentions:
        if (kind == "metric" and not any(m.start() >= start for m in figures)
                and value in _scope(text[:figures[0].start()]).get("metric", ())):
            # A repeated trailing measure ("...85% mức trần học phí theo")
            # has no new value and is not a new/competing subject. Keep the
            # preceding relationships; different subjects still fail closed.
            continue
        if any(m.end() <= start for m in figures) and (
            kind == "metric" or any(k == kind and s < figures[0].start() for s, _, k, _ in mentions)
        ):
            boundaries.append((start, kind))
    shared = dict(inherited)
    if boundaries:
        changing = {kind for _, kind in boundaries}
        prefix_end = min(start for start, _, kind, _ in mentions if kind in changing)
        shared.update(_scope(text[:prefix_end]))
        # One explicit year scopes the coordinated values. A later year starts
        # its own local scope rather than borrowing the first year's figures.
        year_scope = _scope(text).get("year")
        if year_scope and len(year_scope) == 1:
            shared["year"] = year_scope
    starts = sorted({0, *(start for start, _ in boundaries), len(text)})
    result = []
    for start, end in zip(starts, starts[1:]):
        part = text[start:end]
        scope = {**shared, **_scope(part)}
        if "metric" not in scope and "exam" in scope:
            scope["metric"] = frozenset({"threshold"})
        values = _payload_figures(part)
        # A trailing competing subject with no value cannot retroactively label
        # an earlier number. Multiple competing subjects in a part are ambiguous.
        if not values or any(len(bound) != 1 for bound in scope.values()):
            return []
        result.extend(_NumericRelation(_number_value(m[0]), scope) for m in values)
    return result


def _claim_relations(item: LogicalClaim, defaults: dict[str, frozenset[str]]) -> list[_NumericRelation]:
    if not item.relation_text:
        return _numeric_relations(item.text, {**defaults, **_scope(" ".join(item.context))})
    # Keep table columns separate: a 2025 value must never acquire the 2026
    # header of an adjacent cell, and two program columns must not be pooled.
    cells = [_MARKER.sub("", cell).strip() for cell in item.relation_text.strip().strip("|").split("|")]
    header = item.context[-1] if item.context and "|" in item.context[-1] else ""
    columns = [cell.strip() for cell in header.strip().strip("|").split("|")] if header else []
    if header and len(columns) != len(cells):
        return []
    numeric_cells = [cell for cell in cells if _payload_figures(normalize(_plain(cell)))]
    if not header and len(numeric_cells) > 1:
        return []
    context = item.context[:-1] if header else item.context
    labels = " ".join(cell for cell in cells if not _payload_figures(normalize(_plain(cell))))
    base = {**defaults, **_scope(" ".join(context)), **_scope(labels)}
    # A named subject is often a bare table label, without the prose prefix
    # "ngành". Bind that label to its row, never to a neighbouring row's value.
    subjects = set()
    for cell, column in zip(cells, columns):
        field = normalize(_plain(column))
        if (re.search(r"\b(?:nganh|chuong trinh|major|program)\b", field)
                and not re.search(r"\b(?:ma|code)\b", field)
                and not _payload_figures(normalize(_plain(cell)))
                and re.search(r"[^\W\d_]", cell)
                and not _scope(cell).get("program")):
            subjects.add(_major_name(cell))
    if subjects:
        base["major"] = frozenset(subjects)
    result = []
    for index, cell in enumerate(cells):
        if columns and re.fullmatch(r"(?:tt|stt|no\.?|so thu tu|#)", normalize(_plain(columns[index]))):
            continue  # A row ordinal cannot substantiate a cutoff/tuition value.
        column_scope = _scope(columns[index]) if len(columns) == len(cells) else {}
        result.extend(_numeric_relations(cell, {**base, **column_scope}))
    return result


def _source_relations(chunk: Any, requested_year: Years | None = None) -> list[_NumericRelation]:
    metadata = chunk.metadata or {}
    # Metadata supplies shared qualifiers, never numeric payload values. A
    # local sentence/row/column qualifier overrides the broader document label.
    defaults = _scope(" ".join(str(metadata.get(key) or "") for key in (
        "title", "section", "heading", "heading_path", "academic_year", "admission_year",
        "campus", "campus_code", "program", "program_name", "major", "major_code", "program_code",
    )))
    # Admission years and academic ranges describe different scopes. Use the
    # same priority as retrieval instead of treating both as competing years.
    actual_year = source_years(chunk, requested_year or years(""))
    if actual_year.values:
        defaults["year"] = (frozenset({f"{min(actual_year.values)}-{max(actual_year.values)}"})
                            if actual_year.kind == "academic_range"
                            else frozenset(str(value) for value in actual_year.values))
    result = []
    for text, kind, context, wrapped in _blocks(str(chunk.content or "")):
        outline = re.match(r"^\d+(?:\.\d+)*[.)]\s+(.+:)\s*$", _plain(text))
        if outline and _label(outline[1]):
            continue  # Section numbers are not tuition/cutoff payload values.
        caption_title = normalize(metadata.get("table_title") or metadata.get("section") or "")
        if (metadata.get("chunk_type") == "table" and kind == "paragraph" and caption_title
                and re.fullmatch(r"bang\s+\d+\s+thuoc muc\s+" + re.escape(caption_title), normalize(_plain(text)))):
            continue  # Persisted table numbering is structural, not a cutoff.
        # Keep semicolon-separated subject/value clauses in one sentence so
        # their explicit shared prefix (e.g. "Tuition:") remains available.
        units = re.split(r"(?<!TP\.)(?<!Tp\.)(?<!tp\.)(?<=[.!?])\s+", _plain(text))
        for unit in units:
            item = LogicalClaim(unit, (), kind, context, wrapped,
                                unit if kind == "table_row" else "")
            result.extend(_claim_relations(item, defaults))
    return result


def _relationships_supported(item: LogicalClaim, evidence: list[Any]) -> bool:
    requested = _claim_relations(item, {})
    # A numeric claim with no unambiguous parse cannot bypass the new check.
    claimed_values = Counter(_number_value(m[0]) for m in _payload_figures(normalize(item.text)))
    if claimed_values != Counter(relation.value for relation in requested if relation.value != "@scope"):
        return False
    direct_scope = _scope(item.text)
    if not requested and "year" in direct_scope and any(
        kind in direct_scope for kind in ("program", "major", "exam", "campus", "code")
    ):
        return False
    if not requested:
        return True
    requested_year = years(" ".join((*item.context, item.text)))
    available = [relation for chunk in evidence for relation in _source_relations(chunk, requested_year)]
    for relation in requested:
        def compatible(candidate):
            return all(candidate.scope.get(kind) == values for kind, values in relation.scope.items())

        candidates = [candidate for candidate in available if compatible(candidate)]
        if not any(relation.value == "@scope" or candidate.value == relation.value for candidate in candidates):
            return False
        # Omitted qualifiers cannot turn a multi-program/year/campus table into
        # a single generic rate. Each relationship must identify its own scope.
        for kind in ("program", "major", "exam", "campus", "code", "year"):
            if kind not in relation.scope:
                alternatives = {candidate.scope[kind] for candidate in candidates if kind in candidate.scope}
                if len(alternatives) > 1:
                    return False
    return True


def validate_answer(answer: str, chunks: list[Any]) -> None:
    def reject(reason: str, message: str, index: int, cited: tuple[int, ...], kind: str):
        failed_text = claims[index - 1].text if 1 <= index <= len(claims) else ""
        folded = normalize(failed_text)
        shape = "claim"
        if re.search(r"<\|(?:channel|turn)|<channel\|>", failed_text):
            shape = "provider_control"
        elif re.match(r"(?:minh )?(?:chua tim thay|khong tim thay|chua co du|khong co du|chua the xac minh)", folded):
            shape = "abstention"
        logger.warning(
            "grounding_validation status=rejected reason=%s claim_index=%d citation_present=%s source_count=%d block_kind=%s claim_shape=%s",
            reason, index, bool(cited), len(chunks), kind, shape,
        )
        raise RAGGroundingError(message, reason=reason, claim_index=index,
                                citation_present=bool(cited), source_count=len(chunks))

    claims = logical_claims(answer)
    # Even markers in structural headings/headers must refer to a real source.
    invalid = tuple(int(value) for value in _MARKER.findall(answer) if not 1 <= int(value) <= len(chunks))
    if invalid:
        index = next((i for i, claim in enumerate(claims, 1) if set(claim.markers) & set(invalid)), 0)
        reject("invalid_marker", "Missing or invalid citation markers", index, invalid, "answer")
    if not claims:
        reject("missing_claim", "No factual answer claim could be verified", 0, (), "answer")
    for index, item in enumerate(claims, 1):
        if not item.markers:
            reject("missing_citation", "An answer claim has no citation marker", index, (), item.kind)
        evidence = [chunks[marker - 1] for marker in item.markers]
        source_text = "\n".join(chunk_search_text(chunk) for chunk in evidence)
        # Structural scope (e.g. a year/campus in a table header) must also be
        # supported by this claim's own sources, never borrowed from another row.
        claim = " ".join((*item.context, item.text))
        claim = _MARKER.sub("", claim)
        if not numbers(claim) <= numbers(source_text):
            reject("unsupported_figures", "A cited source does not contain the claimed figures", index, item.markers, item.kind)
        if not codes(claim) <= codes(source_text):
            reject("unsupported_code", "A cited source does not contain the claimed entity code", index, item.markers, item.kind)
        for acronym in re.findall(r"\b(?:SAT|ACT|IELTS|TOEFL)\b", claim, re.IGNORECASE):
            if not re.search(r"\b" + re.escape(acronym) + r"\b", source_text, re.IGNORECASE):
                reject("unsupported_exam", "A cited source does not contain the claimed exam", index, item.markers, item.kind)
        claimed_campuses = campuses(claim)
        known_campuses = frozenset().union(*(source_campuses(chunk) for chunk in evidence))
        if claimed_campuses and not claimed_campuses <= known_campuses:
            reject("unsupported_campus", "A cited source does not establish the claimed campus", index, item.markers, item.kind)
        if item.kind == "list_item" and any("phuong thuc" in normalize(scope) for scope in item.context):
            # Generic admission vocabulary alone cannot support a different
            # method. Check discriminating method phrases in the item's sources.
            actual = normalize(_plain(source_text))
            for phrase in ("tai nang", "danh gia nang luc", "danh gia tu duy", "ket hop", "thi tot nghiep thpt", "hoc ba", "phong van"):
                if phrase in normalize(item.text) and phrase not in actual:
                    reject("unsupported_method", "A cited source does not establish the claimed admission method", index, item.markers, item.kind)
        stopwords = {"la", "va", "cua", "cho", "duoc", "co", "tu", "den", "theo", "nam", "muc", "khoang", "diem", "toi", "thieu", "dong", "trieu", "ptit"}
        tokens = {token for token in re.findall(r"\w+", normalize(item.text)) if len(token) > 1 and not token.isdigit()} - stopwords
        source_tokens = set(re.findall(r"\w+", normalize(source_text)))
        if tokens and len(tokens & source_tokens) / len(tokens) < 0.5:
            reject("insufficient_textual_support", "Insufficient textual support for an answer claim", index, item.markers, item.kind)
        if not _relationships_supported(item, evidence):
            reject("unsupported_relationship", "A cited evidence unit does not establish the claimed entity-value relationship",
                   index, item.markers, item.kind)
        if item.wrapped:
            logger.info(
                "grounding_validation status=accepted reason=logical_wrapping claim_index=%d citation_present=True source_count=%d block_kind=%s",
                index, len(chunks), item.kind,
            )
