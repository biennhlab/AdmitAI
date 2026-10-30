"""Recognise a generic PTIT catalog request and explicit name/code table rows.

This module reads existing index records only. It never follows navigation links
or infers a program from a topic heading. Complete catalogs require every part
and source row of the selected tables, including program-category labels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from .domain_signals import QueryScope, normalize, source_campuses, source_years


def is_program_overview(question: str) -> bool:
    text = normalize(question)
    institution = r"\b(?:ptit|hoc vien cong nghe buu chinh vien thong)\b"
    if not re.search(institution, text) or not re.search(r"\b(?:nganh|chuong trinh)\b", text):
        return False
    scope = QueryScope.parse(question)
    if scope.codes or scope.years.kind not in {"missing", "single"}:
        return False
    text = re.sub(institution, " ", text)
    text = re.sub(r"\b(?:19|20|21)\d{2}\b", " ", text)
    text = re.sub(r"\b(?:co so (?:phia )?(?:bac|nam)|phia (?:bac|nam)|ha noi|tp[.\s]*hcm|ho chi minh|bvh|bvs)\b", " ", text)
    allowed = {"cac", "nhung", "nganh", "dao", "tao", "hoc", "nao", "gi", "co",
               "danh", "sach", "liet", "ke", "tai", "o", "cua", "nam", "tuyen", "sinh",
               "tat", "ca", "day", "du", "toan", "bo", "chuong", "trinh", "kem", "ma", "xet", "va"}
    return set(re.findall(r"\w+", text)) <= allowed


def catalog_rows(chunk) -> list[tuple[str, str]]:
    """Read paired cells under explicit name/code headers; reject ambiguous rows."""
    result = []
    columns = None
    for line in str(chunk.content).splitlines():
        if "|" not in line:
            columns = None
            continue
        cells = [re.sub(r"[*`]", "", cell).strip() for cell in line.strip().strip("|").split("|")]
        folded = [normalize(cell) for cell in cells]
        names = [i for i, cell in enumerate(folded) if cell.startswith("ten nganh")]
        codes = [i for i, cell in enumerate(folded) if cell.startswith("ma xet tuyen")]
        codes = codes or [i for i, cell in enumerate(folded) if cell.startswith("ma nganh")]
        if names and codes:
            columns = (names[0], codes[0], len(cells))
            continue
        if columns is None:
            continue
        name_index, code_index, width = columns
        if len(cells) != width:
            continue
        name, code = cells[name_index], cells[code_index]
        if (not _program_code(code)
                or not name or len(name) > 200 or not re.search(r"[^\W\d_]", name)
                or re.search(r"[<>\[\]\\\x00-\x1f\x7f]|https?://", name)):
            continue
        result.append((name, re.sub(r"\s+", "", code)))
    return result


def _program_code(code: str) -> bool:
    return bool(re.fullmatch(r"[78]\d{6}(?:\s*_?\s*[A-Za-z][A-Za-z0-9]{0,11})?", code))


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    code: str
    sections: tuple[tuple[str, str], ...]
    chunk_id: str
    fields: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class CatalogTable:
    campus: str
    year: int
    parts: tuple
    entries: tuple[CatalogEntry, ...]


def _cells(line: str) -> list[str]:
    return [re.sub(r"\*\*|__|`", "", cell).strip()
            for cell in line.strip().strip('|').split('|')]


def _catalog_columns(cells):
    folded = [normalize(cell) for cell in cells]
    names = [i for i, cell in enumerate(folded) if cell.startswith('ten nganh')]
    codes = [i for i, cell in enumerate(folded) if cell.startswith('ma xet tuyen')]
    if not names or not codes:
        return None
    return names[0], codes[0]


def _parse_complete_table(parts: list) -> CatalogTable | None:
    """Prove part/row continuity and parse every data row without silent skips."""
    first = parts[0]
    metadata = first.metadata or {}
    count = metadata.get('table_part_count')
    campus = source_campuses(first)
    actual_year = source_years(first, QueryScope.parse('').years)
    if (type(count) is not int or not 1 <= count <= 1000 or len(parts) != count
            or len(campus) != 1 or actual_year.kind != 'single'
            or any(type(c.metadata.get('table_part')) is not int
                   or c.metadata.get('chunk_type') != 'table' for c in parts)):
        return None
    ordered = sorted(parts, key=lambda c: c.metadata.get('table_part', 0))
    if [c.metadata.get('table_part') for c in ordered] != list(range(1, count + 1)):
        return None
    identity_keys = ('doc_id', 'source_url', 'source_file', 'raw_file', 'version',
                     'content_hash', 'admission_year', 'academic_year', 'table_index')
    identity = tuple(metadata.get(key) for key in identity_keys)
    seen_ids = set()
    expected_row = 1
    expected_header = None
    sections = ()
    entries = []
    seen_codes = set()
    for chunk in ordered:
        current = chunk.metadata or {}
        if (not chunk.chunk_id or chunk.chunk_id in seen_ids
                or tuple(current.get(key) for key in identity_keys) != identity
                or current.get('table_part_count') != count
                or source_campuses(chunk) != campus):
            return None
        seen_ids.add(chunk.chunk_id)
        start, end = current.get('row_start'), current.get('row_end')
        if type(start) is not int or type(end) is not int or start != expected_row or end < start:
            return None
        header = None
        field_labels = None
        physical_rows = 0
        for line in str(chunk.content).splitlines():
            if '|' not in line:
                continue
            cells = _cells(line)
            columns = _catalog_columns(cells)
            if columns is not None:
                candidate_header = tuple(' '.join(normalize(c).split()) for c in cells)
                if expected_header is not None and candidate_header != expected_header:
                    return None
                expected_header, header = candidate_header, columns
                field_labels = tuple(cells)
                continue
            if header is None or all(re.fullmatch(r':?-{3,}:?', c) for c in cells):
                continue
            if len(cells) != len(expected_header):
                return None
            physical_rows += 1
            name_index, code_index = header
            name, code = cells[name_index], cells[code_index]
            if re.fullmatch(r'\d+', cells[0]):
                if (not _program_code(code) or not name or len(name) > 300
                        or re.search(r'[<>\[\]\\\x00-\x1f\x7f]|https?://', name)):
                    return None
                code = re.sub(r'\s+', '', code)
                key = (tuple(normalize(s[0]) for s in sections), code)
                if key in seen_codes:
                    return None
                seen_codes.add(key)
                entries.append(CatalogEntry(name, code, sections, chunk.chunk_id,
                                            tuple(zip(field_labels[1:], cells[1:]))))
            elif (re.fullmatch(r'[IVX]+|', cells[0]) and name and name == code
                  and not re.search(r'[<>\[\]\\\x00-\x1f\x7f]|https?://', name)):
                section = (name, chunk.chunk_id)
                sections = (section,) if cells[0] else (*sections[:1], section)
            else:
                return None
        if header is None or physical_rows != end - start + 1:
            return None
        expected_row = end + 1
    if not entries:
        return None
    return CatalogTable(next(iter(campus)), next(iter(actual_year.values)), tuple(ordered), tuple(entries))


def complete_catalog(question: str, families) -> tuple[CatalogTable, ...] | None:
    """Select complete catalog tables of one document/year for requested campuses.

    Prefer the widest declared catalog table per campus. An incomplete larger
    table cannot be replaced with a smaller subset or an older year's table.
    """
    if not is_program_overview(question):
        return None
    return _complete_catalog(QueryScope.parse(question), families)


def _complete_catalog(scope: QueryScope, families) -> tuple[CatalogTable, ...] | None:
    """Shared source-completeness proof for overviews and named-program lookup."""
    required = scope.campuses or frozenset({'BVH', 'BVS'})
    documents = {}
    for parts in families:
        if not parts:
            continue
        first = parts[0]
        metadata = first.metadata or {}
        try:
            host = urlsplit(str(metadata.get('source_url') or '')).hostname
        except ValueError:
            continue
        actual = source_years(first, scope.years)
        campus = source_campuses(first)
        if (host not in {'ptit.edu.vn', 'tuyensinh.ptit.edu.vn'} or not metadata.get('doc_id')
                or actual.kind != 'single' or len(campus) != 1 or not campus <= required
                or (scope.years.values and actual.values != scope.years.values)
                or not any(_catalog_columns(_cells(line)) is not None
                           for line in str(first.content).splitlines() if '|' in line)):
            continue
        year = next(iter(actual.values))
        document = (year, *(str(metadata.get(key) or '') for key in
                    ('doc_id', 'source_url', 'source_file', 'raw_file', 'version', 'content_hash')))
        by_campus = documents.setdefault(document, {})
        declared_size = max((c.metadata.get('row_end', 0) for c in parts
                             if type(c.metadata.get('row_end')) is int), default=0)
        by_campus.setdefault(next(iter(campus)), []).append((declared_size, parts))
    if not documents:
        return None
    newest = max(key[0] for key in documents)
    candidates = []
    for identity, by_campus in documents.items():
        if identity[0] != newest or set(by_campus) != set(required):
            continue
        selected = []
        for campus in sorted(required):
            widest = max(size for size, _ in by_campus[campus])
            choices = [parts for size, parts in by_campus[campus] if size == widest]
            if len(choices) != 1:
                break
            table = _parse_complete_table(choices[0])
            if table is None:
                break
            selected.append(table)
        if len(selected) == len(required):
            candidates.append(tuple(selected))
    if not candidates:
        return None
    coverage = [sum(len(table.entries) for table in candidate) for candidate in candidates]
    best = [candidate for candidate, count in zip(candidates, coverage) if count == max(coverage)]
    # Equally complete conflicting versions cannot be resolved by arbitrary rank.
    signatures = {tuple((table.campus, tuple((e.name, e.code, tuple(s[0] for s in e.sections), e.fields)
                                           for e in table.entries)) for table in candidate) for candidate in best}
    if len(signatures) != 1:
        return None
    result = best[0]
    ids = [part.chunk_id for table in result for part in table.parts]
    return result if len(ids) == len(set(ids)) else None


def named_program_query(question: str) -> str | None:
    """Recognize a whole program-information/existence question, never a filter.

    Names are matched against source cells later, not a hand-maintained list.
    Fee/cutoff/curriculum/career questions retain their specific generation path.
    """
    text = normalize(question).strip(' ?!.')
    scope = QueryScope.parse(question)
    if scope.codes or scope.years.kind not in {'missing', 'single'}:
        return None
    if re.search(r'\b(?:hoc phi|diem|mon hoc|hoc gi|viec lam|luong|so sanh|hoc bong|thoi gian|bao nhieu|chuong trinh hoc)\b', text):
        return None
    text = re.sub(r'\b(?:(?:tai|o|cua)\s+)?(?:ptit|hoc vien cong nghe buu chinh vien thong)\b', ' ', text)
    text = re.sub(r'\b(?:nam\s+)?(?:19|20|21)\d{2}\b', ' ', text)
    text = re.sub(r'\b(?:(?:tai|o)\s+)?(?:co so (?:phia )?(?:bac|nam)|phia (?:bac|nam)|ha noi|tp[.\s]*hcm|ho chi minh|bvh|bvs)\b', ' ', text)
    text = ' '.join(text.split()).strip(' ?!.')
    patterns = (
        r'(?:(?:cho (?:minh|toi|em) )?(?:biet |xin )?(?:thong tin(?: chi tiet)? |tim hieu )?(?:ve )?)?(?:nganh|chuong trinh)\s+(.+)',
        r'co\s+(?:(?:dao tao|tuyen sinh|tuyen)\s+)?nganh\s+(.+?)(?:\s+khong)?',
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            name = match[1].strip()
            if name and name not in {'nao', 'gi', 'dao tao', 'hoc'}:
                from src.query_transform.abbreviations import ABBREVIATIONS
                for abbreviation, expansion in ABBREVIATIONS.items():
                    short, long = normalize(abbreviation), normalize(expansion)
                    if name in {short, f'{short} ({long})', f'{long} ({short})'}:
                        return long
                return name
    return None


def named_program_catalog(question: str, families) -> tuple[CatalogTable, ...] | None:
    """Find all application entries matching a display or official major name."""
    requested = named_program_query(question)
    if requested is None:
        return None
    catalog = _complete_catalog(QueryScope.parse(question), families)
    if catalog is None:
        return None
    result = []
    for table in catalog:
        entries = []
        for entry in table.entries:
            names = {normalize(entry.name)}
            names.update(normalize(value) for label, value in entry.fields
                         if normalize(label).startswith('ten nganh'))
            if requested in names:
                entries.append(entry)
        if entries:
            result.append(CatalogTable(table.campus, table.year, table.parts, tuple(entries)))
    return tuple(result) or None


def catalog_seeds(question: str, ranked: list, top_k: int) -> list:
    """Prefer actual catalog rows, retaining the eligible source's year scope."""
    scope = QueryScope.parse(question)
    eligible = []
    for item in ranked:
        chunk = item[0]
        actual = source_years(chunk, scope.years)
        if not catalog_rows(chunk) or actual.kind != "single":
            continue
        if scope.years.values and actual.values != scope.years.values:
            continue
        eligible.append((item, next(iter(actual.values))))
    if not eligible:
        return []
    # A yearless question is answered from the newest retrieved catalog, not a
    # mixture of historical and current tables. No wall-clock year is invented.
    selected_year = max(year for _, year in eligible)
    remaining = [item for item, year in eligible if year == selected_year]
    selected = []
    seen_campuses = set()
    seen_names = set()
    while remaining and len(selected) < max(0, top_k):
        # Preserve rank as the tie-break, but do not fill the budget with the
        # same major's repeated table rows while another campus/major is present.
        best = max(range(len(remaining)), key=lambda i: (
            len(source_campuses(remaining[i][0]) - seen_campuses),
            len({normalize(name) for name, _ in catalog_rows(remaining[i][0])} - seen_names),
            -i,
        ))
        item = remaining.pop(best)
        names = {normalize(name) for name, _ in catalog_rows(item[0])}
        if names <= seen_names and source_campuses(item[0]) <= seen_campuses:
            continue
        selected.append(item)
        seen_names.update(names)
        seen_campuses.update(source_campuses(item[0]))
    return selected
