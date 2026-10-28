"""Narrow extractive recovery for a complete, explicitly numbered methods list.

No model calls, retrieval, guessed sources or paraphrased facts. Any missing,
conflicting or out-of-scope evidence declines recovery. The caller must still
run the normal factual/relational validator before presenting the result.
"""
import re
from urllib.parse import urlsplit

from src.retrieval.domain_signals import QueryScope, normalize, source_campuses, source_years


def admission_methods(question: str, chunks: list) -> str | None:
    query = normalize(question)
    scope = QueryScope.parse(question)
    if (not re.search(r"\bptit\b|hoc vien cong nghe buu chinh vien thong", query)
            or "phuong thuc" not in query
            or not re.search(r"\b(?:nhung|cac|nao|liet ke|danh sach|bao nhieu|may)\b", query)
            or re.search(r"\b(?:diem|dieu kien|sat|act|ielts|toefl|hoc phi|nganh|chuong trinh|khong|mien|chi|rieng|ngoai|tru|danh cho|hoc ba)\b", query)
            or scope.codes or scope.years.kind not in ("missing", "single")):
        return None

    totals = set()
    methods = {}
    known_years = set()
    known_campuses = set()
    for marker, chunk in enumerate(chunks, 1):
        try:
            host = urlsplit(str((chunk.metadata or {}).get('source_url') or '')).hostname
        except ValueError:
            return None
        if host not in {'ptit.edu.vn', 'tuyensinh.ptit.edu.vn'}:
            return None
        actual_year = source_years(chunk, scope.years)
        actual_campus = source_campuses(chunk)
        if (actual_year.kind != "single" or not actual_year.values
                or (scope.years.values and actual_year.values != scope.years.values)
                or (scope.campuses and not scope.campuses <= actual_campus)):
            return None
        known_years.add(actual_year.values)
        known_campuses.add(actual_campus)
        text = re.sub(r"\\([.()\-])", r"\1", str(chunk.content))
        text = re.sub(r"[*_`~]", "", text)
        totals.update(int(value) for value in re.findall(
            r"\b(\d{1,2})\s+phuong thuc\s+(?:tuyen sinh|xet tuyen)\b", normalize(text)))
        for line in text.splitlines():
            match = re.match(r"^\s*(?:\d+(?:[.)]\d+)*[.)]?\s*)?phương thức\s+(\d{1,2})\s*:\s*(.+)",
                             line.lstrip('# '), re.IGNORECASE)
            if not match:
                continue
            number = int(match[1])
            # The institutions following this heading describe exam organisers,
            # not campuses or an additional admission method.
            title = re.split(r"\s+của các đơn vị\b", match[2], flags=re.IGNORECASE)[0]
            title = " ".join(title.split()).strip(' .:;')
            if (not normalize(title).startswith('xet tuyen') or len(title) > 800
                    or re.search(r"[<>\[\]]|https?://", title)):
                return None
            previous = methods.get(number)
            if previous and normalize(previous[0]) != normalize(title):
                return None
            methods.setdefault(number, (title, marker))
    # A global overview may include sources whose heading mentions both cities
    # as exam organisers. Never generalise a source scoped to just one campus,
    # or combine conflicting single-campus evidence.
    single_campuses = {value for value in known_campuses if len(value) == 1}
    if (len(known_years) != 1 or len(totals) != 1 or len(single_campuses) > 1
            or (single_campuses and not scope.campuses)):
        return None
    total = next(iter(totals))
    if not 1 <= total <= 10 or set(methods) != set(range(1, total + 1)):
        return None
    return '\n'.join(f'- {methods[number][0]} [{methods[number][1]}].'
                     for number in range(1, total + 1))
