"""Load canonical PTIT documents while preserving citation metadata."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from bs4 import BeautifulSoup

from .parser import Heading, ParsedPage, Table, parse_pdf
from .markdown import infer_header_rows, is_separator, parse_row


@dataclass
class LoadedDocument:
    doc_id: str
    content: str
    metadata: dict[str, Any]
    file_path: str
    pages: list[ParsedPage] = field(default_factory=list)


@dataclass
class LoadResult:
    documents: list[LoadedDocument] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    skipped_duplicates: list[dict[str, str]] = field(default_factory=list)
    source_dir: str = ""
    warnings: list[dict[str, Any]] = field(default_factory=list)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def _markdown_row(line: str) -> list[str] | None:
    return parse_row(line)


def _is_markdown_separator(row: list[str]) -> bool:
    return is_separator(row)


def _parse_markdown_page(content: str, page_number: int) -> ParsedPage:
    """Restore headings/tables from canonical processed Markdown for Module 2."""
    lines = content.replace("\r\n", "\n").replace("\r", "\n").strip().splitlines()
    headings: list[Heading] = []
    tables: list[Table] = []
    text_lines: list[str] = []
    index = 0
    offset = 0
    fence: str | None = None

    def append_line(line: str) -> None:
        nonlocal offset
        text_lines.append(line)
        offset += len(line) + 1

    while index < len(lines):
        fence_match = re.match(r"^\s*(`{3,}|~{3,})", lines[index])
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            append_line(lines[index])
            index += 1
            continue
        if fence is not None:
            append_line(lines[index])
            index += 1
            continue
        heading_match = re.match(r"^\s*(#{1,6})\s+(.+?)\s*$", lines[index])
        if heading_match:
            heading_text = heading_match.group(2).strip()
            headings.append(Heading(heading_text, len(heading_match.group(1)), page_number, offset))
            append_line(heading_text)
            index += 1
            continue

        # WordPress/PDF conversions frequently encode numbered subheadings
        # as bold paragraphs rather than ATX headings. Restrict detection to
        # short, fully emphasized outline labels, not arbitrary bold facts.
        line = lines[index].strip()
        outline = re.sub(r"[*_]", "", line).replace("\\.", ".").replace("\\)", ")")
        outline_match = re.match(r"^(\d+(?:\.\d+)*\.|[IVXLCDM]+\.|[a-z]\))\s+", outline)
        if line.startswith("**") and line.rstrip(":.").endswith("**") and outline_match and len(outline) <= 180:
            marker = outline_match.group(1)
            level = 1 if re.match(r"[IVXLCDM]", marker) else (4 if marker.endswith(")") else min(5, 2 + marker.count(".") - 1))
            headings.append(Heading(line, level, page_number, offset))
            append_line(line)
            index += 1
            continue

        first_row = _markdown_row(lines[index])
        second_row = _markdown_row(lines[index + 1]) if index + 1 < len(lines) else None
        if first_row is not None and second_row is not None and _is_markdown_separator(second_row):
            table_start = offset
            rows = [first_row]
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines):
                row = _markdown_row(lines[index])
                if row is None:
                    break
                if not _is_markdown_separator(row):
                    rows.append(row)
                table_lines.append(lines[index])
                index += 1
            # Keep the source position available to table-aware chunking. The
            # combined strategy removes these rows before creating text chunks.
            for line in table_lines:
                append_line(line)
            tables.append(Table(rows, page_number, table_start, offset - 1, infer_header_rows(rows)))
            continue

        append_line(lines[index])
        index += 1

    return ParsedPage(page_number, "\n".join(text_lines).strip(), tables=tables, headings=headings)


def parse_markdown_document(path: Path) -> LoadedDocument:
    raw = path.read_text(encoding="utf-8-sig")
    metadata: dict[str, Any] = {}
    content = raw
    frontmatter = re.match(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", raw, flags=re.DOTALL)
    if frontmatter:
        for line in frontmatter.group(1).splitlines():
            if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = _parse_scalar(value)
        content = raw[frontmatter.end():].strip()

    doc_id = str(metadata.get("doc_id") or hashlib.sha256(str(path).encode()).hexdigest()[:24])
    title = str(metadata.get("title") or path.stem)
    metadata.update(
        {
            "doc_id": doc_id,
            "title": title,
            "source_type": str(metadata.get("source_type") or "processed_markdown"),
            "source_file": path.as_posix(),
            "source": title,
        }
    )
    metadata.setdefault("source_url", "")
    metadata.setdefault("published_at", "")
    metadata.setdefault("retrieved_at", "")
    metadata.setdefault("section", "Toàn văn")
    # Never trust a stale frontmatter hash after a local content edit.
    metadata["content_hash"] = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
    try:
        page_number = int(metadata.get("page_number") or 1)
    except (TypeError, ValueError):
        page_number = 1
    pages = [_parse_markdown_page(content, page_number)]
    if "cau-hoi-thuong-gap" in doc_id or "câu hỏi thường gặp" in title.casefold():
        page = pages[0]
        published_year = str(metadata.get("published_at", ""))[:4]
        assigned_year = str(metadata.get("admission_year", ""))
        if published_year.isdigit() and assigned_year.isdigit() and published_year < assigned_year:
            # A crawl's collection year is not proof of policy applicability.
            # Keep it for provenance but do not label an old FAQ as current.
            metadata["collection_admission_year"] = metadata.get("admission_year")
            metadata["collection_academic_year"] = metadata.get("academic_year", "")
            policy_years = sorted(set(re.findall(r"(?:THPT|tuyển sinh|năm)\s+(20\d{2})\b", re.sub(r"[*_\\]", "", content), flags=re.IGNORECASE)))
            metadata["policy_years_mentioned"] = policy_years
            metadata["admission_year"] = policy_years[0] if len(policy_years) == 1 else ""
            metadata["academic_year"] = ""
            metadata["temporal_scope"] = "historical_or_unspecified"
            metadata["applicability_note"] = (
                f"FAQ này được công bố năm {published_year}; chưa xác minh phạm vi áp dụng cho mùa tuyển sinh hiện tại. "
                "Các điều kiện và mốc thời gian cần đối chiếu với thông báo đúng năm tuyển sinh."
            )
        # Existing processed FAQ files predate accordion-to-heading support.
        # Restore question boundaries without rewriting their source wording.
        known = {heading.start_offset for heading in page.headings}
        for match in re.finditer(r"(?m)^[^\n]+(?:\n(?!\n)[^\n]+)*", page.text):
            question = match.group().strip()
            if "?" in question and len(question) <= 700 and match.start() not in known:
                page.headings.append(Heading(question, 3, page_number, match.start()))
        page.headings.sort(key=lambda heading: heading.start_offset or 0)
    return LoadedDocument(doc_id, content, metadata, str(path), pages=pages)


def _find_processed_dir(requested: Path) -> Path | None:
    candidates: list[Path] = []
    parts = list(requested.parts)
    if "raw" in parts:
        raw_index = parts.index("raw")
        candidates.append(Path(*parts[:raw_index], "processed", *parts[raw_index + 1 :]))
        candidates.append(Path(*parts[:raw_index], "processed"))
    candidates.extend([requested / "processed", requested])
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("*.md")):
            return candidate
    return None


def _metadata_file_for(raw_dir: Path) -> Path | None:
    for candidate in [raw_dir / "metadata.json", *[p / "metadata.json" for p in raw_dir.parents]]:
        if candidate.exists():
            return candidate
    return None


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _raw_metadata_index(raw_dir: Path) -> dict[str, dict[str, Any]]:
    metadata_path = _metadata_file_for(raw_dir)
    if metadata_path is None:
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for item in _walk_dicts(payload):
        raw_file = item.get("raw_file")
        if raw_file:
            normalized = Path(str(raw_file)).as_posix().lower()
            result[normalized] = item
            result[Path(normalized).name] = item
    return result


def _metadata_for_raw(path: Path, index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    normalized = path.as_posix().lower()
    return dict(index.get(normalized) or index.get(path.name.lower()) or {})


def _load_raw(raw_dir: Path) -> LoadResult:
    result = LoadResult(source_dir=str(raw_dir))
    metadata_index = _raw_metadata_index(raw_dir)
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".html", ".htm"}:
            continue
        try:
            base = _metadata_for_raw(path, metadata_index)
            base_doc_id = str(base.get("doc_id") or hashlib.sha256(path.as_posix().encode()).hexdigest()[:24])
            common = {
                **base,
                "title": str(base.get("title") or path.stem),
                "source_url": str(base.get("source_url") or base.get("final_url") or ""),
                "source_file": path.as_posix(),
                "published_at": str(base.get("published_at") or ""),
                "retrieved_at": str(base.get("retrieved_at") or ""),
            }
            common["source"] = common["title"]
            if path.suffix.lower() == ".pdf":
                for page in parse_pdf(str(path)):
                    if not page.text.strip() and not page.tables:
                        result.warnings.append({"file": str(path), "page_number": page.page_number, "code": "ocr_required"})
                        continue
                    doc_id = f"{base_doc_id}-p{page.page_number:03d}"
                    metadata = {
                        **common,
                        "doc_id": doc_id,
                        "parent_doc_id": base_doc_id,
                        "source_type": "official_pdf",
                        "page_number": page.page_number,
                        "section": f"Trang {page.page_number}",
                        "content_hash": hashlib.sha256(page.text.encode("utf-8")).hexdigest(),
                    }
                    # A page containing a table but no extractable prose is
                    # still indexable; never silently discard its cells.
                    content = page.text or "\n".join(" | ".join(row) for table in page.tables for row in table.rows)
                    result.documents.append(LoadedDocument(doc_id, content, metadata, str(path), pages=[page]))
            else:
                soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                # Share the collector's structural conversion with the raw
                # fallback instead of flattening headings and table cells.
                from scripts.collect_ptit_admissions import clean_content_html, html_to_md, select_content_html
                cleaned_html, _links, _stats = clean_content_html(select_content_html(soup), common["source_url"])
                content = html_to_md(cleaned_html).strip()
                metadata = {
                    **common,
                    "doc_id": base_doc_id,
                    "source_type": str(base.get("source_type") or "official_web"),
                    "section": "Toàn văn",
                    "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
                result.documents.append(
                    LoadedDocument(
                        base_doc_id,
                        content,
                        metadata,
                        str(path),
                        pages=[_parse_markdown_page(content, 1)],
                    )
                )
        except Exception as exc:
            result.errors.append({"file": str(path), "error": str(exc)})
    return result


def load_documents(data_dir: str | Path, prefer_processed: bool = True) -> LoadResult:
    requested = Path(data_dir).resolve()
    if not requested.exists():
        raise FileNotFoundError(f"Data directory does not exist: {requested}")

    processed = _find_processed_dir(requested) if prefer_processed else None
    if processed is not None:
        result = LoadResult(source_dir=str(processed))
        seen_ids: set[str] = set()
        for path in sorted(processed.rglob("*.md")):
            try:
                document = parse_markdown_document(path)
                if document.doc_id in seen_ids:
                    raise ValueError(f"Duplicate doc_id: {document.doc_id}")
                if not document.content.strip():
                    raise ValueError("Document content is empty")
                seen_ids.add(document.doc_id)
                result.documents.append(document)
            except Exception as exc:
                result.errors.append({"file": str(path), "error": str(exc)})
        # Resolve aliases only after every target is loaded. A stale/missing
        # duplicate_of must never cause unique evidence to disappear.
        by_id = {document.doc_id: document for document in result.documents}
        retained: list[LoadedDocument] = []
        for document in result.documents:
            target_id = str(document.metadata.get("duplicate_of") or "")
            if not target_id:
                retained.append(document)
                continue
            target = by_id.get(target_id)
            visited = {document.doc_id}
            while target is not None and target.metadata.get("duplicate_of") and target.doc_id not in visited:
                visited.add(target.doc_id)
                target = by_id.get(str(target.metadata["duplicate_of"]))
            if (target is not None and target.doc_id not in visited
                    and document.content.strip() == target.content.strip()):
                # Equal text can originate from different official URLs. Keep
                # the alias's title and provenance even when indexing it once.
                target.metadata.setdefault("duplicate_sources", []).append({
                    "doc_id": document.doc_id, "title": document.metadata.get("title", ""),
                    "source_url": document.metadata.get("source_url", ""),
                    "source_file": document.file_path,
                })
                result.skipped_duplicates.append({"doc_id": document.doc_id, "duplicate_of": target.doc_id})
            else:
                result.warnings.append({"file": document.file_path, "code": "invalid_duplicate_reference", "duplicate_of": target_id})
                retained.append(document)
        result.documents = retained
        return result
    return _load_raw(requested)
