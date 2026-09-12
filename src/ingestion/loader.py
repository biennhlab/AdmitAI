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
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    return [cell.replace("\\|", "|").strip() for cell in stripped[1:-1].split("|")]


def _is_markdown_separator(row: list[str]) -> bool:
    return bool(row) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in row)


def _parse_markdown_page(content: str, page_number: int) -> ParsedPage:
    """Restore headings/tables from canonical processed Markdown for Module 2."""
    lines = content.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    headings: list[Heading] = []
    tables: list[Table] = []
    text_lines: list[str] = []
    index = 0

    while index < len(lines):
        heading_match = re.match(r"^\s*(#{1,6})\s+(.+?)\s*$", lines[index])
        if heading_match:
            heading_text = heading_match.group(2).strip()
            headings.append(Heading(heading_text, len(heading_match.group(1)), page_number))
            text_lines.append(heading_text)
            index += 1
            continue

        first_row = _markdown_row(lines[index])
        second_row = _markdown_row(lines[index + 1]) if index + 1 < len(lines) else None
        if first_row is not None and second_row is not None and _is_markdown_separator(second_row):
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
            tables.append(Table(rows=rows, page_number=page_number))
            # Keep the source position available to table-aware chunking. The
            # combined strategy removes these rows before creating text chunks.
            text_lines.extend(table_lines)
            continue

        text_lines.append(lines[index])
        index += 1

    return ParsedPage(page_number, "\n".join(text_lines).strip(), tables=tables, headings=headings)


def parse_markdown_document(path: Path) -> LoadedDocument:
    raw = path.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {}
    content = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) == 3:
            for line in parts[1].splitlines():
                if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                metadata[key.strip()] = _parse_scalar(value)
            content = parts[2].strip()

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
    metadata.setdefault("content_hash", hashlib.sha256(content.encode("utf-8")).hexdigest())
    try:
        page_number = int(metadata.get("page_number") or 1)
    except (TypeError, ValueError):
        page_number = 1
    pages = [_parse_markdown_page(content, page_number)]
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
                    if not page.text.strip():
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
                    result.documents.append(LoadedDocument(doc_id, page.text, metadata, str(path), pages=[page]))
            else:
                soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                content = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
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
                        pages=[ParsedPage(page_number=1, text=content)],
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
                if document.metadata.get("duplicate_of"):
                    result.skipped_duplicates.append(
                        {"doc_id": document.doc_id, "duplicate_of": str(document.metadata["duplicate_of"])}
                    )
                    continue
                if document.doc_id in seen_ids:
                    raise ValueError(f"Duplicate doc_id: {document.doc_id}")
                if not document.content.strip():
                    raise ValueError("Document content is empty")
                seen_ids.add(document.doc_id)
                result.documents.append(document)
            except Exception as exc:
                result.errors.append({"file": str(path), "error": str(exc)})
        return result
    return _load_raw(requested)
