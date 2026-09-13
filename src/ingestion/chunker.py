from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from .parser import Heading, ParsedPage, Table


@dataclass
class Chunk:
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Section:
    page_number: int
    heading: Heading | None
    heading_path: tuple[str, ...]
    body: str


def _stable_chunk_id(
    content: str,
    metadata: Dict[str, Any],
    *,
    ordinal: int,
    parent_chunk_id: str = "",
) -> str:
    """Build a repeatable ID without depending on process-local state."""
    identity = "\x1f".join(
        [
            str(metadata.get("doc_id") or metadata.get("source_file") or metadata.get("source") or ""),
            str(metadata.get("chunk_type", "text")),
            str(metadata.get("page_number", "")),
            str(metadata.get("heading", "")),
            parent_chunk_id,
            str(ordinal),
            content,
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _metadata_for_chunk(
    metadata: Dict[str, Any] | None,
    *,
    page_number: int,
    chunk_type: str,
    chunk_index: int,
    heading: Heading | None,
    heading_path: Iterable[str] = (),
) -> Dict[str, Any]:
    result = dict(metadata or {})
    result.update(
        {
            "page_number": page_number,
            "chunk_type": chunk_type,
            "chunk_index": chunk_index,
            "heading": heading.text if heading else str(result.get("heading") or result.get("section") or ""),
            "heading_level": heading.level if heading else result.get("heading_level"),
            "heading_path": list(heading_path),
        }
    )
    return result


def _validate_size(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")


def naive_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    metadata: Dict[str, Any] | None = None,
) -> List[Chunk]:
    """Split text by character count, retaining the Module-1 contract."""
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(text), step):
        content = text[i : i + chunk_size]
        if not content.strip():
            continue

        chunk_metadata = dict(metadata or {})
        chunk_metadata["chunk_index"] = len(chunks)
        identity = "\x1f".join(
            [
                str(chunk_metadata.get("doc_id", "")),
                str(chunk_metadata["chunk_index"]),
                content,
            ]
        )
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]

        chunks.append(Chunk(chunk_id=chunk_id, content=content, metadata=chunk_metadata))
        if i + chunk_size >= len(text):
            break

    return chunks


def _update_heading_stack(stack: list[Heading], heading: Heading) -> list[Heading]:
    kept = [item for item in stack if item.level < heading.level]
    kept.append(heading)
    return kept


def _heading_positions(page: ParsedPage) -> list[tuple[int, int, Heading]]:
    positions: list[tuple[int, int, Heading]] = []
    cursor = 0
    for order, heading in enumerate(page.headings):
        position = page.text.find(heading.text, cursor)
        if position < 0:
            position = page.text.find(heading.text)
        if position >= 0:
            positions.append((position, order, heading))
            cursor = position + len(heading.text)
    positions.sort(key=lambda item: (item[0], item[1]))
    return positions


def _sections(pages: List[ParsedPage]) -> list[_Section]:
    sections: list[_Section] = []
    heading_stack: list[Heading] = []

    for page in pages:
        text = (page.text or "").replace("\r\n", "\n").replace("\r", "\n")
        positions = _heading_positions(page)
        current_heading = heading_stack[-1] if heading_stack else None

        if not positions:
            body = text.strip()
            if page.headings:
                if body:
                    for heading in page.headings:
                        heading_stack = _update_heading_stack(heading_stack, heading)
                    current_heading = heading_stack[-1]
                    sections.append(
                        _Section(
                            page.page_number,
                            current_heading,
                            tuple(item.text for item in heading_stack),
                            body,
                        )
                    )
                else:
                    # A parser may detect a heading whose normalized form is
                    # absent from extract_text(). It is still meaningful.
                    for heading in page.headings:
                        heading_stack = _update_heading_stack(heading_stack, heading)
                        sections.append(
                            _Section(
                                page.page_number,
                                heading,
                                tuple(item.text for item in heading_stack),
                                "",
                            )
                        )
            elif body:
                sections.append(
                    _Section(
                        page.page_number,
                        current_heading,
                        tuple(item.text for item in heading_stack),
                        body,
                    )
                )
            continue

        first_position = positions[0][0]
        preface = text[:first_position].strip()
        if preface:
            sections.append(
                _Section(
                    page.page_number,
                    current_heading,
                    tuple(item.text for item in heading_stack),
                    preface,
                )
            )

        for index, (position, _order, heading) in enumerate(positions):
            heading_stack = _update_heading_stack(heading_stack, heading)
            next_position = positions[index + 1][0] if index + 1 < len(positions) else len(text)
            body_start = position + len(heading.text)
            body = text[body_start:next_position].strip()
            sections.append(
                _Section(
                    page.page_number,
                    heading,
                    tuple(item.text for item in heading_stack),
                    body,
                )
            )

    return sections


def _hard_wrap(text: str, max_size: int) -> list[str]:
    """Wrap a long sentence at whitespace, falling back to exact char bounds."""
    result: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_size:
        boundary = remaining.rfind(" ", 0, max_size + 1)
        if boundary <= 0:
            boundary = max_size
        result.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        result.append(remaining)
    return result


def _split_oversize_unit(text: str, max_size: int) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?…])\s+|\n+", text) if part.strip()]
    if not sentences:
        return []
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidates = [sentence] if len(sentence) <= max_size else _hard_wrap(sentence, max_size)
        for candidate in candidates:
            joined = f"{current} {candidate}".strip()
            if current and len(joined) > max_size:
                pieces.append(current)
                current = candidate
            else:
                current = joined
    if current:
        pieces.append(current)
    return pieces


def _split_body(body: str, max_size: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", body.strip()) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        units.extend([paragraph] if len(paragraph) <= max_size else _split_oversize_unit(paragraph, max_size))

    chunks: list[str] = []
    current = ""
    for unit in units:
        separator = "\n\n" if current else ""
        candidate = f"{current}{separator}{unit}"
        if current and len(candidate) > max_size:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_section(section: _Section, max_size: int) -> list[str]:
    heading_text = section.heading.text.strip() if section.heading else ""
    if not section.body:
        return [heading_text] if heading_text else []
    if not heading_text:
        return _split_body(section.body, max_size)

    # Repeating the heading on each fragment preserves section context without
    # creating overlap between body text fragments.
    available = max_size - len(heading_text) - 2
    if available <= 0:
        return [heading_text, *_split_body(section.body, max_size)]
    body_chunks = _split_body(section.body, available)
    return [f"{heading_text}\n\n{body_chunk}" for body_chunk in body_chunks] or [heading_text]


def structure_aware_chunk(
    pages: List[ParsedPage],
    max_size: int = 500,
    metadata: Dict[str, Any] | None = None,
) -> List[Chunk]:
    """Chunk pages at heading and paragraph boundaries with no overlap."""
    _validate_size("max_size", max_size)
    chunks: list[Chunk] = []
    for section in _sections(pages):
        for content in _split_section(section, max_size):
            chunk_metadata = _metadata_for_chunk(
                metadata,
                page_number=section.page_number,
                chunk_type="text",
                chunk_index=len(chunks),
                heading=section.heading,
                heading_path=section.heading_path,
            )
            chunk_metadata["chunking_strategy"] = "structure_aware"
            chunks.append(
                Chunk(
                    _stable_chunk_id(content, chunk_metadata, ordinal=len(chunks)),
                    content,
                    chunk_metadata,
                )
            )
    return chunks


def _markdown_row(row: List[str]) -> str:
    cells = [str(cell or "").replace("\n", " ").replace("|", "\\|").strip() for cell in row]
    return "| " + " | ".join(cells) + " |"


def _table_heading(
    page: ParsedPage,
    table: Table,
    inherited: Heading | None,
    search_start: int,
) -> tuple[Heading | None, int]:
    if not page.headings:
        return inherited, -1
    positions = _heading_positions(page)
    first_row_text = " ".join(cell.strip() for cell in table.rows[0] if cell.strip()) if table.rows else ""
    candidates = [_markdown_row(table.rows[0]), first_row_text] if table.rows else []
    table_position = -1
    for candidate in candidates:
        if candidate:
            table_position = page.text.find(candidate, search_start)
            if table_position >= 0:
                break
    if table_position >= 0:
        preceding = [heading for position, _order, heading in positions if position <= table_position]
        if preceding:
            return preceding[-1], table_position
    return (positions[-1][2] if positions else page.headings[-1]), table_position


def _table_parts(table: Table, prefix: str, max_size: int) -> list[tuple[str, int, int]]:
    if not table.rows:
        return []
    header = table.rows[0]
    header_line = _markdown_row(header)
    separator = "| " + " | ".join("---" for _ in header) + " |"
    fixed = "\n".join(part for part in [prefix, header_line, separator] if part)
    body_rows = table.rows[1:]
    if not body_rows:
        return [(fixed, 0, 0)]

    parts: list[tuple[str, int, int]] = []
    current_rows: list[str] = []
    start_row = 1
    for row_number, row in enumerate(body_rows, start=1):
        rendered = _markdown_row(row)
        candidate = "\n".join([fixed, *current_rows, rendered])
        if current_rows and len(candidate) > max_size:
            parts.append(("\n".join([fixed, *current_rows]), start_row, row_number - 1))
            current_rows = [rendered]
            start_row = row_number
        else:
            current_rows.append(rendered)
    if current_rows:
        parts.append(("\n".join([fixed, *current_rows]), start_row, len(body_rows)))
    return parts


def table_aware_chunk(
    pages: List[ParsedPage],
    max_size: int = 500,
    metadata: Dict[str, Any] | None = None,
) -> List[Chunk]:
    """Render each table separately and repeat its header across row groups."""
    _validate_size("max_size", max_size)
    chunks: list[Chunk] = []
    inherited_heading: Heading | None = None
    table_index = 0

    for page in pages:
        table_search_start = 0
        for table in page.tables:
            if not table.rows:
                continue
            heading, table_position = _table_heading(page, table, inherited_heading, table_search_start)
            if table_position >= 0:
                table_search_start = table_position + 1
            context = heading.text if heading else str((metadata or {}).get("section") or "Toàn văn")
            table_title = context or f"Bảng {table_index + 1}"
            prefix = f"Bảng {table_index + 1} thuộc mục {table_title}"
            parts = _table_parts(table, prefix, max_size)
            for part_number, (content, row_start, row_end) in enumerate(parts, start=1):
                chunk_metadata = _metadata_for_chunk(
                    metadata,
                    page_number=table.page_number or page.page_number,
                    chunk_type="table",
                    chunk_index=len(chunks),
                    heading=heading,
                    heading_path=[heading.text] if heading else [],
                )
                chunk_metadata.update(
                    {
                        "chunking_strategy": "table_aware",
                        "table_index": table_index,
                        "table_title": table_title,
                        "table_part": part_number,
                        "table_part_count": len(parts),
                        "row_start": row_start,
                        "row_end": row_end,
                    }
                )
                chunks.append(
                    Chunk(
                        _stable_chunk_id(content, chunk_metadata, ordinal=len(chunks)),
                        content,
                        chunk_metadata,
                    )
                )
            table_index += 1
        if page.headings:
            inherited_heading = page.headings[-1]
    return chunks


def _strip_heading(content: str, heading: str) -> str:
    if heading and content.startswith(heading):
        return content[len(heading) :].lstrip()
    return content


def parent_child_chunk(
    pages: List[ParsedPage],
    parent_size: int = 1500,
    child_size: int = 300,
    metadata: Dict[str, Any] | None = None,
) -> List[Chunk]:
    """Return parent chunks followed by their addressable retrieval children."""
    _validate_size("parent_size", parent_size)
    _validate_size("child_size", child_size)
    if child_size > parent_size:
        raise ValueError("child_size must be less than or equal to parent_size")

    result: list[Chunk] = []
    parent_index = 0
    child_index = 0
    for section in _sections(pages):
        for parent_content in _split_section(section, parent_size):
            parent_metadata = _metadata_for_chunk(
                metadata,
                page_number=section.page_number,
                chunk_type="parent",
                chunk_index=parent_index,
                heading=section.heading,
                heading_path=section.heading_path,
            )
            parent_metadata["chunking_strategy"] = "parent_child"
            parent_id = _stable_chunk_id(parent_content, parent_metadata, ordinal=parent_index)
            parent = Chunk(parent_id, parent_content, parent_metadata)

            heading_text = section.heading.text if section.heading else ""
            child_section = _Section(
                section.page_number,
                section.heading,
                section.heading_path,
                _strip_heading(parent_content, heading_text),
            )
            children: list[Chunk] = []
            for child_content in _split_section(child_section, child_size):
                child_metadata = _metadata_for_chunk(
                    metadata,
                    page_number=section.page_number,
                    chunk_type="child",
                    chunk_index=child_index,
                    heading=section.heading,
                    heading_path=section.heading_path,
                )
                child_metadata.update(
                    {
                        "chunking_strategy": "parent_child",
                        "parent_chunk_id": parent_id,
                        "parent_chunk_index": parent_index,
                    }
                )
                child_id = _stable_chunk_id(
                    child_content,
                    child_metadata,
                    ordinal=child_index,
                    parent_chunk_id=parent_id,
                )
                children.append(Chunk(child_id, child_content, child_metadata))
                child_index += 1

            parent.metadata["child_chunk_ids"] = [child.chunk_id for child in children]
            parent.metadata["child_count"] = len(children)
            result.append(parent)
            result.extend(children)
            parent_index += 1
    return result


def _parse_markdown_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    return [cell.replace("\\|", "|").strip() for cell in stripped[1:-1].split("|")]


def _pages_without_table_rows(pages: List[ParsedPage]) -> list[ParsedPage]:
    cleaned: list[ParsedPage] = []
    for page in pages:
        row_signatures = {
            tuple(" ".join(str(cell or "").split()) for cell in row)
            for table in page.tables
            for row in table.rows
        }
        plain_rows = {" ".join(cell for cell in signature if cell).strip() for signature in row_signatures}
        kept_lines: list[str] = []
        for line in (page.text or "").splitlines():
            parsed = _parse_markdown_row(line)
            if parsed is not None:
                signature = tuple(" ".join(cell.split()) for cell in parsed)
                if signature in row_signatures or all(re.fullmatch(r":?-{3,}:?", cell) for cell in parsed):
                    continue
            normalized = " ".join(line.split())
            if normalized and normalized in plain_rows:
                continue
            kept_lines.append(line)
        cleaned.append(
            ParsedPage(
                page_number=page.page_number,
                text="\n".join(kept_lines).strip(),
                tables=page.tables,
                headings=page.headings,
            )
        )
    return cleaned


def _normalized_content(content: str) -> str:
    return " ".join(content.split()).casefold()


def combined_chunk(
    pages: List[ParsedPage],
    max_size: int = 500,
    parent_size: int = 1500,
    child_size: int = 300,
    metadata: Dict[str, Any] | None = None,
    parent_chunks: List[Chunk] | None = None,
) -> List[Chunk]:
    """Return de-duplicated child + table chunks and expose parents separately."""
    _validate_size("max_size", max_size)
    text_pages = _pages_without_table_rows(pages)
    hierarchy = parent_child_chunk(text_pages, parent_size, child_size, metadata)
    parents = [chunk for chunk in hierarchy if chunk.metadata.get("chunk_type") == "parent"]
    children = [chunk for chunk in hierarchy if chunk.metadata.get("chunk_type") == "child"]
    tables = table_aware_chunk(pages, max_size, metadata)

    for table in tables:
        same_page = [
            parent
            for parent in parents
            if parent.metadata.get("page_number") == table.metadata.get("page_number")
        ]
        same_heading = [
            parent
            for parent in same_page
            if parent.metadata.get("heading") == table.metadata.get("heading")
        ]
        parent = (same_heading or same_page or [None])[0]
        if parent is None:
            parent_metadata = _metadata_for_chunk(
                metadata,
                page_number=int(table.metadata["page_number"]),
                chunk_type="parent",
                chunk_index=len(parents),
                heading=None,
            )
            parent_metadata.update(
                {
                    "heading": table.metadata.get("heading", ""),
                    "heading_level": table.metadata.get("heading_level"),
                    "heading_path": table.metadata.get("heading_path", []),
                    "chunking_strategy": "parent_child",
                }
            )
            parent_id = _stable_chunk_id(table.content, parent_metadata, ordinal=len(parents))
            parent = Chunk(parent_id, table.content, parent_metadata)
            parents.append(parent)

        table.metadata["parent_chunk_id"] = parent.chunk_id
        table.metadata["parent_chunk_index"] = parent.metadata["chunk_index"]
        table.chunk_id = _stable_chunk_id(
            table.content,
            table.metadata,
            ordinal=int(table.metadata["chunk_index"]),
            parent_chunk_id=parent.chunk_id,
        )

    candidates = [*children, *tables]
    deduplicated: list[Chunk] = []
    seen_content: set[str] = set()
    seen_ids: set[str] = set()
    for chunk in candidates:
        content_key = _normalized_content(chunk.content)
        if not content_key or content_key in seen_content or chunk.chunk_id in seen_ids:
            continue
        chunk.metadata["combined_chunk_index"] = len(deduplicated)
        deduplicated.append(chunk)
        seen_content.add(content_key)
        seen_ids.add(chunk.chunk_id)

    live_child_ids = {
        chunk.chunk_id
        for chunk in deduplicated
        if chunk.metadata.get("chunk_type") == "child"
    }
    for parent in parents:
        child_ids = [child_id for child_id in parent.metadata.get("child_chunk_ids", []) if child_id in live_child_ids]
        table_ids = [
            chunk.chunk_id
            for chunk in deduplicated
            if chunk.metadata.get("chunk_type") == "table"
            and chunk.metadata.get("parent_chunk_id") == parent.chunk_id
        ]
        parent.metadata["child_chunk_ids"] = [*child_ids, *table_ids]
        parent.metadata["child_count"] = len(parent.metadata["child_chunk_ids"])

    if parent_chunks is not None:
        parent_chunks.extend(parent for parent in parents if parent.metadata.get("child_count", 0) > 0)
    return deduplicated
