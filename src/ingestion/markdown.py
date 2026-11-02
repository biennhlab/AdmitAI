"""Lossless helpers shared by the Markdown loader and table chunker."""

from __future__ import annotations

import re


def parse_row(line: str) -> list[str] | None:
    text = line.strip()
    if "|" not in text:
        return None
    # A pipe is escaped only after an odd number of backslashes.
    # Use a scanner so escaped delimiters and literal backslashes round-trip.
    values: list[str] = []
    cell: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text) and text[index + 1] in {"\\", "|"}:
            cell.append(text[index + 1])
            index += 2
            continue
        if char == "|":
            values.append("".join(cell).strip())
            cell = []
        else:
            cell.append(char)
        index += 1
    values.append("".join(cell).strip())
    if text.startswith("|"):
        values = values[1:]
    if text.endswith("|") and not re.search(r"(?<!\\)(?:\\\\)*\\\|$", text):
        values = values[:-1]
    return values


def is_separator(row: list[str] | None) -> bool:
    return bool(row) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in row)


def infer_header_rows(rows: list[list[str]]) -> int:
    """Recognize expanded rowspan headers conservatively, never numeric data rows.

    Official tables repeat the first two row-spanned identifier labels while
    later columns refine method names and score scales.
    """
    count = 1
    if not rows or len(rows[0]) < 3:
        return count
    first = rows[0]
    for row in rows[1:]:
        if len(row) != len(first) or row[:2] != first[:2] or not any(row[:2]):
            break
        if row == first or any(re.search(r"[A-Za-zÀ-ỹ]", cell) for cell in row[2:]):
            count += 1
        else:
            break
    return count
