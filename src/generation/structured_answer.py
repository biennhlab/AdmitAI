"""Render provider JSON as Markdown, with optional source references.

Only the response format is checked. Text, facts and source support are not
verified; absent or unusable source IDs do not reject an answer.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
import unicodedata

from .llm_client import LLMProviderError


class GenerationFormatError(LLMProviderError):
    """The provider returned output that cannot be rendered as an answer."""

    code = 'INVALID_GENERATION'

    def __init__(self, message: str, *, reason: str = 'invalid_generation'):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class StructuredAnswer:
    text: str
    abstained: bool = False


def render_structured_answer(raw: str, source_count: int) -> StructuredAnswer | None:
    """Parse complete JSON, or leave ordinary text for direct presentation."""
    def reject(reason: str = "invalid_generation"):
        raise GenerationFormatError('Invalid structured answer format', reason=reason)

    text = raw.strip()
    if text.startswith("```"):
        fence = re.fullmatch(r"```(?:json)?\s*\n(?P<body>.*?)\n```", text,
                             flags=re.IGNORECASE | re.DOTALL)
        if fence is None:
            if re.match(r"```(?:json|\s*\n\s*[\[{])", text, re.IGNORECASE):
                reject()
            return None
        text = fence['body'].strip()
    if not text.startswith(("{", "[")):
        return None
    if len(text) > 64000:
        reject()

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                reject()
            result[key] = value
        return result

    def invalid_constant(_):
        reject()

    try:
        data = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, RecursionError):
        reject()
    if not isinstance(data, dict) or set(data) != {'status', 'blocks'}:
        reject()
    blocks = data['blocks']
    if not isinstance(blocks, list):
        reject()
    if data['status'] == 'abstain':
        if blocks:
            reject()
        return StructuredAnswer('', abstained=True)
    if data['status'] != 'answer' or not 1 <= len(blocks) <= 64:
        reject()

    def content(value):
        if not isinstance(value, str) or not value.strip() or len(value) > 4000:
            reject()
        value = value.strip()
        if (any(unicodedata.category(char).startswith('C') for char in value)
                or re.search(r"[\[\]<>|]|https?://|```|~~~", value)
                or re.match(r"(?:#{1,6}\s|[-+*>]\s|\d+[.)]\s)", value)
                or re.match(r"(?:assistant|system|analysis|reasoning|thought|sources|references|nguồn tham khảo)\s*:",
                            value, re.IGNORECASE)):
            reject()
        return value

    def markers(value):
        if not isinstance(value, list):
            return ''
        ids = dict.fromkeys(index for index in value if type(index) is int and 1 <= index <= source_count)
        return ' '.join(f'[{index}]' for index in ids)

    def attributed(value, ids):
        body, suffix = content(value), markers(ids)
        if not suffix:
            return body
        if body.endswith(':'):
            return f'{body[:-1].rstrip()} {suffix}:'
        return f'{body} {suffix}'

    rendered = []
    previous = None
    ordinal = 0
    row_count = 0
    for block in blocks:
        if not isinstance(block, dict):
            reject()
        kind = block.get('kind')
        if kind == 'table':
            if set(block) != {'kind', 'columns', 'rows'}:
                reject()
            columns, rows = block['columns'], block['rows']
            if (not isinstance(columns, list) or not 2 <= len(columns) <= 12
                    or not isinstance(rows, list) or not rows):
                reject()
            columns = [content(column) for column in columns]
            lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
            for row in rows:
                row_count += 1
                if (row_count > 128 or not isinstance(row, dict) or 'cells' not in row
                        or set(row) - {'cells', 'source_ids'}):
                    reject()
                cells = row['cells']
                if not isinstance(cells, list) or len(cells) != len(columns):
                    reject()
                cells = [content(cell) for cell in cells]
                cells[-1] = attributed(cells[-1], row.get('source_ids'))
                lines.append('| ' + ' | '.join(cells) + ' |')
            fragment = '\n'.join(lines)
        elif kind in ('paragraph', 'bullet', 'numbered_item', 'heading'):
            if not {'kind', 'text'} <= set(block) or set(block) - {'kind', 'text', 'source_ids'}:
                reject()
            body = attributed(block['text'], block.get('source_ids'))
            if kind == 'bullet':
                fragment = '- ' + body
            elif kind == 'numbered_item':
                ordinal = ordinal + 1 if previous == kind else 1
                fragment = f'{ordinal}. {body}'
            elif kind == 'heading':
                fragment = '### ' + body
            else:
                fragment = body
        else:
            reject()
        separator = '\n' if kind in ('bullet', 'numbered_item') and previous == kind else '\n\n'
        if rendered:
            rendered.append(separator)
        rendered.append(fragment)
        previous = kind
    return StructuredAnswer(''.join(rendered))
