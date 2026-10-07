"""Deterministic expansion of verified admissions abbreviations."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping


logger = logging.getLogger(__name__)

# Keep this vocabulary deliberately small. Every entry must be an exact,
# domain-verified abbreviation; unknown tokens are left untouched.
ABBREVIATIONS: dict[str, str] = {
    "cntt": "Công nghệ thông tin",
    "attt": "An toàn thông tin",
    "đgnl": "đánh giá năng lực",
    "dgnl": "đánh giá năng lực",
    "xtkh": "xét tuyển kết hợp",
    "clc": "chất lượng cao",
}


class AbbreviationNormalizer:
    """Expand known abbreviations while preserving their original spelling."""

    def __init__(self, abbreviations: Mapping[str, str] | None = None):
        source = ABBREVIATIONS if abbreviations is None else abbreviations
        self.abbreviations = {
            key.casefold(): value
            for key, value in source.items()
            if key and value
        }
        alternatives = sorted(self.abbreviations, key=len, reverse=True)
        self._pattern = (
            re.compile(
                rf"(?<!\w)(?P<abbr>{'|'.join(map(re.escape, alternatives))})(?!\w)",
                flags=re.IGNORECASE | re.UNICODE,
            )
            if alternatives
            else None
        )

    def normalize(self, query: str) -> str:
        """Return *query* with exact known tokens expanded at most once."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if self._pattern is None:
            return query

        expanded: list[str] = []

        def replace(match: re.Match[str]) -> str:
            abbreviation = match.group("abbr")
            expansion = self.abbreviations[abbreviation.casefold()]
            suffix = query[match.end():]
            already_expanded = re.match(
                rf"\s*\(\s*{re.escape(expansion)}\s*\)",
                suffix,
                flags=re.IGNORECASE | re.UNICODE,
            )
            if already_expanded:
                return abbreviation
            expanded.append(abbreviation.upper())
            return f"{abbreviation} ({expansion})"

        normalized = self._pattern.sub(replace, query)
        for abbreviation in dict.fromkeys(expanded):
            logger.info("Query abbreviation normalized: %s", abbreviation)
        return normalized


def is_known_abbreviation(token: str) -> bool:
    """Return whether *token* is one of the verified abbreviation spellings."""
    return token.casefold() in ABBREVIATIONS


__all__ = ["ABBREVIATIONS", "AbbreviationNormalizer", "is_known_abbreviation"]
