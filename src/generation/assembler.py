from typing import Any, Hashable, List, Tuple
import re


class ContextAssembler:
    def __init__(self, max_chars: int = 15000):
        self.max_chars = max_chars

    def _normalize(self, text: str) -> str:
        """Normalize text for deduplication by stripping whitespaces and lowercasing."""
        return re.sub(r"\s+", " ", text).strip().casefold()

    @classmethod
    def _freeze(cls, value: Any) -> Hashable:
        """Convert metadata values to stable, non-mutating identity components."""
        if isinstance(value, dict):
            return tuple(
                sorted((str(key), cls._freeze(item)) for key, item in value.items())
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(repr(cls._freeze(item)) for item in value))
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    @classmethod
    def _evidence_key(cls, chunk: Any, normalized_content: str) -> Hashable | None:
        """Return a key only when provenance is sufficient to prove duplication."""
        metadata = getattr(chunk, "metadata", {}) or {}
        source = tuple(
            (key, str(metadata.get(key)).strip().casefold())
            for key in ("doc_id", "source_url", "source_file", "raw_file", "parent_doc_id")
            if metadata.get(key) not in (None, "")
        )
        locator_keys = (
            "page_number",
            "section",
            "heading",
            "heading_path",
            "chunk_index",
            "parent_chunk_index",
            "table_index",
            "table_part",
            "row_start",
            "row_end",
        )
        has_locator = any(metadata.get(key) not in (None, "", []) for key in locator_keys)
        provenance = tuple(
            (key, cls._freeze(metadata.get(key)))
            for key in (
                *locator_keys,
                "admission_year",
                "academic_year",
                "campus",
                "campus_code",
                "campuses",
                "version",
                "content_hash",
                "updated_at",
                "provenance",
                "title",
                "source",
                "source_type",
                "published_at",
                "retrieved_at",
                "chunk_type",
                "parent_chunk_id",
            )
            if metadata.get(key) not in (None, "", [])
        )
        if source and has_locator:
            return normalized_content, source, provenance

        chunk_id = str(getattr(chunk, "chunk_id", "") or "").strip()
        if chunk_id:
            return normalized_content, source, (("chunk_id", chunk_id),), provenance
        return None

    def assemble(self, retrieved: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """
        Assemble chunks with the following rules:
        1. Prioritize: exact seeds -> parent/table siblings -> neighbors
        2. Deduplicate only chunks with matching content and provenance
        3. Enforce context budget (max_chars) and avoid splitting rows by skipping entirely if overflow
        """
        exact = []
        siblings = []
        neighbors = []
        
        for item in retrieved:
            chunk, score = item
            metadata = getattr(chunk, "metadata", {}) or {}
            exp_type = metadata.get("expansion_type", "exact")
            if exp_type == "table_sibling" or exp_type == "parent":
                siblings.append(item)
            elif exp_type == "neighbor":
                neighbors.append(item)
            else:
                exact.append(item)
                
        # Preserve original score ordering within each category
        prioritized = exact + siblings + neighbors
        
        final_list = []
        seen_evidence: set[Hashable] = set()
        current_chars = 0
        
        for item in prioritized:
            chunk, score = item
            content = getattr(chunk, "content", "")
            
            norm = self._normalize(content)
            if not norm:
                continue

            evidence_key = self._evidence_key(chunk, norm)
            if evidence_key is not None and evidence_key in seen_evidence:
                continue
                
            chunk_chars = len(content)
            if current_chars + chunk_chars > self.max_chars:
                # To avoid splitting e.g. table rows, we skip the chunk entirely if it doesn't fit
                continue
                
            if evidence_key is not None:
                seen_evidence.add(evidence_key)
            final_list.append(item)
            current_chars += chunk_chars
            
        return final_list
