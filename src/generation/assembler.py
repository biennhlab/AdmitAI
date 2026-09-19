from typing import Any, List, Tuple, Set
import re

class ContextAssembler:
    def __init__(self, max_chars: int = 15000):
        self.max_chars = max_chars

    def _normalize(self, text: str) -> str:
        """Normalize text for deduplication by stripping whitespaces and lowercasing."""
        return re.sub(r'\s+', ' ', text).strip().lower()

    def assemble(self, retrieved: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """
        Assemble chunks with the following rules:
        1. Prioritize: exact seeds -> parent/table siblings -> neighbors
        2. Deduplicate by normalized content
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
        seen_content: Set[str] = set()
        current_chars = 0
        
        for item in prioritized:
            chunk, score = item
            content = getattr(chunk, "content", "")
            
            norm = self._normalize(content)
            if not norm or norm in seen_content:
                continue
                
            chunk_chars = len(content)
            if current_chars + chunk_chars > self.max_chars:
                # To avoid splitting e.g. table rows, we skip the chunk entirely if it doesn't fit
                continue
                
            seen_content.add(norm)
            final_list.append(item)
            current_chars += chunk_chars
            
        return final_list
