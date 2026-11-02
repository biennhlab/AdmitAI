from __future__ import annotations

import re

from src.ingestion.chunker import Chunk


def chunk_search_text(chunk: Chunk) -> str:
    """
    Combine semantic metadata and content for retrieval.
    Excludes technical IDs like chunk_id, parent_chunk_id, doc_id.
    """
    metadata = chunk.metadata or {}
    
    parts = []
    
    title = str(metadata.get("title") or "").strip()
    if title:
        parts.append(title)
        
    section = str(metadata.get("section") or "").strip()
    if section and section != title:
        parts.append(section)
        
    heading_path = metadata.get("heading_path")
    if isinstance(heading_path, list) and heading_path:
        path_str = " > ".join(str(h) for h in heading_path if h).strip()
        if path_str and path_str not in parts:
            parts.append(path_str)
    else:
        heading = str(metadata.get("heading") or "").strip()
        if heading and heading not in parts:
            parts.append(heading)
            
    content = str(chunk.content or "").strip()
    if content:
        parts.append(content)
        
    return "\n".join(parts)


def tokenize(text: str) -> list[str]:
    """
    Tokenize using Unicode word boundaries after normalizing punctuation.
    """
    if not text:
        return []
    
    # Remove apostrophes used in quotes or contractions, etc if needed.
    # We use re.findall(r'\w+', ...) which extracts words.
    # In Vietnamese, words are spaces-separated, but we also want to split on punctuation.
    # \w+ matches any word character (alphanumeric & underscore) in Unicode.
    # To truly normalize punctuation before tokenizing:
    text = text.casefold()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()
