import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Chunk:
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

def naive_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    metadata: Dict[str, Any] | None = None,
) -> List[Chunk]:
    """
    Splits text into chunks of `chunk_size` characters with an `overlap`.
    """
    if not text:
        return []
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
        
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(text), step):
        content = text[i:i + chunk_size]
        # Avoid creating an empty chunk
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
        
        chunk = Chunk(
            chunk_id=chunk_id,
            content=content,
            metadata=chunk_metadata,
        )
        chunks.append(chunk)
        
        # If this chunk already reaches the end of the text, stop.
        if i + chunk_size >= len(text):
            break
            
    return chunks
