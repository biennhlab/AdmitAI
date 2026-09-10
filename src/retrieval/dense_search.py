import numpy as np
from typing import List, Tuple, Optional
from src.ingestion.chunker import Chunk
from src.retrieval.embedder import Embedder


def chunk_embedding_text(chunk: Chunk) -> str:
    """Include citation metadata that carries retrieval meaning, not payload IDs."""
    metadata = chunk.metadata or {}
    prefix = "\n".join(
        value
        for value in [str(metadata.get("title") or ""), str(metadata.get("section") or "")]
        if value
    )
    return f"{prefix}\n{chunk.content}" if prefix else chunk.content

class NaiveDenseSearch:
    def __init__(
        self,
        embedder: Embedder,
        chunks: List[Chunk],
        chunk_embeddings: Optional[np.ndarray] = None,
    ):
        """
        Initialize search by embedding all chunks in-memory.
        """
        self.embedder = embedder
        self.chunks = chunks
        if chunk_embeddings is not None:
            matrix = np.asarray(chunk_embeddings)
            if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
                raise ValueError("Precomputed embeddings must align with chunks")
            self.chunk_embeddings = matrix
        elif chunks:
            texts = [chunk_embedding_text(chunk) for chunk in chunks]
            self.chunk_embeddings = self.embedder.embed(texts)
        else:
            self.chunk_embeddings = np.array([])

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        """
        if a.size == 0 or b.size == 0:
            return 0.0
            
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            sim = float(np.dot(a, b) / (norm_a * norm_b))
        
        # Handle potential numerical precision issues leading to NaN
        if not np.isfinite(sim):
            return 0.0
            
        return sim

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """
        Search for the top_k most similar chunks to the query.
        """
        if not query or not query.strip():
            return []
            
        if not self.chunks or self.chunk_embeddings.size == 0:
            return []
            
        if top_k <= 0:
            return []
            
        query_emb = self.embedder.embed_query(query)
        
        # Handle zero vector query
        if np.linalg.norm(query_emb) == 0:
            return []
            
        if query_emb.shape[0] != self.chunk_embeddings.shape[1]:
            raise ValueError(f"Dimension mismatch: query is {query_emb.shape[0]} but chunks are {self.chunk_embeddings.shape[1]}")
            
        # Ensure we don't return more results than we have chunks
        actual_top_k = min(top_k, len(self.chunks))
        
        matrix = self.chunk_embeddings
        row_norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query_emb)
        denominator = row_norms * query_norm
        scores = np.divide(
            matrix @ query_emb,
            denominator,
            out=np.zeros_like(row_norms, dtype=float),
            where=denominator != 0,
        )
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        results = [(self.chunks[i], float(score)) for i, score in enumerate(scores)]
            
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:actual_top_k]
