import numpy as np
from typing import List, Tuple
from src.ingestion.chunker import Chunk
from src.retrieval.embedder import Embedder

class NaiveDenseSearch:
    def __init__(self, embedder: Embedder, chunks: List[Chunk]):
        """
        Initialize search by embedding all chunks in-memory.
        """
        self.embedder = embedder
        self.chunks = chunks
        if chunks:
            texts = [chunk.content for chunk in chunks]
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
            
        sim = float(np.dot(a, b) / (norm_a * norm_b))
        
        # Handle potential numerical precision issues leading to NaN
        if np.isnan(sim):
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
        
        results = []
        for i, chunk_emb in enumerate(self.chunk_embeddings):
            score = self.cosine_similarity(query_emb, chunk_emb)
            results.append((self.chunks[i], score))
            
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:actual_top_k]
