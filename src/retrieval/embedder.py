from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import settings

class Embedder:
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Embedder with a SentenceTransformer model.
        Loads model from config by default.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings into vectors.
        """
        if not texts:
            return np.array([])
        
        # We use normalize_embeddings=True for cosine similarity to just be dot product
        # but our requirements specifically ask to calculate cosine similarity with numpy
        embeddings = self.model.encode(texts, normalize_embeddings=False)
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string into a vector.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        embedding = self.model.encode(query, normalize_embeddings=False)
        return np.array(embedding)
