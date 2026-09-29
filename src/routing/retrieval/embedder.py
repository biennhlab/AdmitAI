from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from threading import Lock
from typing import Any, ClassVar, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings


class Embedder:
    """SentenceTransformer wrapper shared by the dense retrieval backends."""

    # API workers commonly construct more than one retriever. Reusing the model
    # here prevents another multi-GB BGE-M3 load in the same Python process.
    _model_cache: ClassVar[dict[tuple[str, object], Any]] = {}
    _model_cache_lock: ClassVar[Lock] = Lock()

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("Embedding model name cannot be empty")
        self.model_name = self.model_name.strip()
        self.model = self._get_or_load_model(self.model_name)
        self._dimension = self._model_dimension()

    @classmethod
    def _get_or_load_model(cls, model_name: str) -> SentenceTransformer:
        # Including the factory in the key keeps test/injected factories isolated
        # while still caching normal SentenceTransformer instances by model name.
        cache_key = (model_name, SentenceTransformer)
        with cls._model_cache_lock:
            cached = cls._model_cache.get(cache_key)
            if cached is not None:
                return cached
            try:
                model = SentenceTransformer(model_name, local_files_only=True)
            # Some transformers releases surface an incomplete local snapshot
            # as AttributeError rather than OSError. In that case, let the
            # normal Hugging Face resolution path repair/download the model.
            except (OSError, ValueError, AttributeError):
                model = SentenceTransformer(model_name)
            cls._model_cache[cache_key] = model
            return model

    def _model_dimension(self) -> int | None:
        getter = getattr(self.model, "get_sentence_embedding_dimension", None)
        if not callable(getter):
            return None
        dimension = getter()
        if isinstance(dimension, Integral) and not isinstance(dimension, bool) and dimension > 0:
            return int(dimension)
        return None

    @property
    def dimension(self) -> int | None:
        """Return the model output size once known (1024 for BGE-M3)."""
        return self._dimension

    def _validate_matrix(self, value: Any, expected_rows: int) -> np.ndarray:
        matrix = np.asarray(value, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] <= 0:
            raise ValueError(
                "Embedding model returned an invalid matrix: "
                f"expected ({expected_rows}, dimension), got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Embedding model returned NaN or infinite values")
        if self._dimension is None:
            self._dimension = int(matrix.shape[1])
        elif matrix.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension changed from {self._dimension} to {matrix.shape[1]}"
            )
        return matrix

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Embed non-empty texts as a finite ``float32`` matrix."""
        if texts is None:
            raise TypeError("texts must be an iterable of strings, not None")
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be an iterable of strings, not a single string")
        try:
            values = list(texts)
        except TypeError as exc:
            raise TypeError("texts must be an iterable of strings") from exc
        if not values:
            return np.empty((0, self._dimension or 0), dtype=np.float32)
        for index, text in enumerate(values):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"texts[{index}] must be a non-empty string")

        embeddings = self.model.encode(values, normalize_embeddings=False)
        return self._validate_matrix(embeddings, len(values))

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one non-empty query as a finite one-dimensional vector."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty")
        vector = np.asarray(
            self.model.encode(query, normalize_embeddings=False),
            dtype=np.float32,
        )
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(
                "Embedding model returned an invalid query vector: "
                f"expected one dimension, got {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("Embedding model returned NaN or infinite values")
        if self._dimension is None:
            self._dimension = int(vector.shape[0])
        elif vector.shape[0] != self._dimension:
            raise ValueError(
                f"Embedding dimension changed from {self._dimension} to {vector.shape[0]}"
            )
        return vector
