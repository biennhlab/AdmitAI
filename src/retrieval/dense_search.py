from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from numbers import Real
from typing import Any, List, Optional, Tuple

import numpy as np
from qdrant_client import models

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


_POINT_ID_NAMESPACE = uuid.UUID("260f54e7-d506-4f94-a0ac-b6eac955ff1b")
_PAYLOAD_RESERVED_KEYS = {"chunk_id", "content", "metadata"}


class QdrantDenseSearch:
    """Dense retrieval backed by one unnamed Qdrant cosine vector."""

    def __init__(self, qdrant_client: Any, collection: str, embedder: Embedder | None):
        if qdrant_client is None:
            raise ValueError("qdrant_client is required")
        if not isinstance(collection, str) or not collection.strip():
            raise ValueError("collection must be a non-empty string")
        self.qdrant_client = qdrant_client
        self.collection = collection.strip()
        self.embedder = embedder

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Use a native UUID as-is, otherwise derive a stable UUID from chunk_id."""
        try:
            return str(uuid.UUID(chunk_id))
        except (ValueError, TypeError, AttributeError):
            return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))

    @staticmethod
    def _json_compatible(metadata: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return json.loads(json.dumps(dict(metadata), ensure_ascii=False, default=str))
        except (TypeError, ValueError) as exc:
            raise ValueError("Chunk metadata must be JSON-compatible") from exc

    def _collection_exists(self) -> bool:
        try:
            return bool(self.qdrant_client.collection_exists(self.collection))
        except Exception as exc:
            raise RuntimeError(
                f"Could not connect to Qdrant collection '{self.collection}'"
            ) from exc

    @staticmethod
    def _vector_params(collection_info: Any) -> Any:
        vectors = collection_info.config.params.vectors
        if isinstance(vectors, Mapping):
            if len(vectors) != 1:
                raise ValueError("QdrantDenseSearch requires exactly one vector configuration")
            return next(iter(vectors.values()))
        return vectors

    def _validate_collection(self, vector_size: int) -> None:
        try:
            info = self.qdrant_client.get_collection(self.collection)
        except Exception as exc:
            raise RuntimeError(
                f"Could not inspect Qdrant collection '{self.collection}'"
            ) from exc
        params = self._vector_params(info)
        actual_size = int(params.size)
        distance = params.distance
        if actual_size != vector_size:
            raise ValueError(
                f"Qdrant collection '{self.collection}' has vector size {actual_size}, "
                f"but the embedder returns {vector_size}. Create a new {vector_size}-dimensional "
                "collection or explicitly recreate/reindex the old collection; no data was deleted."
            )
        if distance != models.Distance.COSINE:
            raise ValueError(
                f"Qdrant collection '{self.collection}' uses distance {distance}, "
                "but QdrantDenseSearch requires Cosine; no data was changed."
            )

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_exists():
            self._validate_collection(vector_size)
            return
        try:
            self.qdrant_client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        except Exception as exc:
            # A concurrent worker may have created it after collection_exists().
            if not self._collection_exists():
                raise RuntimeError(
                    f"Could not create Qdrant collection '{self.collection}'"
                ) from exc
        self._validate_collection(vector_size)

    def validate_ready(self, vector_size: int) -> None:
        """Fail clearly unless the configured collection can serve this vector size."""
        if not self._collection_exists():
            raise RuntimeError(f"Qdrant collection '{self.collection}' does not exist")
        self._validate_collection(vector_size)

    @staticmethod
    def _validate_matrix(vectors: Any, chunk_count: int) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != chunk_count or matrix.shape[1] <= 0:
            raise ValueError(
                "Embedding matrix must be 2D and aligned with chunks: "
                f"expected {chunk_count} rows, got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Embedding matrix contains NaN or infinite values")
        if np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise ValueError("Embedding matrix contains a zero vector")
        return matrix

    def index(self, chunks: list[Chunk], embeddings: Any = None) -> None:
        """Idempotently upsert chunks, optionally using precomputed embeddings."""
        if chunks is None:
            raise TypeError("chunks cannot be None")
        chunks = list(chunks)
        if not chunks:
            return

        seen_ids: set[str] = set()
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise TypeError(f"chunks[{index}] must be a Chunk")
            if not isinstance(chunk.chunk_id, str) or not chunk.chunk_id.strip():
                raise ValueError(f"chunks[{index}].chunk_id must be a non-empty string")
            if chunk.chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id in indexing batch: {chunk.chunk_id}")
            seen_ids.add(chunk.chunk_id)
            if not isinstance(chunk.content, str) or not chunk.content.strip():
                raise ValueError(f"chunks[{index}].content must be a non-empty string")
            if chunk.metadata is None or not isinstance(chunk.metadata, Mapping):
                raise TypeError(f"chunks[{index}].metadata must be a mapping")

        if embeddings is None:
            if self.embedder is None:
                raise ValueError("embedder is required when embeddings are not provided")
            vectors = self.embedder.embed([chunk.content for chunk in chunks])
        else:
            vectors = embeddings
        matrix = self._validate_matrix(vectors, len(chunks))
        self._ensure_collection(int(matrix.shape[1]))

        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, matrix, strict=True):
            metadata = self._json_compatible(chunk.metadata)
            payload = {
                **metadata,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                # A nested copy makes reconstruction lossless while root fields
                # remain directly filterable by Qdrant.
                "metadata": metadata,
            }
            points.append(
                models.PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )
        try:
            self.qdrant_client.upsert(
                collection_name=self.collection,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not upsert {len(points)} points into Qdrant collection "
                f"'{self.collection}'"
            ) from exc

    @staticmethod
    def _build_filter(filters: Any) -> models.Filter | None:
        if filters is None:
            return None
        if isinstance(filters, models.Filter):
            return filters
        if not isinstance(filters, Mapping):
            raise TypeError("filters must be a mapping, qdrant Filter, or None")
        conditions: list[models.FieldCondition] = []
        for key, value in filters.items():
            if not isinstance(key, str) or not key:
                raise ValueError("filter field names must be non-empty strings")
            if isinstance(value, Mapping):
                allowed = {"gt", "gte", "lt", "lte"}
                unknown = set(value) - allowed
                if unknown or not value:
                    raise ValueError(
                        f"Unsupported range filter for '{key}': {sorted(unknown)}"
                    )
                conditions.append(
                    models.FieldCondition(key=key, range=models.Range(**dict(value)))
                )
            elif isinstance(value, (list, tuple, set, frozenset)):
                values = list(value)
                if not values:
                    raise ValueError(f"Filter list for '{key}' cannot be empty")
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=values))
                )
            else:
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
        return models.Filter(must=conditions)

    def _query_points(
        self,
        query_vector: np.ndarray,
        top_k: int,
        query_filter: models.Filter | None,
    ) -> list[Any]:
        try:
            if hasattr(self.qdrant_client, "query_points"):
                response = self.qdrant_client.query_points(
                    collection_name=self.collection,
                    query=query_vector.tolist(),
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                )
                return list(response.points)
            return list(
                self.qdrant_client.search(
                    collection_name=self.collection,
                    query_vector=query_vector.tolist(),
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not search Qdrant collection '{self.collection}'"
            ) from exc

    @staticmethod
    def _chunk_from_payload(payload: Any) -> Chunk:
        if not isinstance(payload, Mapping):
            raise ValueError("Qdrant result is missing its payload")
        chunk_id = payload.get("chunk_id")
        content = payload.get("content")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Qdrant result payload has no valid chunk_id")
        if not isinstance(content, str):
            raise ValueError("Qdrant result payload has no valid content")
        nested = payload.get("metadata")
        if nested is not None and not isinstance(nested, Mapping):
            raise ValueError("Qdrant result payload has invalid metadata")
        metadata = dict(nested or {})
        metadata.update(
            {
                key: value
                for key, value in payload.items()
                if key not in _PAYLOAD_RESERVED_KEYS
            }
        )
        return Chunk(chunk_id=chunk_id, content=content, metadata=metadata)

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Any = None,
    ) -> list[tuple[Chunk, float]]:
        if not isinstance(query, str) or not query.strip():
            return []
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")
        if top_k <= 0:
            return []
        if not self._collection_exists():
            return []
        if self.embedder is None:
            raise RuntimeError("embedder is required for dense search")

        vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("Query embedding must be a non-empty one-dimensional vector")
        if not np.isfinite(vector).all():
            raise ValueError("Query embedding contains NaN or infinite values")
        if np.linalg.norm(vector) == 0:
            raise ValueError("Query embedding cannot be a zero vector")
        self._validate_collection(int(vector.shape[0]))
        query_filter = self._build_filter(filters)
        points = self._query_points(vector, top_k, query_filter)

        results: list[tuple[Chunk, float]] = []
        for point in points:
            score = point.score
            if not isinstance(score, Real) or not np.isfinite(score):
                raise ValueError("Qdrant returned an invalid relevance score")
            results.append((self._chunk_from_payload(point.payload), float(score)))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    def delete_by_source(self, source_file: str) -> None:
        """Delete only points whose root payload source_file exactly matches."""
        if not isinstance(source_file, str) or not source_file.strip():
            raise ValueError("source_file must be a non-empty string")
        if not self._collection_exists():
            return
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_file",
                        match=models.MatchValue(value=source_file),
                    )
                ]
            )
        )
        try:
            self.qdrant_client.delete(
                collection_name=self.collection,
                points_selector=selector,
                wait=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not delete source '{source_file}' from Qdrant collection "
                f"'{self.collection}'"
            ) from exc
