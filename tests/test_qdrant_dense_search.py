from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from src.ingestion import Chunk
from src.retrieval.dense_search import QdrantDenseSearch


class FakeEmbedder:
    def __init__(self, document_vectors, query_vectors=None):
        self.document_vectors = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in document_vectors.items()
        }
        self.query_vectors = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in (query_vectors or {}).items()
        }
        self.embed_calls = []
        self.query_calls = []

    def embed(self, texts):
        self.embed_calls.append(list(texts))
        return np.vstack([self.document_vectors[text] for text in texts])

    def embed_query(self, query):
        self.query_calls.append(query)
        return self.query_vectors[query]


@pytest.fixture
def qdrant():
    client = QdrantClient(":memory:")
    yield client
    client.close()


def _chunks():
    return [
        Chunk(
            chunk_id="chunk-a",
            content="Điểm chuẩn ngành Công nghệ thông tin",
            metadata={
                "doc_id": "doc-a",
                "source_file": "a.pdf",
                "source_url": "https://ptit.edu.vn/a",
                "title": "Đề án tuyển sinh",
                "page_number": 3,
                "section": "Điểm chuẩn",
                "citation_label": "PTIT 2026, tr. 3",
            },
        ),
        Chunk(
            chunk_id="chunk-b",
            content="Học phí chương trình đại trà",
            metadata={
                "doc_id": "doc-b",
                "source_file": "b.pdf",
                "source_url": "https://ptit.edu.vn/b",
                "title": "Thông báo học phí",
                "page_number": 7,
                "section": "Học phí",
                "citation_label": "PTIT 2026, tr. 7",
            },
        ),
        Chunk(
            chunk_id="chunk-c",
            content="Phương thức xét tuyển kết hợp",
            metadata={
                "doc_id": "doc-c",
                "source_file": "a.pdf",
                "source_url": "https://ptit.edu.vn/c",
                "title": "Phương thức tuyển sinh",
                "page_number": 11,
                "section": "Xét tuyển",
                "citation_label": "PTIT 2026, tr. 11",
            },
        ),
    ]


def test_index_preserves_chunk_vector_and_citation_mapping(qdrant):
    chunks = _chunks()
    vectors = {
        chunks[0].content: [1.0, 0.0, 0.0],
        chunks[1].content: [0.0, 1.0, 0.0],
        chunks[2].content: [0.0, 0.0, 1.0],
    }
    search = QdrantDenseSearch(qdrant, "chunks", FakeEmbedder(vectors))

    search.index(chunks)

    info = qdrant.get_collection("chunks")
    assert info.config.params.vectors.size == 3
    assert info.config.params.vectors.distance == models.Distance.COSINE
    points, _ = qdrant.scroll("chunks", limit=10, with_payload=True, with_vectors=True)
    assert len(points) == len(chunks)
    by_chunk_id = {point.payload["chunk_id"]: point for point in points}
    for chunk in chunks:
        point = by_chunk_id[chunk.chunk_id]
        assert point.payload["content"] == chunk.content
        assert point.payload["metadata"] == chunk.metadata
        for citation_key in (
            "doc_id",
            "source_file",
            "source_url",
            "title",
            "page_number",
            "section",
            "citation_label",
        ):
            assert point.payload[citation_key] == chunk.metadata[citation_key]
        np.testing.assert_allclose(point.vector, vectors[chunk.content])


def test_search_ranking_top_k_filters_and_vietnamese(qdrant):
    chunks = _chunks()
    embedder = FakeEmbedder(
        {
            chunks[0].content: [1.0, 0.0],
            chunks[1].content: [0.8, 0.6],
            chunks[2].content: [0.0, 1.0],
        },
        {"ngành nào có điểm chuẩn cao?": [1.0, 0.0]},
    )
    search = QdrantDenseSearch(qdrant, "chunks", embedder)
    search.index(chunks)

    results = search.search("ngành nào có điểm chuẩn cao?", top_k=2)
    assert [chunk.chunk_id for chunk, _ in results] == ["chunk-a", "chunk-b"]
    assert results[0][1] > results[1][1]
    assert results[0][0].metadata == chunks[0].metadata

    source_results = search.search(
        "ngành nào có điểm chuẩn cao?", top_k=20, filters={"source_file": "b.pdf"}
    )
    assert [chunk.chunk_id for chunk, _ in source_results] == ["chunk-b"]

    range_results = search.search(
        "ngành nào có điểm chuẩn cao?",
        top_k=20,
        filters={"page_number": {"gte": 7, "lte": 11}},
    )
    assert {chunk.chunk_id for chunk, _ in range_results} == {"chunk-b", "chunk-c"}

    assert search.search("ngành nào có điểm chuẩn cao?", top_k=0) == []
    assert search.search("", top_k=10) == []
    assert search.search("   ", top_k=10) == []
    assert embedder.query_calls == [
        "ngành nào có điểm chuẩn cao?",
        "ngành nào có điểm chuẩn cao?",
        "ngành nào có điểm chuẩn cao?",
    ]


def test_empty_index_returns_no_results_without_embedding(qdrant):
    embedder = MagicMock()
    search = QdrantDenseSearch(qdrant, "missing", embedder)
    search.index([])
    assert search.search("truy vấn", top_k=5) == []
    embedder.embed.assert_not_called()
    embedder.embed_query.assert_not_called()


def test_reindex_is_idempotent_and_updates_same_point(qdrant):
    chunks = _chunks()[:2]
    embedder = FakeEmbedder(
        {
            chunks[0].content: [1.0, 0.0],
            chunks[1].content: [0.0, 1.0],
            "Nội dung đã cập nhật": [0.6, 0.8],
        }
    )
    search = QdrantDenseSearch(qdrant, "chunks", embedder)
    search.index(chunks)
    original_point_id = search._point_id(chunks[0].chunk_id)

    updated = Chunk(
        chunk_id=chunks[0].chunk_id,
        content="Nội dung đã cập nhật",
        metadata={**chunks[0].metadata, "title": "Tiêu đề mới"},
    )
    search.index([updated])

    assert qdrant.count("chunks", exact=True).count == 2
    points = qdrant.retrieve("chunks", ids=[original_point_id], with_payload=True)
    assert points[0].payload["content"] == "Nội dung đã cập nhật"
    assert points[0].payload["title"] == "Tiêu đề mới"


def test_delete_by_source_is_exact_and_does_not_affect_other_sources(qdrant):
    chunks = _chunks()
    embedder = FakeEmbedder(
        {
            chunks[0].content: [1.0, 0.0],
            chunks[1].content: [0.0, 1.0],
            chunks[2].content: [0.7, 0.7],
        }
    )
    search = QdrantDenseSearch(qdrant, "chunks", embedder)
    search.index(chunks)

    search.delete_by_source("a.pdf")

    remaining, _ = qdrant.scroll("chunks", limit=10, with_payload=True)
    assert [point.payload["chunk_id"] for point in remaining] == ["chunk-b"]
    assert remaining[0].payload["source_file"] == "b.pdf"


def test_existing_collection_dimension_mismatch_is_non_destructive(qdrant):
    qdrant.create_collection(
        "chunks",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    chunk = _chunks()[0]
    search = QdrantDenseSearch(
        qdrant,
        "chunks",
        FakeEmbedder({chunk.content: [1.0, 0.0, 0.0]}),
    )

    with pytest.raises(ValueError, match="vector size 2.*returns 3.*no data was deleted"):
        search.index([chunk])
    assert qdrant.count("chunks", exact=True).count == 0
    assert qdrant.get_collection("chunks").config.params.vectors.size == 2


def test_existing_collection_distance_mismatch_is_non_destructive(qdrant):
    qdrant.create_collection(
        "chunks",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
    )
    chunk = _chunks()[0]
    search = QdrantDenseSearch(
        qdrant,
        "chunks",
        FakeEmbedder({chunk.content: [1.0, 0.0]}),
    )

    with pytest.raises(ValueError, match="requires Cosine.*no data was changed"):
        search.index([chunk])
    assert qdrant.count("chunks", exact=True).count == 0
    assert qdrant.get_collection("chunks").config.params.vectors.distance == models.Distance.DOT


def test_invalid_vectors_are_rejected_before_qdrant_write(qdrant):
    chunks = _chunks()[:2]
    embedder = MagicMock()
    embedder.embed.return_value = np.array([[1.0, 0.0]])
    search = QdrantDenseSearch(qdrant, "chunks", embedder)
    with pytest.raises(ValueError, match="aligned with chunks"):
        search.index(chunks)
    assert not qdrant.collection_exists("chunks")

    embedder.embed.return_value = np.array([[np.nan, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="NaN or infinite"):
        search.index(chunks)
    assert not qdrant.collection_exists("chunks")


def test_qdrant_unavailable_errors_are_not_silenced():
    client = MagicMock()
    client.collection_exists.side_effect = ConnectionError("offline")
    embedder = MagicMock()
    embedder.embed.return_value = np.array([[1.0, 0.0]])
    search = QdrantDenseSearch(client, "chunks", embedder)

    with pytest.raises(RuntimeError, match="Could not connect to Qdrant"):
        search.index([_chunks()[0]])
    with pytest.raises(RuntimeError, match="Could not connect to Qdrant"):
        search.search("truy vấn")
    with pytest.raises(RuntimeError, match="Could not connect to Qdrant"):
        search.delete_by_source("a.pdf")


def test_invalid_query_vector_and_top_k_are_rejected(qdrant):
    chunk = _chunks()[0]
    embedder = FakeEmbedder(
        {chunk.content: [1.0, 0.0]},
        {"bad": [np.inf, 0.0], "zero": [0.0, 0.0]},
    )
    search = QdrantDenseSearch(qdrant, "chunks", embedder)
    search.index([chunk])

    with pytest.raises(ValueError, match="NaN or infinite"):
        search.search("bad")
    with pytest.raises(ValueError, match="zero vector"):
        search.search("zero")
    with pytest.raises(TypeError, match="top_k must be an integer"):
        search.search("bad", top_k=1.5)
