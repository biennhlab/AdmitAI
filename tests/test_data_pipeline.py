import json
from pathlib import Path

import numpy as np

from src.generation.rag_chain import RAGChain
from src.ingestion import Chunk, load_documents, naive_chunk, parse_markdown_document
from src.retrieval import NaiveDenseSearch, load_dense_index, save_dense_index


class FakeEmbedder:
    def embed(self, texts):
        return np.asarray([[len(text), text.count("PTIT")] for text in texts], dtype=float)

    def embed_query(self, query):
        return np.asarray([len(query), query.count("PTIT")], dtype=float)


class FakeLLM:
    def __init__(self):
        self.system_prompt = ""

    def generate(self, system_prompt, messages):
        self.system_prompt = system_prompt
        return "Câu trả lời grounded [1]."


def test_processed_markdown_metadata_and_duplicate_policy():
    tmp_path = Path("data/index/test-loader")
    processed = tmp_path / "processed" / "ptit"
    processed.mkdir(parents=True, exist_ok=True)
    source = processed / "source.md"
    source.write_text(
        '---\ndoc_id: "ptit-doc-1"\ntitle: "Nguồn PTIT"\n'
        'source_url: "https://ptit.example/source"\nsource_type: "official_web"\n'
        'published_at: "2026-01-01"\nretrieved_at: "2026-02-01"\n'
        'section: "Học phí"\nduplicate_of: ""\n---\nNội dung PTIT',
        encoding="utf-8",
    )
    duplicate = processed / "duplicate.md"
    duplicate.write_text(
        '---\ndoc_id: "ptit-doc-2"\ntitle: "Bản trùng"\n'
        'duplicate_of: "ptit-doc-1"\n---\nNội dung PTIT',
        encoding="utf-8",
    )

    result = load_documents(tmp_path / "processed")
    assert len(result.documents) == 1
    assert len(result.skipped_duplicates) == 1
    assert result.documents[0].metadata["source_url"] == "https://ptit.example/source"


def test_chunk_to_document_to_url_and_index_round_trip():
    tmp_path = Path("data/index/test-round-trip")
    metadata = {
        "doc_id": "ptit-doc-1",
        "title": "Nguồn PTIT",
        "source_url": "https://ptit.example/source",
        "section": "Phương thức tuyển sinh",
    }
    chunks = naive_chunk("PTIT tuyển sinh bằng nhiều phương thức.", 25, 5, metadata)
    embedder = FakeEmbedder()
    embeddings = embedder.embed([chunk.content for chunk in chunks])
    save_dense_index(
        tmp_path,
        chunks,
        embeddings,
        {"document_count": 1, "embedding_model": "fake", "corpus_fingerprint": "abc"},
    )
    loaded_chunks, loaded_embeddings, manifest = load_dense_index(tmp_path)
    assert manifest["chunk_count"] == len(chunks)
    assert loaded_chunks[0].metadata["doc_id"] == "ptit-doc-1"
    assert loaded_chunks[0].metadata["source_url"] == "https://ptit.example/source"
    np.testing.assert_array_equal(loaded_embeddings, embeddings)


def test_rag_response_has_real_structured_citation():
    chunk = Chunk(
        "chunk-1",
        "PTIT có phương thức xét tuyển theo điểm thi.",
        {
            "doc_id": "ptit-doc-1",
            "title": "Thông báo tuyển sinh",
            "source_url": "https://ptit.example/admissions",
            "page_number": 2,
            "section": "Phương thức 1",
        },
    )
    retriever = NaiveDenseSearch(FakeEmbedder(), [chunk])
    llm = FakeLLM()
    response = RAGChain(retriever, llm, min_score=-1).answer("Phương thức xét tuyển của PTIT?")
    assert response.citations[0]["source_url"] == "https://ptit.example/admissions"
    assert response.citations[0]["page"] == 2
    assert response.citations[0]["section"] == "Phương thức 1"
    assert "https://ptit.example/admissions" in llm.system_prompt


def test_low_score_falls_back_without_llm_call():
    class LowRetriever:
        def search(self, query, top_k):
            return [(Chunk("x", "unrelated"), 0.1)]

    llm = FakeLLM()
    response = RAGChain(LowRetriever(), llm, min_score=0.3).answer("Ngoài dữ liệu")
    assert response.route_type == "out_of_scope"
    assert response.citations == []
    assert llm.system_prompt == ""


def test_high_dense_score_without_query_evidence_falls_back():
    class MisleadingRetriever:
        def search(self, query, top_k):
            return [(Chunk("x", "Thông tin tuyển sinh ngành Công nghệ thông tin"), 0.9)]

    llm = FakeLLM()
    response = RAGChain(MisleadingRetriever(), llm, min_score=0.3).answer("Cách nấu phở bò?")
    assert response.route_type == "out_of_scope"
    assert llm.system_prompt == ""
