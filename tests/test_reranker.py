from copy import deepcopy
from itertools import permutations
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import re

import numpy as np
import pytest

from src.config import settings
from src.ingestion.chunker import Chunk
from src.retrieval import reranker as reranker_module
from src.retrieval.document import chunk_search_text
from src.retrieval.reranker import Reranker
from src.generation.rag_chain import (
    FALLBACK_ANSWER, RAGChain, RAGRerankingError, RAGRetrievalError,
)


@pytest.fixture(autouse=True)
def mock_cross_encoder():
    with patch("src.retrieval.reranker.CrossEncoder") as factory:
        yield factory


def test_loads_configured_model(mock_cross_encoder):
    reranker = Reranker()
    mock_cross_encoder.assert_called_once_with(
        settings.RERANKER_MODEL,
        cache_folder=settings.HUGGINGFACE_CACHE_DIR,
    )
    assert reranker.model is mock_cross_encoder.return_value


def test_load_failure_is_immediate(mock_cross_encoder):
    error = OSError("model unavailable")
    mock_cross_encoder.side_effect = error
    with pytest.raises(OSError) as caught:
        Reranker()
    assert caught.value is error


def test_reranks_in_one_batch_and_preserves_inputs(mock_cross_encoder):
    first = Chunk("first", "First", {"provenance": {"dense": 1}})
    second = Chunk("second", "Second", {"provenance": {"bm25": 2}})
    candidates = [(first, 0.9), (second, 0.1)]
    before = deepcopy(candidates)
    metadata = [first.metadata, second.metadata]
    mock_cross_encoder.return_value.predict.return_value = np.array([-2.0, 3.0])

    results = Reranker().rerank("query", candidates)

    assert results == [(second, 0.1, 3.0), (first, 0.9, -2.0)]
    assert results[0][0] is second
    assert results[1][0] is first
    assert first.metadata is metadata[0]
    assert second.metadata is metadata[1]
    assert candidates == before
    mock_cross_encoder.return_value.predict.assert_called_once_with(
        [("query", "First"), ("query", "Second")]
    )


@pytest.mark.parametrize("top_k, expected_count", [(1, 1), (2, 2), (20, 3), (None, 2)])
def test_top_k(mock_cross_encoder, monkeypatch, top_k, expected_count):
    monkeypatch.setattr(settings, "RERANK_TOP_K", 2)
    candidates = [(Chunk(str(i), str(i)), 0.1) for i in range(3)]
    mock_cross_encoder.return_value.predict.return_value = [1, 3, 2]
    results = Reranker().rerank("query", candidates, top_k=top_k)
    assert [chunk.chunk_id for chunk, _, _ in results] == ["1", "2", "0"][:expected_count]
    assert len(mock_cross_encoder.return_value.predict.call_args.args[0]) == 3


@pytest.mark.parametrize("candidates, top_k", [([], None), ([(Chunk("a", "A"), 1.0)], 0), ([(Chunk("a", "A"), 1.0)], -1)])
def test_empty_results_skip_inference(mock_cross_encoder, candidates, top_k):
    assert Reranker().rerank("query", candidates, top_k) == []
    mock_cross_encoder.return_value.predict.assert_not_called()


@pytest.mark.parametrize("query", ["", "  ", "\t\n"])
def test_empty_query(mock_cross_encoder, query):
    with pytest.raises(ValueError, match="query cannot be empty"):
        Reranker().rerank(query, [(Chunk("a", "A"), 1.0)])
    mock_cross_encoder.return_value.predict.assert_not_called()


def test_deterministic_ties(mock_cross_encoder):
    candidates = [(Chunk("z", "Z"), 0.9), (Chunk("b", "B"), 0.2), (Chunk("a", "A"), 0.2)]
    mock_cross_encoder.return_value.predict.return_value = [1.0, 1.0, 1.0]
    reranker = Reranker()
    for order in permutations(candidates):
        results = reranker.rerank("query", list(order))
        assert [chunk.chunk_id for chunk, _, _ in results] == ["z", "a", "b"]


@pytest.mark.parametrize("scores", [[], [1, 2], 1.0, [[1.0]], [float("nan")], [float("inf")], [float("-inf")], ["invalid"], None])
def test_invalid_model_scores(mock_cross_encoder, scores):
    mock_cross_encoder.return_value.predict.return_value = scores
    with pytest.raises(RuntimeError, match="Reranker"):
        Reranker().rerank("query", [(Chunk("a", "A"), 1.0)])


def test_inference_error_preserves_cause(mock_cross_encoder):
    error = ValueError("inference error")
    mock_cross_encoder.return_value.predict.side_effect = error
    with pytest.raises(RuntimeError, match="^Reranker inference failed$") as caught:
        Reranker().rerank("query", [(Chunk("a", "A"), 1.0)])
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("heading_metadata, heading_text", [({"heading_path": ["Parent", "Child"]}, "Parent > Child"), ({"heading": "Heading"}, "Heading")])
def test_uses_search_text_helper(mock_cross_encoder, heading_metadata, heading_text):
    chunk = Chunk("a", "Content", {"title": "Title", "section": "Section", **heading_metadata})
    mock_cross_encoder.return_value.predict.return_value = [1.0]
    with patch.object(reranker_module, "chunk_search_text", wraps=chunk_search_text) as helper:
        Reranker().rerank("query", [(chunk, 0.1)])
    helper.assert_called_once_with(chunk)
    mock_cross_encoder.return_value.predict.assert_called_once_with(
        [("query", f"Title\nSection\n{heading_text}\nContent")]
    )


@pytest.mark.parametrize("candidate", [None, (), (Chunk("a", "A"),), (Chunk("a", "A"), 1, 2), ("a", 1.0), (Chunk("a", "A"), "1"), (Chunk("a", "A"), True), (Chunk("a", "A"), float("nan")), (Chunk("a", "A"), float("inf")), (Chunk(None, "A"), 1.0)])
def test_malformed_candidate(mock_cross_encoder, candidate):
    with pytest.raises(TypeError, match="Candidate 0"):
        Reranker().rerank("query", [candidate])
    mock_cross_encoder.return_value.predict.assert_not_called()


@pytest.mark.parametrize("kwargs, message", [({"query": None, "candidates": []}, "query"), ({"query": "query", "candidates": None}, "candidates"), ({"query": "query", "candidates": [], "top_k": True}, "top_k"), ({"query": "query", "candidates": [], "top_k": 1.5}, "top_k")])
def test_invalid_arguments(mock_cross_encoder, kwargs, message):
    with pytest.raises(TypeError, match=message):
        Reranker().rerank(**kwargs)
    mock_cross_encoder.return_value.predict.assert_not_called()


@pytest.mark.parametrize("scores", [[True], ["1.5"], [1 + 2j], np.array([1], dtype=object)])
def test_scores_must_be_numeric_not_silently_coerced(mock_cross_encoder, scores):
    mock_cross_encoder.return_value.predict.return_value = scores
    with pytest.raises(RuntimeError, match="numeric"):
        Reranker().rerank("query", [(Chunk("a", "A"), 0.1)])


@pytest.mark.parametrize("metadata", [None, {}, {"heading_path": [], "title": None}])
def test_missing_metadata_is_preserved(mock_cross_encoder, metadata):
    chunk = Chunk("a", "Content", metadata)
    mock_cross_encoder.return_value.predict.return_value = [0.5]
    result = Reranker().rerank("query", [(chunk, 0.1)])
    assert result == [(chunk, 0.1, 0.5)]
    assert result[0][0] is chunk
    assert chunk.metadata is metadata
    mock_cross_encoder.return_value.predict.assert_called_once_with([("query", "Content")])


@pytest.mark.parametrize("metadata", [[], ["title"], "title", 42, False])
def test_malformed_metadata_fails_before_inference(mock_cross_encoder, metadata):
    chunk = Chunk("a", "Content", metadata)
    with pytest.raises(TypeError, match="metadata"):
        Reranker().rerank("query", [(chunk, 0.1)])
    assert chunk.metadata is metadata
    mock_cross_encoder.return_value.predict.assert_not_called()


def test_duplicates_and_identical_ids_keep_score_object_alignment(mock_cross_encoder):
    first = Chunk("same-id", "First source", {"doc_id": "one"})
    second = Chunk("same-id", "Second source", {"doc_id": "two"})
    candidates = [(first, 0.2), (second, 0.2), (first, 0.2)]
    mock_cross_encoder.return_value.predict.return_value = [3, 3, 3]
    result = Reranker().rerank("query", candidates)
    assert len(result) == 3
    assert [id(row[0]) for row in result] == [id(first), id(second), id(first)]
    assert [row[1:] for row in result] == [(0.2, 3.0)] * 3


class RecordingRetriever:
    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return self.candidates[:top_k]


class RecordingLLM:
    def __init__(self, answer="Supported answer [1] [2]"):
        self.answer_text = answer
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.answer_text

    def generate_stream(self, **kwargs):
        self.calls.append(kwargs)
        # Exercise the real streaming buffer and marker parsing across fragments.
        yield from (self.answer_text[i:i + 3] for i in range(0, len(self.answer_text), 3))


def run_chain(chain, streaming, question="admission", history=None):
    if streaming:
        events = list(chain.answer_stream(question, history))
        metadata = next(event for event in events if event["type"] == "metadata")
        return SimpleNamespace(
            answer="".join(event["text"] for event in events if event["type"] == "chunk"),
            citations=metadata["citations"], route_type=metadata["route_type"],
        )
    return chain.answer(question, history)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("answer, expected_markers", [
    ("Claim [1] and another [2]", [1, 2]),
    ("Only second source [2] [2] [99]", [2]),
    ("Answer without markers", [1, 2]),
])
def test_integration_ranking_provenance_and_citation_mapping(
    mock_cross_encoder, monkeypatch, streaming, answer, expected_markers,
):
    from api.schemas import Citation
    from src.generation.assembler import ContextAssembler

    monkeypatch.setattr(settings, "RERANK_TOP_K", 2)
    chunks = [
        Chunk("a", "admission tuition deadline", {
            "doc_id": "doc-a", "page_number": 4, "heading": "Fees",
            "chunk_type": "table", "parent_chunk_id": "parent-a",
            "title": "Source A", "source_url": "https://example.org/a",
            "expansion_type": "exact",
        }),
        Chunk("b", "admission eligibility", {
            "doc_id": "doc-b", "page_number": 9, "heading": "Eligibility",
            "chunk_type": "child", "parent_chunk_id": "parent-b",
            "title": "Source B", "source_url": "https://example.org/b",
            "expansion_type": "neighbor",
        }),
        Chunk("c", "admission other evidence", {"doc_id": "doc-c"}),
    ]
    candidates = list(zip(chunks, [0.9, 0.7, 0.6]))
    before = deepcopy(candidates)
    retriever = RecordingRetriever(candidates)
    llm = RecordingLLM(answer)
    mock_cross_encoder.return_value.predict.return_value = [-10.0, 9.0, -20.0]
    chain = RAGChain(retriever, llm, top_k=17, reranker=Reranker())
    history = [{"role": "user", "content": "Earlier question"}]
    assembled_inputs = []
    original_assemble = ContextAssembler.assemble

    def capture_assemble(self, retrieved):
        assembled_inputs.extend(retrieved)
        return original_assemble(self, retrieved)

    with patch.object(ContextAssembler, "assemble", capture_assemble):
        result = run_chain(chain, streaming, "admission tuition deadline", history)

    assert result.route_type == "general"
    assert result.answer == answer
    assert retriever.calls == [("admission tuition deadline", 17)]
    assert [id(c) for c, _ in assembled_inputs] == [id(chunks[1]), id(chunks[0])]
    assert candidates == before
    for chunk, _ in assembled_inputs:
        assert chunk.metadata is chunks[1 if chunk.chunk_id == "b" else 0].metadata
        assert {"doc_id", "page_number", "heading", "chunk_type", "parent_chunk_id"} <= chunk.metadata.keys()
    prompt = llm.calls[0]["system_prompt"]
    # Neither assembler expansion priority nor lexical overlap may undo reranking.
    assert re.findall(r'<document id="\d+" doc_id="([^"]*)"', prompt) == ["doc-b", "doc-a"]
    assert 'retrieval_score="0.7000"' in prompt
    assert 'retrieval_score="0.9000"' in prompt
    assert "doc-c" not in prompt
    assert llm.calls[0]["messages"][0] == history[0]
    assert history == [{"role": "user", "content": "Earlier question"}]
    assert [c["marker"] for c in result.citations] == expected_markers
    for citation in result.citations:
        chunk, score = [candidates[1], candidates[0]][citation["marker"] - 1]
        assert citation["chunk_id"] == chunk.chunk_id
        assert citation["doc_id"] == chunk.metadata["doc_id"]
        assert citation["page"] == chunk.metadata["page_number"]
        assert citation["source_url"] == chunk.metadata["source_url"]
        assert citation["score"] == score
        assert citation["snippet"] in chunk.content
        assert Citation(**citation).model_dump() == citation


@pytest.mark.parametrize("streaming", [False, True])
def test_threshold_filters_retrieval_score_before_rerank(mock_cross_encoder, streaming):
    keep = Chunk("keep", "admission supported")
    drop = Chunk("drop", "admission unsupported")
    mock_cross_encoder.return_value.predict.return_value = [-100.0]
    llm = RecordingLLM()
    result = run_chain(RAGChain(
        RecordingRetriever([(drop, 0.2), (keep, 0.8)]), llm,
        min_score=0.5, reranker=Reranker(),
    ), streaming)
    assert result.route_type == "general"
    assert result.citations[0]["chunk_id"] == "keep"
    assert result.citations[0]["score"] == 0.8
    mock_cross_encoder.return_value.predict.assert_called_once_with([("admission", "admission supported")])


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("case", ["question", "candidates", "filtered", "zero_k", "lexical"])
def test_empty_or_unsupported_context_skips_generation(mock_cross_encoder, monkeypatch, streaming, case):
    monkeypatch.setattr(settings, "RERANK_TOP_K", 0 if case == "zero_k" else 5)
    candidates = [] if case == "candidates" else [(Chunk("a", "admission"), 0.1)]
    retriever = RecordingRetriever(candidates)
    llm = RecordingLLM()
    mock_cross_encoder.return_value.predict.return_value = [1.0]
    chain = RAGChain(retriever, llm, min_score=0.5 if case == "filtered" else None, reranker=Reranker())
    question = "  " if case == "question" else "unrelated topic" if case == "lexical" else "admission"
    result = run_chain(chain, streaming, question)
    assert result.answer == FALLBACK_ANSWER
    assert result.citations == []
    assert result.route_type == "out_of_scope"
    assert llm.calls == []
    if case != "lexical":
        mock_cross_encoder.return_value.predict.assert_not_called()
    if case == "question":
        assert retriever.calls == []


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("metadata", [None, {}])
def test_integration_missing_metadata(mock_cross_encoder, streaming, metadata):
    chunk = Chunk("a", "admission", metadata)
    mock_cross_encoder.return_value.predict.return_value = [1.0]
    result = run_chain(RAGChain(RecordingRetriever([(chunk, 0.4)]), RecordingLLM(), reranker=Reranker()), streaming)
    assert result.citations[0]["doc_id"] == ""
    assert result.citations[0]["page"] is None
    assert result.citations[0]["title"] == "Chunk a"
    assert chunk.metadata is metadata


@pytest.mark.parametrize("streaming", [False, True])
def test_duplicate_content_does_not_duplicate_citations(mock_cross_encoder, streaming):
    a = Chunk("a", "admission alpha", {"doc_id": "a"})
    duplicate = Chunk("duplicate", " admission  alpha ", {"doc_id": "other"})
    b = Chunk("b", "admission beta", {"doc_id": "b"})
    candidates = [(a, 0.8), (a, 0.8), (duplicate, 0.7), (b, 0.6)]
    mock_cross_encoder.return_value.predict.return_value = [4, 4, 3, 2]
    result = run_chain(RAGChain(RecordingRetriever(candidates), RecordingLLM(), reranker=Reranker()), streaming)
    assert [c["chunk_id"] for c in result.citations] == ["a", "b"]
    assert [c["marker"] for c in result.citations] == [1, 2]


@pytest.mark.parametrize("streaming", [False, True])
def test_assembler_budget_still_applies(mock_cross_encoder, streaming):
    oversized = Chunk("large", "admission " * 1300, {"chunk_type": "table"})
    small = Chunk("small", "admission row intact")
    mock_cross_encoder.return_value.predict.return_value = [10, 1]
    result = run_chain(RAGChain(
        RecordingRetriever([(oversized, 0.9), (small, 0.8)]), RecordingLLM(), reranker=Reranker(),
    ), streaming)
    assert [c["chunk_id"] for c in result.citations] == ["small"]
    assert result.citations[0]["snippet"] == small.content


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failure", ["predict", "count", "nan", "metadata"])
def test_model_failures_never_fall_back_or_generate(mock_cross_encoder, streaming, failure):
    model = mock_cross_encoder.return_value
    error = OSError("model unavailable")
    model.predict.return_value = [] if failure == "count" else [float("nan")] if failure == "nan" else [1.0]
    if failure == "predict":
        model.predict.side_effect = error
    chunk = Chunk("a", "admission", "broken" if failure == "metadata" else {})
    llm = RecordingLLM()
    chain = RAGChain(RecordingRetriever([(chunk, 0.8)]), llm, reranker=Reranker())
    with pytest.raises(RAGRerankingError) as caught:
        run_chain(chain, streaming)
    assert isinstance(caught.value, RAGRetrievalError)
    assert caught.value.__cause__ is not None
    if failure == "predict":
        assert caught.value.__cause__.__cause__ is error
    assert llm.calls == []


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failure", ["none", "empty", "short", "pair", "invented", "clone", "swapped_score", "duplicate", "nan", "inf", "string", "bool", "unsorted"])
def test_injected_reranker_contract_is_checked(streaming, failure):
    a, b = Chunk("a", "admission alpha"), Chunk("b", "admission beta")
    valid = [(a, 0.8, 2.0), (b, 0.6, 1.0)]
    outputs = {
        "none": None, "empty": [], "short": valid[:1],
        "pair": [(a, 0.8), (b, 0.6)],
        "invented": [(Chunk("unknown", "invented"), 0.8, 2.0), valid[1]],
        "clone": [(deepcopy(a), 0.8, 2.0), valid[1]],
        "swapped_score": [(a, 0.6, 2.0), (b, 0.8, 1.0)],
        "duplicate": [valid[0], valid[0]],
        "nan": [(a, 0.8, float("nan")), valid[1]],
        "inf": [(a, 0.8, float("inf")), valid[1]],
        "string": [(a, 0.8, "2.0"), valid[1]],
        "bool": [(a, 0.8, True), valid[1]],
        "unsorted": list(reversed(valid)),
    }
    # Only this adversarial dependency is fake; the orchestration is real.
    dependency = SimpleNamespace(rerank=lambda *args, **kwargs: outputs[failure])
    llm = RecordingLLM()
    with pytest.raises(RAGRerankingError):
        run_chain(RAGChain(RecordingRetriever([(a, 0.8), (b, 0.6)]), llm, reranker=dependency), streaming)
    assert llm.calls == []


@pytest.mark.parametrize("streaming", [False, True])
def test_legacy_constructor_and_duck_typed_chunks_remain_supported(mock_cross_encoder, streaming):
    chunk = SimpleNamespace(chunk_id=1, content="admission", metadata={"source": "legacy.pdf"})
    # All original positional constructor arguments remain valid.
    chain = RAGChain(RecordingRetriever([(chunk, 0.9)]), RecordingLLM(), 5, None, 0.34, None, None)
    result = run_chain(chain, streaming)
    assert result.citations[0]["source"] == "legacy.pdf"
    assert result.citations[0]["chunk_id"] == "1"
    mock_cross_encoder.assert_not_called()


@pytest.mark.parametrize("streaming", [False, True])
def test_normalized_rewritten_query_reaches_reranker(mock_cross_encoder, streaming):
    rewriter = MagicMock()
    rewriter.rewrite.return_value = "admission rewritten"
    retriever = RecordingRetriever([(Chunk("a", "admission"), 0.8)])
    mock_cross_encoder.return_value.predict.return_value = [1.0]
    llm = RecordingLLM()
    run_chain(RAGChain(retriever, llm, query_rewriter=rewriter, reranker=Reranker()), streaming, "admission cntt")
    assert "Công nghệ thông tin" in rewriter.rewrite.call_args.args[0]
    assert retriever.calls == [("admission rewritten", 5)]
    mock_cross_encoder.return_value.predict.assert_called_once_with([("admission rewritten", "admission")])
    assert "admission cntt" in llm.calls[0]["messages"][-1]["content"]


@pytest.mark.parametrize("streaming", [False, True])
def test_real_dense_bm25_fusion_pipeline_is_unchanged(mock_cross_encoder, streaming):
    from src.retrieval.dense_search import NaiveDenseSearch
    from src.retrieval.sparse_search import BM25Search
    from src.retrieval.fusion import HybridRetriever

    chunks = [Chunk(str(i), f"admission course {i}", {"doc_id": str(i)}) for i in range(3)]
    before = deepcopy(chunks)
    embedder = SimpleNamespace(embed_query=lambda query: np.array([1.0, 0.0]))
    dense = NaiveDenseSearch(embedder, chunks, chunk_embeddings=np.array([[1, 0], [0.8, 0.2], [0, 1]]))
    sparse = BM25Search()
    sparse.index(chunks)
    hybrid = HybridRetriever(dense, sparse)
    original = hybrid.search("admission", top_k=3)
    target = original[-1][0]
    mock_cross_encoder.return_value.predict.side_effect = lambda pairs: [
        10.0 if text == chunk_search_text(target) else -1.0 for _, text in pairs
    ]
    result = run_chain(RAGChain(hybrid, RecordingLLM("Answer [1]"), top_k=3, reranker=Reranker()), streaming)
    assert result.citations[0]["chunk_id"] == target.chunk_id
    assert result.citations[0]["score"] == round(original[-1][1], 6)
    assert hybrid.search("admission", top_k=3) == original
    assert chunks == before


@pytest.fixture
def isolated_chat_state(monkeypatch):
    from api.routers import chat
    for name in ("rag_chain", "rag_manifest", "rag_initialization_error", "rag_public_error"):
        monkeypatch.setattr(chat, name, None)
    return chat


def test_api_initialization_enables_real_reranker(mock_cross_encoder, isolated_chat_state):
    chat = isolated_chat_state
    manifest = {"embedding_model": settings.EMBEDDING_MODEL, "document_count": 1, "chunk_count": 1}
    retriever = RecordingRetriever([(Chunk("a", "admission"), 0.8)])
    with patch.object(chat, "LLMClient"):
        assert chat.initialize_rag(retriever=retriever, manifest=manifest) is manifest
    assert isinstance(chat.rag_chain.reranker, Reranker)
    assert chat.rag_chain.retriever is retriever
    assert chat.rag_chain.top_k == settings.RETRIEVAL_TOP_K
    assert chat.rag_chain.min_score is None
    mock_cross_encoder.assert_called_once_with(
        settings.RERANKER_MODEL,
        cache_folder=settings.HUGGINGFACE_CACHE_DIR,
    )


def test_api_model_load_failure_leaves_service_unavailable(mock_cross_encoder, isolated_chat_state):
    chat = isolated_chat_state
    error = OSError("private model path")
    mock_cross_encoder.side_effect = error
    manifest = {"embedding_model": settings.EMBEDDING_MODEL, "document_count": 1, "chunk_count": 1}
    with patch.object(chat, "LLMClient"), pytest.raises(OSError) as caught:
        chat.initialize_rag(retriever=RecordingRetriever([]), manifest=manifest)
    assert caught.value is error
    assert chat.rag_chain is None
    assert chat.rag_manifest is None
    assert chat.rag_public_error == chat.RAG_UNAVAILABLE_MESSAGE
    assert "private" not in chat.rag_public_error


@pytest.mark.parametrize("streaming", [False, True])
def test_retrieval_error_still_preserves_cause_and_skips_reranker(mock_cross_encoder, streaming):
    error = ConnectionError("retrieval unavailable")
    retriever = MagicMock()
    retriever.search.side_effect = error
    llm = RecordingLLM()
    chain = RAGChain(retriever, llm, reranker=Reranker())
    with pytest.raises(RAGRetrievalError) as caught:
        run_chain(chain, streaming)
    assert caught.value.__cause__ is error
    assert not isinstance(caught.value, RAGRerankingError)
    assert llm.calls == []
    mock_cross_encoder.return_value.predict.assert_not_called()


@pytest.mark.parametrize("streaming", [False, True])
def test_equal_scores_same_document_get_stable_distinct_markers(mock_cross_encoder, streaming):
    candidates = [
        (Chunk(key, f"admission course {key}", {"doc_id": "shared", "page_number": page}), 0.5)
        for page, key in enumerate(["c", "a", "b"], 1)
    ]
    mock_cross_encoder.return_value.predict.return_value = [1.0] * 3
    snapshots = []
    for order in permutations(candidates):
        result = run_chain(RAGChain(
            RecordingRetriever(list(order)), RecordingLLM("Evidence [1] [2] [3]"), reranker=Reranker(),
        ), streaming)
        snapshots.append(result.citations)
    assert all(snapshot == snapshots[0] for snapshot in snapshots)
    assert [c["chunk_id"] for c in snapshots[0]] == ["a", "b", "c"]
    assert [c["page"] for c in snapshots[0]] == [2, 3, 1]
    assert [c["marker"] for c in snapshots[0]] == [1, 2, 3]


def test_api_dense_initialization_keeps_threshold_and_backend(mock_cross_encoder, isolated_chat_state):
    chat = isolated_chat_state
    chunks = [Chunk("a", "admission")]
    manifest = {"embedding_model": settings.EMBEDDING_MODEL, "document_count": 1, "chunk_count": 1}
    with (
        patch.object(chat, "load_dense_index", return_value=(chunks, np.array([[1, 0]]), manifest)),
        patch.object(chat, "Embedder"),
        patch.object(chat, "LLMClient"),
    ):
        chat.initialize_rag(index_dir="unused-test-index")
    assert isinstance(chat.rag_chain.retriever, chat.NaiveDenseSearch)
    assert chat.rag_chain.retriever.chunks is chunks
    assert chat.rag_chain.min_score == settings.RETRIEVAL_MIN_SCORE
    assert isinstance(chat.rag_chain.reranker, Reranker)
    mock_cross_encoder.assert_called_once()


@pytest.mark.asyncio
async def test_api_stream_reports_reranking_failure_safely(mock_cross_encoder, isolated_chat_state, monkeypatch):
    from api.schemas import ChatRequest
    from fastapi import HTTPException
    from src.generation.session_memory import SessionMemory

    chat = isolated_chat_state
    monkeypatch.setattr(chat, "session_memory", SessionMemory())
    mock_cross_encoder.return_value.predict.side_effect = OSError("private model secret")
    llm = RecordingLLM()
    chat.rag_chain = RAGChain(
        RecordingRetriever([(Chunk("a", "admission"), 0.8)]), llm, reranker=Reranker(),
    )
    # Reranking fails before the first event, so the HTTP status is still mutable.
    with pytest.raises(HTTPException) as caught:
        await chat.chat(ChatRequest(message="admission"), db=MagicMock())
    assert caught.value.status_code == 503
    assert caught.value.detail["code"] == "RAG_UNAVAILABLE"
    assert "private model secret" not in str(caught.value.detail)
    assert isinstance(caught.value.__cause__, RAGRerankingError)
    assert llm.calls == []
