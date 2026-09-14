from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.query_transform import HyDE, MultiQuery, QueryRewriter


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
@pytest.mark.parametrize("factory,method", [
    (lambda llm: QueryRewriter(llm), "rewrite"),
    (lambda llm: MultiQuery(llm), "expand"),
])
def test_text_transforms_reject_empty_query(factory, method, query):
    llm = MagicMock()
    transform = factory(llm)

    with pytest.raises(ValueError, match="empty"):
        getattr(transform, method)(query)

    llm.generate.assert_not_called()


def test_rewriter_returns_clearer_query_and_uses_existing_client_contract():
    llm = MagicMock()
    llm.generate.return_value = "Điểm chuẩn ngành CNTT của PTIT năm 2024 là bao nhiêu?"

    result = QueryRewriter(llm).rewrite("Điểm chuẩn CNTT PTIT 2024?")

    assert result == "Điểm chuẩn ngành CNTT của PTIT năm 2024 là bao nhiêu?"
    kwargs = llm.generate.call_args.kwargs
    assert kwargs["temperature"] == 0.0
    assert kwargs["messages"][0]["role"] == "user"


@pytest.mark.parametrize("output", ["", "   ", None])
def test_rewriter_falls_back_on_empty_llm_output(output):
    llm = MagicMock()
    llm.generate.return_value = output

    assert QueryRewriter(llm).rewrite("Điểm chuẩn CNTT?") == "Điểm chuẩn CNTT?"


def test_rewriter_falls_back_on_llm_exception_or_timeout():
    llm = MagicMock()
    llm.generate.side_effect = TimeoutError("provider timeout")

    assert QueryRewriter(llm).rewrite("Học phí PTIT?") == "Học phí PTIT?"


@pytest.mark.parametrize(
    "unsafe_output",
    [
        "Điểm chuẩn ngành này năm gần nhất là bao nhiêu?",
        "Điểm chuẩn ngành CNTT của PTIT là bao nhiêu?",
        "Thông tin tuyển sinh PTIT năm 2024?",
        "Điểm chuẩn CNTT PTIT năm 2024 và 2025 là bao nhiêu?",
    ],
)
def test_rewriter_rejects_lost_entity_number_or_intent_term(unsafe_output):
    llm = MagicMock()
    llm.generate.return_value = unsafe_output
    original = "Điểm chuẩn CNTT PTIT năm 2024?"

    assert QueryRewriter(llm).rewrite(original) == original


def test_multi_query_returns_original_and_two_unique_variants():
    llm = MagicMock()
    llm.generate.return_value = (
        '["Mức điểm chuẩn của ngành CNTT PTIT năm 2024 là bao nhiêu?", '
        '"Năm 2024, ngành CNTT PTIT có điểm chuẩn bao nhiêu?"]'
    )

    result = MultiQuery(llm).expand("Điểm chuẩn CNTT PTIT năm 2024?")

    assert result == [
        "Điểm chuẩn CNTT PTIT năm 2024?",
        "Mức điểm chuẩn của ngành CNTT PTIT năm 2024 là bao nhiêu?",
        "Năm 2024, ngành CNTT PTIT có điểm chuẩn bao nhiêu?",
    ]


def test_multi_query_deduplicates_and_caps_result_at_three():
    llm = MagicMock()
    llm.generate.return_value = """1. Điểm chuẩn CNTT PTIT năm 2024?
2. Mức điểm chuẩn CNTT PTIT năm 2024 là bao nhiêu?
3. Mức điểm chuẩn CNTT PTIT năm 2024 là bao nhiêu?
4. Cho biết điểm chuẩn năm 2024 của CNTT PTIT?"""

    result = MultiQuery(llm).expand("Điểm chuẩn CNTT PTIT năm 2024?")

    assert len(result) == 3
    assert len({item.casefold() for item in result}) == 3


def test_multi_query_filters_wrong_intent_and_new_facts():
    llm = MagicMock()
    llm.generate.return_value = """[
        "Học phí ngành CNTT PTIT năm 2024 là bao nhiêu?",
        "Điểm chuẩn CNTT PTIT năm 2025 là bao nhiêu?",
        "Mức điểm chuẩn ngành CNTT PTIT năm 2024 là bao nhiêu?"
    ]"""

    result = MultiQuery(llm).expand("Điểm chuẩn CNTT PTIT năm 2024?")

    assert result == [
        "Điểm chuẩn CNTT PTIT năm 2024?",
        "Mức điểm chuẩn ngành CNTT PTIT năm 2024 là bao nhiêu?",
    ]


@pytest.mark.parametrize("failure", ["", "   ", None])
def test_multi_query_empty_output_falls_back_to_original(failure):
    llm = MagicMock()
    llm.generate.return_value = failure

    assert MultiQuery(llm).expand("Chỉ tiêu PTIT?") == ["Chỉ tiêu PTIT?"]


def test_multi_query_exception_falls_back_to_original():
    llm = MagicMock()
    llm.generate.side_effect = TimeoutError("timeout")

    assert MultiQuery(llm).expand("Chỉ tiêu PTIT?") == ["Chỉ tiêu PTIT?"]


class StubEmbedder:
    dimension = 3

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def embed_query(self, text):
        self.calls.append(text)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_hyde_embeds_hypothetical_document_not_as_final_answer():
    llm = MagicMock()
    llm.generate.return_value = "Tài liệu giả định về chỉ tiêu CNTT PTIT."
    embedder = StubEmbedder([np.array([1.0, 2.0, 3.0])])

    result = HyDE(llm, embedder).transform("Chỉ tiêu CNTT PTIT?")

    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert result.ndim == 1
    assert embedder.calls == ["Tài liệu giả định về chỉ tiêu CNTT PTIT."]


@pytest.mark.parametrize("generation", ["", "   ", None])
def test_hyde_empty_generation_embeds_original(generation):
    llm = MagicMock()
    llm.generate.return_value = generation
    embedder = StubEmbedder([np.array([1.0, 0.0, 0.0])])

    result = HyDE(llm, embedder).transform("Học phí PTIT?")

    assert result.shape == (3,)
    assert embedder.calls == ["Học phí PTIT?"]


def test_hyde_generation_exception_embeds_original():
    llm = MagicMock()
    llm.generate.side_effect = TimeoutError("timeout")
    embedder = StubEmbedder([np.array([0.0, 1.0, 0.0])])

    result = HyDE(llm, embedder).transform("Học phí PTIT?")

    assert result.shape == (3,)
    assert embedder.calls == ["Học phí PTIT?"]


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([[1.0, 2.0, 3.0]]),
        np.array([1.0, 2.0]),
        np.array([np.nan, 1.0, 2.0]),
        np.array([np.inf, 1.0, 2.0]),
    ],
)
def test_hyde_invalid_primary_vector_falls_back_to_original(invalid):
    llm = MagicMock()
    llm.generate.return_value = "Hypothetical document"
    embedder = StubEmbedder([invalid, np.array([3.0, 2.0, 1.0])])

    result = HyDE(llm, embedder).transform("Điểm chuẩn PTIT?")

    np.testing.assert_array_equal(result, np.array([3.0, 2.0, 1.0], dtype=np.float32))
    assert embedder.calls == ["Hypothetical document", "Điểm chuẩn PTIT?"]


def test_hyde_embedding_exception_falls_back_to_original():
    llm = MagicMock()
    llm.generate.return_value = "Hypothetical document"
    embedder = StubEmbedder([RuntimeError("model error"), np.ones(3)])

    result = HyDE(llm, embedder).transform("Điểm chuẩn PTIT?")

    assert result.shape == (3,)
    assert embedder.calls[-1] == "Điểm chuẩn PTIT?"


def test_hyde_raises_when_fallback_vector_is_also_invalid():
    llm = MagicMock()
    llm.generate.return_value = "Hypothetical document"
    embedder = StubEmbedder([np.array([np.nan, 1.0, 2.0]), np.array([[1.0, 2.0, 3.0]])])

    with pytest.raises(RuntimeError, match="either path"):
        HyDE(llm, embedder).transform("Điểm chuẩn PTIT?")


def test_hyde_rejects_empty_query_without_external_calls():
    llm = MagicMock()
    embedder = StubEmbedder([])

    with pytest.raises(ValueError, match="empty"):
        HyDE(llm, embedder).transform("  \n")

    llm.generate.assert_not_called()
    assert embedder.calls == []
