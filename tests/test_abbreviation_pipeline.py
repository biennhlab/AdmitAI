from __future__ import annotations

from unittest.mock import MagicMock

from src.generation.rag_chain import RAGChain
from src.query_transform import AbbreviationNormalizer


class _Chunk:
    chunk_id = "cntt-2025"
    content = "Điểm chuẩn CNTT, ngành Công nghệ thông tin năm 2025."
    metadata = {"source": "Điểm chuẩn 2025"}


class _RecordingRetriever:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, query: str, top_k: int):
        self.queries.append(query)
        return [(_Chunk(), 0.9)]


def test_rag_chain_normalizes_then_rewrites_before_retrieval():
    retriever = _RecordingRetriever()
    rewriter = MagicMock()
    rewriter.rewrite.return_value = (
        "Điểm chuẩn ngành cntt (Công nghệ thông tin) năm 2025 là bao nhiêu?"
    )
    llm = MagicMock()
    llm.generate.return_value = "Điểm chuẩn được công bố trong nguồn [1]."
    question = "điểm chuẩn cntt 2025"

    response = RAGChain(
        retriever,
        llm,
        query_rewriter=rewriter,
        abbreviation_normalizer=AbbreviationNormalizer(),
    ).answer(question)

    rewriter.rewrite.assert_called_once_with(
        "điểm chuẩn cntt (Công nghệ thông tin) 2025"
    )
    assert retriever.queries == [rewriter.rewrite.return_value]
    assert question in llm.generate.call_args.kwargs["messages"][-1]["content"]
    assert response.route_type == "general"


def test_rag_chain_normal_query_is_unchanged_before_optional_rewrite():
    retriever = _RecordingRetriever()
    llm = MagicMock()
    llm.generate.return_value = "Thông tin trong nguồn [1]."
    query = "học phí Công nghệ thông tin"

    RAGChain(retriever, llm).answer(query)

    assert retriever.queries == [query]
