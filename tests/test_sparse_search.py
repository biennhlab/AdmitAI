import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.sparse_search import BM25Search, tokenize


def chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, content=content)


def test_exact_keyword_ranks_high_and_preserves_score_chunk_mapping() -> None:
    chunks = [
        chunk("fees", "Thông tin học phí chương trình đại trà"),
        chunk("admission", "Chỉ tiêu tuyển sinh năm nay"),
        chunk("benchmark", "Điểm chuẩn quantum ngành công nghệ thông tin"),
        chunk("scholarship", "Các chương trình học bổng"),
    ]
    search = BM25Search()
    search.index(chunks)

    results = search.search("quantum", top_k=20)

    assert [item.chunk_id for item, _ in results] == ["benchmark"]
    raw_scores = search.bm25.get_scores(tokenize("quantum"))
    corpus_index = search.chunks.index(results[0][0])
    assert results[0][1] == pytest.approx(raw_scores[corpus_index])


def test_no_match_and_empty_corpus_return_no_results() -> None:
    empty = BM25Search()
    assert empty.search("học phí") == []

    search = BM25Search()
    search.index([chunk("one", "điểm chuẩn")])
    assert search.search("ký túc xá") == []
    assert search.search("") == []
    assert search.search("   ") == []


def test_duplicate_chunk_ids_are_last_write_wins_upserts() -> None:
    old = chunk("same", "nội dung cũ")
    replacement = chunk("same", "học bổng mới")
    extra = chunk("extra", "thông tin khác")
    search = BM25Search()

    search.index([old, replacement, extra])
    search.index([replacement])

    assert len(search.chunks) == 2
    assert search.chunks[0] is replacement
    assert search.search("cũ") == []
    assert search.search("HỌC BỔNG")[0][0] is replacement


def test_unicode_vietnamese_tokenization_is_case_insensitive() -> None:
    relevant = chunk("vi", "ĐIỂM chuẩn ngành Công nghệ thông tin")
    search = BM25Search()
    search.index([chunk("other", "học phí"), relevant, chunk("third", "ký túc xá")])

    assert tokenize("Điểm CHUẨN") == ["điểm", "chuẩn"]
    assert search.search("điểm chuẩn")[0][0] is relevant


def test_top_k_edges_and_matching_non_positive_score() -> None:
    only = chunk("only", "tuyển sinh")
    search = BM25Search()
    search.index([only])

    assert search.search("tuyển", top_k=0) == []
    assert search.search("tuyển", top_k=-1) == []
    assert search.search("tuyển", top_k=100)[0][0] is only
    assert search.search("tuyển", top_k=100)[0][1] <= 0
    with pytest.raises(TypeError, match="integer"):
        search.search("tuyển", top_k=1.5)


def test_rebuild_reflects_external_corpus_changes_and_rededuplicates() -> None:
    first = chunk("first", "học phí")
    added = chunk("second", "ký túc xá")
    replacement = chunk("first", "điểm chuẩn")
    search = BM25Search()
    search.index([first])

    search.chunks.extend([added, replacement])
    search.rebuild()

    assert [item.chunk_id for item in search.chunks] == ["first", "second"]
    assert search.chunks[0] is replacement
    assert search.search("học phí") == []
    assert search.search("điểm")[0][0] is replacement
    assert search.search("túc")[0][0] is added


def test_empty_upsert_does_not_clear_existing_corpus() -> None:
    search = BM25Search()
    search.index([chunk("one", "học phí")])
    search.index([])
    assert [item.chunk_id for item, _ in search.search("học")] == ["one"]
