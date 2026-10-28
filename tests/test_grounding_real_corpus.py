"""Regression for the reported IT cutoff question, using pinned read-only evidence."""
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from api.main import app
from api.routers import chat
from src.database.connection import get_db
from src.generation.grounding import RAGGroundingError, validate_answer
from src.generation.rag_chain import RAGChain
from src.generation.session_memory import SessionMemory
from src.ingestion.chunker import Chunk
from src.local_profile import LOCAL_INDEX_DIR
from src.retrieval.lite_selector import LiteEvidenceSelector

QUESTION = "Điểm chuẩn ngành Công nghệ thông tin mã 7480201 năm 2026 là bao nhiêu?"


@pytest.fixture
def pinned_it_chunk():
    root = Path(__file__).resolve().parents[1]
    dataset = json.loads((root / "eval/golden_dataset.json").read_text(encoding="utf-8"))
    case = next(case for case in dataset["cases"] if case["id"] == "cutoff_it_bvh_2026")
    assert case["question"] == QUESTION
    manifest = json.loads((root / LOCAL_INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    record = next(record for record in manifest["chunks"] if record["chunk_id"] == case["evidence_ids"][0])
    return Chunk(**record)


@pytest.mark.parametrize("answer", [
    "Điểm chuẩn ngành Công nghệ thông tin mã 7480201 năm 2026 là 23.00 [1].",
    "Điểm chuẩn ngành Công nghệ thông tin mã 7480201 năm 2026 là 23 điểm [1].",
    "Điểm chuẩn ngành Công nghệ thông tin (mã 7480201) năm 2026 là 23,00 [1].",
    "Ngành CNTT mã 7480201 năm 2026 có điểm chuẩn 23.0 [1].",
    "Ngành Công nghệ thông tin mã 7480201 có điểm chuẩn 23.00 [1].",
    "| Ngành | Mã xét tuyển | Điểm chuẩn năm 2026 |\n| --- | --- | --- |\n| Công nghệ thông tin | 7480201 | 23 [1] |",
    "| Ngành | Mã xét tuyển | Điểm chuẩn năm 2026 |\n| --- | --- | --- |\n| Công nghệ thông tin [1] | 7480201 | 23 |",
])
def test_actual_pinned_cutoff_relationship_accepts_supported_forms(pinned_it_chunk, answer):
    validate_answer(answer, [pinned_it_chunk])


@pytest.mark.parametrize("answer", [
    "Điểm chuẩn ngành Công nghệ thông tin mã 7480201 năm 2026 là 23.01 [1].",
    "Điểm chuẩn ngành Công nghệ thông tin mã 7480201 năm 2027 là 23 [1].",
    "Điểm chuẩn ngành An toàn thông tin mã 7480201 năm 2026 là 23 [1].",
    "Điểm chuẩn ngành Công nghệ thông tin mã 7480202 năm 2026 là 23 [1].",
    "Điểm chuẩn ngành Công nghệ thông tin mã 7480201 năm 2026 là 6 [1].",
    "Điểm chuẩn là 1 [1].",
    "Điểm chuẩn ngành Công nghệ thông tin mã 7480201 năm 2026 là 23 [99].",
    "Điểm chuẩn ngành Công nghệ thông tin mã 7480201 năm 2026 là 23.",
])
def test_actual_pinned_cutoff_still_rejects_wrong_entities_years_and_figures(pinned_it_chunk, answer):
    with pytest.raises(RAGGroundingError):
        validate_answer(answer, [pinned_it_chunk])


@pytest.mark.parametrize("academic", [False, True])
def test_admission_year_and_academic_range_are_distinct_supported_scopes(academic):
    source = Chunk("years", "Học phí đại trà năm học 2026-2027 được công bố.\n\nĐại trà: 35,90 triệu/năm.",
                   {"title": "Học phí năm 2026", "admission_year": "2026", "academic_year": "2026-2027", "heading": "Học phí"})
    answer = ("Học phí đại trà năm học 2026-2027 là 35,9 triệu/năm [1]." if academic
              else "Học phí đại trà năm 2026 là 35,9 triệu/năm [1].")
    validate_answer(answer, [source])
    with pytest.raises(RAGGroundingError):
        validate_answer("Học phí đại trà năm 2027 là 35,9 triệu/năm [1].", [source])


def test_table_major_name_and_code_cannot_borrow_another_rows_cutoff():
    source = Chunk("table", "| Ngành | Mã xét tuyển | Điểm chuẩn |\n| --- | --- | --- |\n"
                   "| Công nghệ thông tin | 7480201 | 23.00 |\n| An toàn thông tin | 7480202 | 25.22 |",
                   {"title": "Điểm chuẩn năm 2026", "admission_year": "2026", "academic_year": "2026-2027", "heading": "Điểm chuẩn"})
    validate_answer("Điểm chuẩn ngành Công nghệ thông tin mã 7480201 năm 2026 là 23 [1].", [source])
    with pytest.raises(RAGGroundingError) as error:
        validate_answer("Điểm chuẩn ngành Công nghệ thông tin mã 7480201 năm 2026 là 25,22 [1].", [source])
    assert error.value.reason == "unsupported_relationship"


def test_numeric_equivalence_never_rounds_distinct_values():
    source = Chunk("precise", "Tuition standard 123456789012345678901234567890.10.", {})
    validate_answer("Tuition standard 123456789012345678901234567890.1 [1].", [source])
    with pytest.raises(RAGGroundingError):
        validate_answer("Tuition standard 123456789012345678901234567890.11 [1].", [source])


@pytest.mark.parametrize('escaped', [r'2026\-2027', r'2026\/2027'])
def test_markdown_escaped_academic_ranges_keep_entity_value_boundaries(escaped):
    source = Chunk('escaped-years', f'Học phí đại trà năm học {escaped}: 35,9 triệu/năm. '
                   f'Học phí chất lượng cao năm học {escaped}: 58,3 triệu/năm.', {})
    validate_answer('Học phí đại trà năm học 2026-2027 là 35,9 triệu/năm [1].', [source])
    with pytest.raises(RAGGroundingError):
        validate_answer('Học phí đại trà năm học 2026-2027 là 58,3 triệu/năm [1].', [source])
    with pytest.raises(RAGGroundingError):
        validate_answer('Học phí đại trà năm học 2025-2026 là 35,9 triệu/năm [1].', [source])


@pytest.mark.parametrize('number', ['10.1', '10', '1'])
def test_numbered_section_headings_are_not_tuition_values(number):
    source = Chunk('outline', '*10\\.1\\. Học phí dự kiến:*\n\nHọc phí đại trà: 35,9 triệu/năm.', {})
    with pytest.raises(RAGGroundingError):
        validate_answer(f'Học phí là {number} [1].', [source])


def test_repeated_trailing_metric_does_not_erase_supported_values():
    source = Chunk('trailing-metric', 'Học phí đại trà: 35,9 triệu/năm, bằng khoảng 75% mức trần học phí theo', {})
    validate_answer('Học phí đại trà 35,9 triệu/năm [1].', [source])
    with pytest.raises(RAGGroundingError):
        validate_answer('Học phí chất lượng cao 35,9 triệu/năm [1].', [source])


def test_markdown_normalization_does_not_create_citation_markers():
    with pytest.raises(RAGGroundingError):
        validate_answer(r'Điểm SAT tối thiểu 1130/1600 \[1\].',
                        [Chunk('sat', 'Điểm SAT tối thiểu 1130/1600.', {})])


class _MemoryDB:
    def __init__(self):
        self.records = []
        self.committed = False

    async def get(self, model, identity):
        return None

    def add_all(self, records):
        self.records.extend(records)

    async def flush(self):
        pass

    async def commit(self):
        self.committed = True

    async def rollback(self):
        raise AssertionError("No real database work is performed")


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
@pytest.mark.parametrize("value", ["23", "23.00", "23,0", "23.01"])
async def test_reported_question_http_json_sse_uses_real_evidence_without_model_calls(pinned_it_chunk, monkeypatch, accept, value):
    answer = f"Điểm chuẩn ngành Công nghệ thông tin mã 7480201 năm 2026 là {value} [1]."
    retriever, llm = Mock(), Mock()
    retriever.search.return_value = [(pinned_it_chunk, .02)]
    llm.generate.return_value = answer
    chain = RAGChain(retriever, llm, reranker=LiteEvidenceSelector())
    monkeypatch.setattr(chat, "rag_chain", chain)
    monkeypatch.setattr(chat, "session_memory", SessionMemory())
    db = _MemoryDB()
    async def database():
        yield db
    monkeypatch.setattr(app, "dependency_overrides", {**app.dependency_overrides, get_db: database})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/chat", json={"message": QUESTION}, headers={"Accept": accept})
    if value == "23.01":
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "GENERATION_UNGROUNDED"
        assert not db.committed and db.records == []
    else:
        assert response.status_code == 200, response.text
        if accept == "application/json":
            result = response.json()
        else:
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            result = {**next(event for event in events if event["type"] == "metadata"),
                      "answer": "".join(event.get("text", "") for event in events)}
        assert result["answer"] == answer.replace(" [1]", "") and result["route_type"] == "general"
        assert result["citations"][0]["chunk_id"] == pinned_it_chunk.chunk_id
        assert db.committed
    llm.generate.assert_called_once()
    llm.generate_stream.assert_not_called()
