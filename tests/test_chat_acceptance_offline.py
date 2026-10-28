"""The eight requested questions with pinned candidates and a mocked provider.

This tests transport and actual evidence support; it is not live retrieval or
browser/provider acceptance. No corpus record or golden expectation is edited.
"""
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from api.main import app
from api.routers import chat
from src.database.connection import get_db
from src.generation.rag_chain import FALLBACK_ANSWER, RAGChain
from src.generation.session_memory import SessionMemory
from src.ingestion.chunker import Chunk
from src.local_profile import LOCAL_INDEX_DIR
from src.retrieval.lite_selector import LiteEvidenceSelector

SAT = '53fe7fad77a24410ad395e9f'
STANDARD = '11187c1a63a860be9fe36385'
QUALITY = 'f2b22408cd0bd13c59695829'
INTERNATIONAL = '7113e0d59394ecf7857b27a5'
IT = '71cc0aa0c66a3a0b1cc769d1'
METHODS = ['9999fe294af53f340986ee46', SAT, '66d2863da2871315c6eb5ead',
           '6c913802f7f05cd0cc24121f', 'bd7f622bf4142bd0f15f2f9d']
QUESTIONS = {
    'A': 'PTIT có những phương thức tuyển sinh nào trong năm 2026?',
    'B': 'Điểm SAT hoặc ACT tối thiểu để đăng ký xét tuyển PTIT năm 2026?',
    'C': 'Học phí PTIT năm học 2026–2027 là bao nhiêu?',
    'D': 'Học phí đại trà PTIT 2026–2027 có phải 58,3 triệu đồng/năm không?',
    'E': 'Có phải SAT cần 25 điểm và ACT cần 1130 điểm không?',
    'F': 'Điểm chuẩn ngành CNTT mã 7480201 năm 2026 là bao nhiêu?',
    'G': 'điểm chuẩn cn1',
    'H': 'PTIT có tuyển ngành Thú y năm 2026 không?',
}


@pytest.fixture(scope='module')
def records():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / LOCAL_INDEX_DIR / 'manifest.json').read_text(encoding='utf-8'))
    return {record['chunk_id']: Chunk(**record) for record in manifest['chunks']}


def answer_for(case, markers):
    if case == 'A':
        labels = ['Xét tuyển tài năng', 'Xét tuyển dựa vào kết quả Chứng chỉ đánh giá năng lực quốc tế (SAT/ACT)',
                  'Xét tuyển dựa vào kết quả trong các kỳ thi đánh giá năng lực, đánh giá tư duy',
                  'Xét tuyển kết hợp giữa chứng chỉ tiếng Anh quốc tế (IELTS, TOEFL) với kết quả học tập ở bậc THPT',
                  'Xét tuyển dựa vào kết quả thi tốt nghiệp THPT năm 2026']
        return '\n'.join(f'- {label} [{markers[source]}].' for source, label in zip(METHODS, labels))
    if case in ('B', 'E'):
        return f'Điểm SAT tối thiểu 1130/1600; điểm ACT tối thiểu 25/36 [{markers[SAT]}].'
    if case in ('C', 'D'):
        fees = [(STANDARD, 'đại trà', '35,9', '40,7'),
                (QUALITY, 'chất lượng cao', '44,8', '58,3'),
                (INTERNATIONAL, 'liên kết quốc tế', '58,3', '67,2')]
        return '\n'.join(f'- Học phí chương trình {program} năm học 2026-2027 dự kiến từ khoảng '
                         f'{low} đến {high} triệu đồng/năm [{markers[source]}].'
                         for source, program, low, high in fees if source in markers)
    if case == 'F':
        return f'Điểm chuẩn ngành CNTT mã 7480201 năm 2026 là 23 điểm [{markers[IT]}].'
    return FALLBACK_ANSWER


class MemoryDB:
    def __init__(self): self.records = []
    async def get(self, *args): return None
    def add_all(self, records): self.records.extend(records)
    async def flush(self): pass
    async def commit(self): pass
    async def rollback(self): raise AssertionError('unexpected persistence error')


@pytest.mark.asyncio
@pytest.mark.parametrize('accept', ['application/json', 'text/event-stream'])
@pytest.mark.parametrize('case', list(QUESTIONS))
async def test_requested_question_with_actual_pinned_candidates(records, monkeypatch, case, accept):
    selected = {'A': METHODS, 'B': [SAT], 'C': [STANDARD, QUALITY, INTERNATIONAL],
                'D': [STANDARD, QUALITY], 'E': [SAT], 'F': [IT], 'G': [IT], 'H': [IT]}[case]
    retriever, llm = Mock(), Mock()
    retriever.search.return_value = [(records[source], .9 - index * .01) for index, source in enumerate(selected)]
    chain = RAGChain(retriever, llm, top_k=5, min_lexical_coverage=0, reranker=LiteEvidenceSelector())
    question = QUESTIONS[case]
    prepared = chain._prepare_generation(question, None)
    markers = {source.prompt_chunk.chunk_id: index for index, source in enumerate(prepared.sources, 1)} if prepared else {}
    internal_answer = answer_for(case, markers)
    llm.generate.return_value = internal_answer
    monkeypatch.setattr(chat, 'rag_chain', chain)
    monkeypatch.setattr(chat, 'session_memory', SessionMemory())
    db = MemoryDB()
    async def database(): yield db
    monkeypatch.setattr(app, 'dependency_overrides', {get_db: database})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/chat', json={'message': question}, headers={'Accept': accept})
    assert response.status_code == 200, response.text
    if accept == 'application/json':
        result = response.json()
    else:
        assert 'event: error' not in response.text
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        result = {**next(event for event in events if event['type'] == 'metadata'),
                  'answer': ''.join(event.get('text', '') for event in events)}
    assert result['route_type'] == ('out_of_scope' if case in ('G', 'H') else 'general')
    assert result['answer'] == db.records[-1].content
    assert not __import__('re').search(r'\[\d+\]', result['answer'])
    assert {c['chunk_id'] for c in result['citations']} == (set() if case in ('G', 'H') else set(selected))
    for citation in result['citations']:
        original = records[citation['chunk_id']]
        assert citation['source_url'] == original.metadata['source_url']
        assert citation['title'] == original.metadata['title']
    assert llm.generate.call_count == (0 if prepared is None else 1)
    llm.generate_stream.assert_not_called()


@pytest.mark.parametrize('case', ['D', 'E'])
def test_leading_questions_cannot_approve_inverted_figures(records, case):
    from src.generation.grounding import RAGGroundingError, validate_answer
    sources = [records[STANDARD], records[QUALITY]] if case == 'D' else [records[SAT]]
    answer = ('Học phí đại trà năm học 2026-2027 là 58,3 triệu đồng/năm [1] [2].' if case == 'D'
              else 'Điểm SAT tối thiểu 25/36 và ACT tối thiểu 1130/1600 [1].')
    with pytest.raises(RAGGroundingError) as error:
        validate_answer(answer, sources)
    assert error.value.reason == 'unsupported_relationship'
