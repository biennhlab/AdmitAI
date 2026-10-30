"""Provider failure recovery copies complete source data without a model retry."""
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from api.main import app
from api.routers import chat
from src.database.connection import get_db
from src.generation.evidence_fallback import admission_methods
from src.generation.llm_client import (LLMUpstreamError, LLMTimeoutError, LLMConnectionError,
                                      LLMRateLimitError, LLMAuthenticationError)
from src.generation.rag_chain import RAGChain
from src.generation.session_memory import SessionMemory
from src.ingestion.chunker import Chunk
from src.local_profile import LOCAL_INDEX_DIR
from src.retrieval.lite_selector import LiteEvidenceSelector

QUESTION = 'PTIT có những phương thức tuyển sinh nào trong năm 2026?'
IDS = ['d54b54ae2497010969f6cc4e', 'f915244007250ca76bc6d333', '2362e40d70c923019a96f185',
       '92a4abe898903315cd1455c7', 'd023c570c53e0fbdb0f2870b']


@pytest.fixture
def actual_sources():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / LOCAL_INDEX_DIR / 'manifest.json').read_text(encoding='utf-8'))
    found = {}
    def walk(value):
        if isinstance(value, dict):
            if value.get('chunk_id') in IDS:
                found[value['chunk_id']] = Chunk(**{key: value[key] for key in ('chunk_id', 'content', 'metadata')})
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(manifest)
    return [found[key] for key in IDS]


def make_chain(sources, error=LLMUpstreamError):
    retriever, llm = Mock(), Mock()
    retriever.search.return_value = [(source, .9-index*.01) for index, source in enumerate(sources)]
    llm.generate.side_effect = error('PRIVATE_PROVIDER_PAYLOAD')
    return RAGChain(retriever, llm, top_k=5, min_lexical_coverage=0, reranker=LiteEvidenceSelector()), llm


def test_exact_failed_request_evidence_produces_complete_verified_methods(actual_sources):
    internal = admission_methods(QUESTION, actual_sources)
    assert internal is not None
    assert len(internal.splitlines()) == 5
    assert 'tài năng' in internal and 'SAT/ACT' in internal and 'tốt nghiệp THPT' in internal


class MemoryDB:
    def __init__(self): self.records, self.committed = [], False
    async def get(self, *args): return None
    def add_all(self, records): self.records.extend(records)
    async def flush(self): pass
    async def commit(self): self.committed = True
    async def rollback(self): raise AssertionError('unexpected failure')


@pytest.mark.asyncio
@pytest.mark.parametrize('accept', ['application/json', 'text/event-stream'])
@pytest.mark.parametrize('error', [LLMUpstreamError, LLMTimeoutError, LLMConnectionError, LLMRateLimitError])
async def test_provider_failure_is_recovered_only_with_verified_facts(actual_sources, monkeypatch, accept, error, caplog):
    chain, llm = make_chain(actual_sources, error)
    monkeypatch.setattr(chat, 'rag_chain', chain)
    monkeypatch.setattr(chat, 'session_memory', SessionMemory())
    db = MemoryDB()
    async def database(): yield db
    monkeypatch.setattr(app, 'dependency_overrides', {get_db: database})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/chat', json={'message': QUESTION}, headers={'Accept': accept})
    assert response.status_code == 200, response.text
    if accept == 'application/json':
        result = response.json()
    else:
        assert 'event: error' not in response.text
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        result = {**next(e for e in events if e['type'] == 'metadata'),
                  'answer': ''.join(e.get('text', '') for e in events)}
    assert result['answer_mode'] == 'evidence' and result['route_type'] == 'general'
    assert len(result['answer'].splitlines()) == 5
    assert not __import__('re').search(r'\[\d+\]', result['answer'])
    assert result['citations'] and {c['chunk_id'] for c in result['citations']} <= set(IDS)
    assert db.committed and db.records[-1].content == result['answer']
    assert 'PRIVATE_PROVIDER_PAYLOAD' not in caplog.text + response.text
    llm.generate.assert_called_once()
    llm.generate_stream.assert_not_called()


@pytest.mark.parametrize('case', ['partial', 'wrong_year', 'conflicting_total', 'conflicting_method', 'missing_year', 'special_program', 'wrong_campus', 'conflicting_campuses', 'single_campus_global', 'filtered_question', 'other_institution', 'other_publisher'])
def test_ambiguous_or_incomplete_sources_decline_recovery(actual_sources, case):
    sources = actual_sources
    question = QUESTION
    if case == 'partial': sources = sources[:1]
    if case == 'wrong_year': sources[0].metadata = {**sources[0].metadata, 'admission_year': '2025'}
    if case == 'missing_year': sources[0].metadata = {}
    if case == 'conflicting_total': sources[1].content += '\nCó 6 phương thức tuyển sinh.'
    if case == 'conflicting_method': sources[1].content += '\nPhương thức 1: Xét tuyển học bạ.'
    if case == 'special_program': question = 'Ngành Thú y có những phương thức tuyển sinh nào năm 2026?'
    if case == 'wrong_campus': question = 'Cơ sở Hà Nội có những phương thức tuyển sinh nào năm 2026?'
    if case == 'filtered_question': question = 'PTIT có những phương thức tuyển sinh nào không cần IELTS năm 2026?'
    if case == 'other_institution': question = 'Đại học Bách khoa có những phương thức tuyển sinh nào năm 2026?'
    if case == 'other_publisher': sources[0].metadata = {**sources[0].metadata, 'source_url': 'https://example.test/admissions'}
    if case in ('conflicting_campuses', 'single_campus_global'):
        sources[0].metadata = {**sources[0].metadata, 'campus': 'BVH'}
        if case == 'conflicting_campuses':
            sources[1].metadata = {**sources[1].metadata, 'campus': 'BVS'}
    assert admission_methods(question, sources) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('accept', ['application/json', 'text/event-stream'])
@pytest.mark.parametrize('error', [LLMUpstreamError, LLMAuthenticationError])
async def test_unrecoverable_provider_failures_are_explicit_503_not_fake_answers(monkeypatch, accept, error):
    source = Chunk('sat', 'Điểm SAT tối thiểu 1130/1600.', {'admission_year': '2026'})
    chain, llm = make_chain([source], error)
    monkeypatch.setattr(chat, 'rag_chain', chain)
    monkeypatch.setattr(chat, 'session_memory', SessionMemory())
    db = MemoryDB()
    async def database(): yield db
    monkeypatch.setattr(app, 'dependency_overrides', {get_db: database})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/chat', json={'message': 'Điểm SAT tối thiểu?'}, headers={'Accept': accept})
    assert response.status_code == 503
    assert response.json()['detail']['code'] == error.code
    assert 'answer' not in response.json() and not db.committed and db.records == []
    llm.generate.assert_called_once()


def test_authentication_failure_is_not_hidden_as_evidence_success(actual_sources):
    chain, _ = make_chain(actual_sources, LLMAuthenticationError)
    with pytest.raises(LLMAuthenticationError): chain.answer(QUESTION)


MISSING_THIRD = '\n'.join([
    '- Xét tuyển tài năng [1].',
    '- Xét tuyển dựa vào kết quả Chứng chỉ đánh giá năng lực quốc tế (Chứng chỉ SAT/ACT) [2].',
    '- Xét tuyển kết hợp giữa chứng chỉ tiếng Anh quốc tế (chứng chỉ IELTS, TOEFL) với kết quả học tập ở bậc THPT.',
    '- Xét tuyển dựa vào kết quả trong các kỳ thi đánh giá năng lực, đánh giá tư duy [4].',
    '- Xét tuyển dựa vào kết quả thi tốt nghiệp THPT năm 2026 [5].',
])


@pytest.mark.asyncio
@pytest.mark.parametrize('accept', ['application/json', 'text/event-stream'])
@pytest.mark.parametrize('raw', [MISSING_THIRD, MISSING_THIRD.replace(
    'với kết quả học tập ở bậc THPT.', 'với học phí 9999 triệu/năm.')])
async def test_model_output_with_missing_citations_is_returned_without_replacement(actual_sources, monkeypatch, caplog, accept, raw):
    chain, llm = make_chain(actual_sources)
    llm.generate.side_effect = None
    llm.generate.return_value = raw
    monkeypatch.setattr(chat, 'rag_chain', chain)
    monkeypatch.setattr(chat, 'session_memory', SessionMemory())
    db = MemoryDB()
    async def database(): yield db
    monkeypatch.setattr(app, 'dependency_overrides', {get_db: database})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/chat', json={'message': QUESTION}, headers={'Accept': accept})
    assert response.status_code == 200, response.text
    if accept == 'application/json': result = response.json()
    else:
        assert 'event: error' not in response.text
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        result = {**next(e for e in events if e['type'] == 'metadata'),
                  'answer': ''.join(e.get('text', '') for e in events)}
    assert result['answer_mode'] == 'generated' and result['route_type'] == 'general'
    assert len(result['answer'].splitlines()) == 5
    assert result['answer'] == RAGChain._public_answer(raw)
    assert db.committed and db.records[-1].content == result['answer']
    assert result['citations'] and {c['chunk_id'] for c in result['citations']} <= set(IDS)
    llm.generate.assert_called_once()
    llm.generate_stream.assert_not_called()
