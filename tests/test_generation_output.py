"""Answers reach both transports without a factual or citation rejection gate."""
import json
from unittest.mock import Mock

import httpx
import pytest

from api.main import app
from api.routers import chat
from src.database.connection import get_db
from src.generation.rag_chain import FALLBACK_ANSWER, RAGChain
from src.generation.session_memory import SessionMemory
from src.ingestion.chunker import Chunk
from src.retrieval.lite_selector import LiteEvidenceSelector


class MemoryDB:
    def __init__(self): self.records = []; self.committed = False
    async def get(self, *_): return None
    def add_all(self, records): self.records.extend(records)
    async def flush(self): pass
    async def commit(self): self.committed = True
    async def rollback(self): raise AssertionError('No database failure is simulated')


@pytest.mark.asyncio
@pytest.mark.parametrize('accept', ['application/json', 'text/event-stream'])
@pytest.mark.parametrize('lite', [False, True])
@pytest.mark.parametrize('structured', [False, True])
@pytest.mark.parametrize('text', [
    'Điểm SAT tối đa 9999.',
    'Thí sinh không được xét tuyển bằng học bạ.',
    'Thí sinh bắt buộc nộp chứng chỉ tiếng Anh.',
    'Học phí đại trà 35,9 triệu/tháng.',
    'Câu trả lời không có trích dẫn.',
])
async def test_generated_text_is_returned_and_persisted_without_content_checks(
    monkeypatch, accept, lite, structured, text,
):
    retriever, provider = Mock(), Mock()
    source = Chunk('source', 'Điểm SAT tối thiểu 1130. Thí sinh được xét tuyển bằng học bạ. '
                   'Thí sinh có thể nộp chứng chỉ tiếng Anh. Học phí đại trà 35,9 triệu/năm.',
                   {'doc_id': 'source', 'title': 'Thông tin tuyển sinh', 'source_url': 'https://example.test/source'})
    retriever.search.return_value = [(source, .9)]
    provider.generate.return_value = (json.dumps({'status': 'answer', 'blocks': [
        {'kind': 'paragraph', 'text': text}]}, ensure_ascii=False) if structured else text)
    monkeypatch.setattr(chat, 'rag_chain', RAGChain(retriever, provider, min_lexical_coverage=0,
        reranker=LiteEvidenceSelector() if lite else None))
    monkeypatch.setattr(chat, 'session_memory', SessionMemory())
    db = MemoryDB()
    async def database(): yield db
    monkeypatch.setattr(app, 'dependency_overrides', {**app.dependency_overrides, get_db: database})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/chat', json={'message': 'Thông tin tuyển sinh?'}, headers={'Accept': accept})
    assert response.status_code == 200, response.text
    if accept == 'application/json': result = response.json()
    else:
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        result = {**next(e for e in events if e['type'] == 'metadata'),
                  'answer': ''.join(e.get('text', '') for e in events)}
    assert result['answer'] == text and result['citations'] == []
    assert db.committed and db.records[-1].content == text
    assert chat.session_memory.get_history(result['session_id'])[-1]['content'] == text
    provider.generate.assert_called_once()
    provider.generate_stream.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('accept', ['application/json', 'text/event-stream'])
@pytest.mark.parametrize('output', [None, '', ' \n\t', '{"status":"answer","blocks":['])
async def test_only_unrenderable_provider_output_returns_a_format_error(monkeypatch, accept, output):
    retriever, provider = Mock(), Mock()
    retriever.search.return_value = [(Chunk('sat', 'Điểm SAT tối thiểu 1130.', {}), .9)]
    provider.generate.return_value = output
    monkeypatch.setattr(chat, 'rag_chain', RAGChain(retriever, provider, min_lexical_coverage=0))
    monkeypatch.setattr(chat, 'session_memory', SessionMemory())
    db = MemoryDB()
    async def database(): yield db
    monkeypatch.setattr(app, 'dependency_overrides', {**app.dependency_overrides, get_db: database})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/chat', json={'message': 'Điểm SAT?'}, headers={'Accept': accept})
    assert response.status_code == 502
    assert response.json()['detail']['code'] == 'INVALID_GENERATION'
    assert not db.committed and not db.records


def test_invalid_marker_is_ignored_and_available_citations_keep_real_metadata():
    retriever, provider = Mock(), Mock()
    retriever.search.return_value = [(Chunk('sat', 'Điểm SAT 1130.', {'title': 'SAT'}), .9)]
    provider.generate.return_value = 'Điểm SAT 9999 [99]. Điểm SAT 25 [1].'
    chain = RAGChain(retriever, provider, min_lexical_coverage=0)
    result = chain.answer('Điểm SAT?')
    assert result.answer == 'Điểm SAT 9999. Điểm SAT 25.'
    assert [(c['marker'], c['chunk_id']) for c in result.citations] == [(1, 'sat')]


def test_explicit_abstention_still_returns_normal_fallback():
    retriever, provider = Mock(), Mock()
    retriever.search.return_value = [(Chunk('sat', 'Điểm SAT 1130.', {}), .9)]
    provider.generate.return_value = '{"status":"abstain","blocks":[]}'
    result = RAGChain(retriever, provider, min_lexical_coverage=0).answer('Điểm SAT?')
    assert result.answer == FALLBACK_ANSWER and result.route_type == 'out_of_scope'
