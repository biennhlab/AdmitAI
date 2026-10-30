"""Citation presentation maps references to real source metadata."""
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


def chain_for(answer, lite=True):
    sources = [
        Chunk('sat', 'Điểm SAT tối thiểu 1130/1600.',
              {'title': 'Ngưỡng SAT', 'source_url': 'https://example.test/sat', 'doc_id': 'sat'}),
        Chunk('act', 'Điểm ACT tối thiểu 25/36.',
              {'title': 'Ngưỡng ACT', 'source_url': 'https://example.test/act', 'doc_id': 'act'}),
    ]
    retriever, llm = Mock(), Mock()
    retriever.search.return_value = [(source, .9 - index * .1) for index, source in enumerate(sources)]
    llm.generate.return_value = answer
    llm.generate_stream.return_value = iter([answer])
    return RAGChain(retriever, llm, min_lexical_coverage=0,
                    reranker=LiteEvidenceSelector() if lite else None), llm


@pytest.mark.parametrize('lite', [True, False])
@pytest.mark.parametrize('answer,visible,used', [
    ('Điểm SAT tối thiểu 1130/1600 [1].', 'Điểm SAT tối thiểu 1130/1600.', ['sat']),
    ('Điểm SAT tối thiểu 1130/1600 [1]. Điểm ACT tối thiểu 25/36 [2].',
     'Điểm SAT tối thiểu 1130/1600. Điểm ACT tối thiểu 25/36.', ['sat', 'act']),
    ('Điểm SAT tối thiểu 1130/1600 [1].\n\nĐiểm ACT tối thiểu 25/36 [2].',
     'Điểm SAT tối thiểu 1130/1600.\n\nĐiểm ACT tối thiểu 25/36.', ['sat', 'act']),
    ('- Điểm SAT tối thiểu 1130/1600 [1].\n- Điểm ACT tối thiểu 25/36 [2].',
     '- Điểm SAT tối thiểu 1130/1600.\n- Điểm ACT tối thiểu 25/36.', ['sat', 'act']),
    ('| Kỳ thi | Ngưỡng điểm |\n| --- | --- |\n| SAT | 1130/1600 [1] |\n| ACT | 25/36 [2] |',
     '| Kỳ thi | Ngưỡng điểm |\n| --- | --- |\n| SAT | 1130/1600 |\n| ACT | 25/36 |', ['sat', 'act']),
    ('Điểm SAT tối thiểu 1130/1600 [1][1].', 'Điểm SAT tối thiểu 1130/1600.', ['sat']),
    ('Điểm ACT tối thiểu 25/36 [2].', 'Điểm ACT tối thiểu 25/36.', ['act']),
    ('Điểm SAT tối thiểu 1130/1600 [1].  \nĐiểm ACT tối thiểu 25/36 [2].',
     'Điểm SAT tối thiểu 1130/1600.  \nĐiểm ACT tối thiểu 25/36.', ['sat', 'act']),
    ('Điểm SAT tối thiểu 1130/1600 và ACT tối thiểu 25/36 [1] [2].',
     'Điểm SAT tối thiểu 1130/1600 và ACT tối thiểu 25/36.', ['sat', 'act']),
])
def test_markers_are_presented_as_source_metadata(lite, answer, visible, used):
    chain, llm = chain_for(answer, lite)
    result = chain.answer('Điểm SAT ACT tối thiểu?')
    assert result.answer == visible
    assert [c['chunk_id'] for c in result.citations] == used
    assert all(c['source_url'] == f'https://example.test/{c["chunk_id"]}' for c in result.citations)
    llm.generate.assert_called_once()


@pytest.mark.parametrize('lite', [True, False])
def test_duplicate_retrieved_evidence_has_one_citation(lite):
    chain, _ = chain_for('Điểm SAT tối thiểu 1130/1600 [1][1].', lite)
    sat = chain.retriever.search.return_value[0]
    chain.retriever.search.return_value = [sat, sat, sat]
    result = chain.answer('Điểm SAT tối thiểu?')
    assert result.answer == 'Điểm SAT tối thiểu 1130/1600.'
    assert len(result.citations) == 1
    assert result.citations[0]['chunk_id'] == 'sat'


def test_sources_in_headings_and_body_are_both_presented_as_references():
    chain, _ = chain_for('## Ngưỡng điểm [2]\n\nĐiểm SAT tối thiểu 1130/1600 [1].')
    result = chain.answer('Điểm SAT ACT tối thiểu?')
    assert result.answer == '## Ngưỡng điểm\n\nĐiểm SAT tối thiểu 1130/1600.'
    assert [c['chunk_id'] for c in result.citations] == ['sat', 'act']


@pytest.mark.asyncio
@pytest.mark.parametrize('accept', ['application/json', 'text/event-stream'])
@pytest.mark.parametrize('kind', ['valid', 'missing_citation', 'invalid_marker', 'wrong_mapping', 'swapped', 'abstention', 'provider'])
async def test_private_diagnostics_correlate_transport_with_actual_evidence(monkeypatch, caplog, accept, kind):
    import hashlib
    from src.generation.llm_client import LLMUpstreamError

    answers = {
        'valid': 'Điểm SAT tối thiểu 1130/1600 [1].',
        'missing_citation': 'Điểm SAT tối thiểu 1130/1600.',
        'invalid_marker': 'Điểm SAT tối thiểu 1130/1600 [99].',
        'wrong_mapping': 'Điểm SAT tối thiểu 1130/1600 [2].',
        'swapped': 'Điểm SAT tối thiểu 25/36 và ACT tối thiểu 1130/1600 [1] [2].',
        'abstention': FALLBACK_ANSWER,
        'provider': '',
    }
    answer = answers[kind]
    chain, llm = chain_for(answer)
    if kind == 'provider':
        llm.generate.side_effect = LLMUpstreamError('private provider response secret')
    monkeypatch.setattr(chat, 'rag_chain', chain)
    monkeypatch.setattr(chat, 'session_memory', SessionMemory())
    db = MemoryDB()
    async def database(): yield db
    monkeypatch.setattr(app, 'dependency_overrides', {get_db: database})
    question = 'Điểm SAT ACT tối thiểu?'
    with caplog.at_level('INFO'):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post('/api/chat', json={'message': question}, headers={'Accept': accept})
    records = [json.loads(record.message.removeprefix('rag_generation ')) for record in caplog.records
               if record.message.startswith('rag_generation ')]
    assert len(records) == 1
    record = records[0]
    assert record['request_id'] == response.headers['x-request-id']
    assert record['question_hash'] == hashlib.sha256(question.encode()).hexdigest()
    assert record['selected_source_ids'] == ['sat', 'act']
    assert record['citation_mapping'] == {'1': 'sat', '2': 'act'}
    assert record['reason'] == ('LLM_PROVIDER_ERROR' if kind == 'provider' else None)
    assert record['outcome'] == ('rejected' if kind == 'provider' else 'accepted')
    assert 'claim_index' not in record
    assert question not in caplog.text
    if answer:
        assert answer not in caplog.text
    assert 'private provider response secret' not in caplog.text
    assert response.status_code == (503 if kind == 'provider' else 200)


class MemoryDB:
    def __init__(self):
        self.records = []
        self.committed = False

    async def get(self, *args): return None
    def add_all(self, records): self.records.extend(records)
    async def flush(self): pass
    async def commit(self): self.committed = True
    async def rollback(self): raise AssertionError('unexpected database failure')
