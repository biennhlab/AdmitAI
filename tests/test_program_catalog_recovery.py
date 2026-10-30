"""Catalog retrieval/citation regressions use local records and mocked generation."""
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from api.main import app
from api.routers import chat
from src.config import settings
from src.database.connection import get_db
from src.generation.evidence_fallback import program_catalog
from src.generation.rag_chain import FALLBACK_ANSWER, RAGChain
from src.generation.session_memory import SessionMemory
from src.ingestion.chunker import Chunk
from src.retrieval.domain_signals import source_campuses
from src.retrieval.fusion import HybridRetriever
from src.retrieval.lite_selector import LiteEvidenceSelector
from src.retrieval.program_catalog import catalog_rows, is_program_overview
from src.retrieval.sparse_search import BM25Search


QUESTION = 'các ngành đào tạo ptit'
NAV_IDS = ['7c9c1cf109fa81603d413814', 'af7be1dccdbb69fe9615b367',
           '3858b53c0b42d7a64e258cb7', '55a8aa745c82af03e7243cf9']


def table(name='Công nghệ thông tin', code='7480201', **metadata):
    metadata = {'admission_year': '2026', 'campus': 'BVH', 'doc_id': 'catalog',
                  'chunk_type': 'table', 'table_part': 1, 'table_part_count': 1,
                  'row_start': 1, 'row_end': 1,
                  'title': 'Tuyển sinh năm 2026', 'source_url': 'https://tuyensinh.ptit.edu.vn/catalog',
                  **metadata}
    metadata.setdefault('table_index', 0 if metadata['campus'] == 'BVH' else 1)
    return Chunk(code, '| TT | Tên ngành | Mã xét tuyển |\n| --- | --- | --- |\n'
                 f'| 1 | {name} | {code} |', metadata)


@pytest.mark.parametrize('question', [QUESTION, 'PTIT có những ngành nào?',
    'Liệt kê các ngành đào tạo của PTIT năm 2026 tại cơ sở phía Bắc',
    'Các ngành của Học viện Công nghệ Bưu chính Viễn thông'])
def test_overview_intent(question):
    assert is_program_overview(question)


@pytest.mark.parametrize('question', ['Các ngành đào tạo HUST',
    'PTIT có ngành Thú y không?', 'Học phí các ngành đào tạo PTIT',
    'Các ngành PTIT có điểm chuẩn bao nhiêu?', 'Các ngành PTIT không đào tạo',
    'Các ngành PTIT có học bổng không?', 'Các ngành PTIT năm 2025 và 2026',
    'Các ngành PTIT mã 7480201', 'Phương thức tuyển sinh PTIT là gì?',
    'Các ngành PTIT và HUST', 'Các ngành PTIT bằng tiếng Anh'])
def test_other_questions_are_unchanged(question):
    assert not is_program_overview(question)
    chain = RAGChain(Mock(), Mock(), reranker=LiteEvidenceSelector())
    assert 'tên ngành mã ngành' not in chain._retrieval_query(question)
    assert program_catalog(question, [table()]) is None


def test_extraction_is_from_paired_cells_only():
    good = table()
    assert catalog_rows(good) == [('Công nghệ thông tin', '7480201')]
    good.content = '| Mã ngành | Tên ngành |\n| --- | --- |\n| 7480201 | Công nghệ thông tin |'
    assert catalog_rows(good) == [('Công nghệ thông tin', '7480201')]
    good.content = '[Các ngành đào tạo](https://example.com/)\n7480201'
    assert catalog_rows(good) == []


@pytest.mark.parametrize('case', ['missing_year', 'unknown_campus', 'wrong_year',
    'wrong_campus', 'publisher', 'conflicting_name', 'mixed_years', 'navigation', 'no_rows', 'controls'])
def test_ambiguous_catalog_declines_recovery(case):
    sources = [table(), table('Kế toán', '7340301', campus='BVS')]
    question = QUESTION
    if case == 'missing_year': sources[0].metadata = {'campus': 'BVH'}
    elif case == 'unknown_campus': sources[0].metadata.pop('campus')
    elif case == 'wrong_year': question += ' năm 2025'
    elif case == 'wrong_campus': question += ' tại cơ sở phía Nam'
    elif case == 'publisher': sources[0].metadata['source_url'] = 'https://tuyensinh.ptit.edu.vn.evil.test/'
    elif case == 'conflicting_name': sources.append(table('Thú y'))
    elif case == 'mixed_years': sources.append(table(admission_year='2025'))
    elif case == 'navigation': sources[0].content = '[Danh sách ngành](https://tuyensinh.ptit.edu.vn/)'
    elif case == 'no_rows': sources = []
    elif case == 'controls': sources[0].content = sources[0].content.replace('Công nghệ thông tin', '<analysis>Thú y</analysis>')
    assert program_catalog(question, sources) is None


def test_seeds_prioritise_rows_year_and_campus_without_mutation():
    nav = Chunk('nav', 'Các ngành đào tạo PTIT: xem danh mục', {})
    old = table('Kế toán', '7340301', admission_year='2025')
    north = table()
    repeat = table()
    south = table('Kế toán', '7340301', campus='BVS')
    candidates = [(c, .9-i*.01) for i, c in enumerate([nav, old, north, repeat, south])]
    before = json.dumps([c.metadata for c, _ in candidates], sort_keys=True)
    selector = LiteEvidenceSelector()
    ranked = selector.rerank(QUESTION, candidates, top_k=5)
    seeds = selector.select_seeds(QUESTION, ranked, 5)
    assert {id(c) for c, _, _ in seeds} == {id(north), id(south)}
    assert all(any(c is original and score == s for original, s in candidates) for c, score, _ in seeds)
    assert json.dumps([c.metadata for c, _ in candidates], sort_keys=True) == before


@pytest.fixture(scope='module')
def corpus():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / settings.INDEX_DIR / 'manifest.json').read_text(encoding='utf-8'))
    records = {}
    def walk(value):
        if isinstance(value, dict):
            if 'chunk_id' in value and 'content' in value:
                records[value['chunk_id']] = Chunk(**{k: value[k] for k in ('chunk_id', 'content', 'metadata')})
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(manifest)
    sparse = BM25Search()
    sparse.index([Chunk(**r) for r in manifest['chunks']])
    class EmptyDense:
        def search(self, query, top_k): return []
    return records, HybridRetriever(EmptyDense(), sparse, manifest.get('parent_chunks'))


class MemoryDB:
    def __init__(self): self.records = []
    async def get(self, *args): return None
    def add_all(self, records): self.records.extend(records)
    async def flush(self): pass
    async def commit(self): pass
    async def rollback(self): raise AssertionError('unexpected rollback')


@pytest.mark.asyncio
@pytest.mark.parametrize('accept', ['application/json', 'text/event-stream'])
@pytest.mark.parametrize('failure', ['missing_citation', 'provider', 'supported'])
async def test_actual_bm25_question_transport_recovery(corpus, monkeypatch, accept, failure):
    from src.generation.llm_client import LLMUpstreamError

    _, retriever = corpus
    llm = Mock()
    chain = RAGChain(retriever, llm, top_k=5, reranker=LiteEvidenceSelector())
    prepared = chain._prepare_generation(QUESTION, None)
    assert prepared is not None
    sources = [s.prompt_chunk for s in prepared.sources]
    assert len(sources) > 5
    assert set().union(*(source_campuses(c) for c in sources)) == {'BVH', 'BVS'}
    assert not set(NAV_IDS) & {c.chunk_id for c in sources}
    internal = program_catalog(QUESTION, sources)
    assert internal is not None
    if failure == 'provider': llm.generate.side_effect = LLMUpstreamError('mock provider failure')
    elif failure == 'missing_citation':
        llm.generate.return_value = 'PTIT có 9999 ngành đào tạo.\n\n' + internal
    else: llm.generate.return_value = internal
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
    assert result['answer_mode'] == 'evidence'
    assert result['route_type'] == 'general' and result['session_id']
    assert '9999' not in result['answer'] and '[1]' not in result['answer']
    assert result['answer'].startswith('### Ngành và chương trình')
    assert result['answer'].count('- Ngành ') == 61
    assert result['citations'] and {c['chunk_id'] for c in result['citations']} <= {c.chunk_id for c in sources}
    assert db.records[-1].content == result['answer']
    llm.generate.assert_not_called()
    llm.generate_stream.assert_not_called()
