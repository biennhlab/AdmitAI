"""Named-program answers preserve every available admissions row and qualifier."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from api.main import app
from api.routers import chat
from src.config import settings
from src.database.connection import get_db
from src.generation.evidence_fallback import program_profile
from src.generation.rag_chain import FALLBACK_ANSWER, RAGChain
from src.generation.session_memory import SessionMemory
from src.ingestion.chunker import Chunk
from src.retrieval.fusion import HybridRetriever
from src.retrieval.lite_selector import LiteEvidenceSelector
from src.retrieval.program_catalog import named_program_catalog, named_program_query


QUESTION = 'ngành khoa học máy tính'
HEADER = '| TT | Mã xét tuyển | Tên ngành/chương trình xét tuyển | Mã ngành | Tên ngành theo Danh mục cấp IV | Số lượng dự kiến | Phương thức tuyển sinh |\n| --- | --- | --- | --- | --- | --- | --- |\n'


def family(campus='BVH', year=2026, doc='catalog'):
    if campus == 'BVH':
        rows = [
            '| I | CHƯƠNG TRÌNH TÀI NĂNG | CHƯƠNG TRÌNH TÀI NĂNG | CHƯƠNG TRÌNH TÀI NĂNG | CHƯƠNG TRÌNH TÀI NĂNG | CHƯƠNG TRÌNH TÀI NĂNG | CHƯƠNG TRÌNH TÀI NĂNG |',
            '| 1 | 8480101TN | Khoa học máy tính | 8480101 | Khoa học máy tính | 30 | Xét tuyển thẳng và Phương thức tuyển sinh 5 |',
            '| II | ĐẠI TRÀ | ĐẠI TRÀ | ĐẠI TRÀ | ĐẠI TRÀ | ĐẠI TRÀ | ĐẠI TRÀ |',
            '| 1 | 7480101 | Khoa học dữ liệu | 7480101 | Khoa học máy tính | 165 | Phương thức tuyển sinh: 1, 2, 3, 4, 5 Tổ hợp xét tuyển: A00, A01, X06, X26 |',
        ]
    else:
        rows = ['| 1 | 7480201 | Công nghệ thông tin | 7480201 | Công nghệ thông tin | 200 | Phương thức tuyển sinh 1 |']
    return [Chunk(f'{doc}-{campus}-{index}', HEADER + row, {
        'doc_id': doc, 'title': f'Tuyển sinh năm {year}',
        'source_url': f'https://tuyensinh.ptit.edu.vn/{doc}', 'admission_year': str(year),
        'campus': campus, 'chunk_type': 'table', 'table_index': 0 if campus == 'BVH' else 1,
        'table_part': index, 'table_part_count': len(rows), 'row_start': index, 'row_end': index,
    }) for index, row in enumerate(rows, 1)]


def chain_for(chunks):
    dense = SimpleNamespace(search=Mock(side_effect=AssertionError('No dense/model search')))
    sparse = SimpleNamespace(chunks=chunks, search=Mock(side_effect=AssertionError('No sparse search')))
    llm = Mock()
    llm.generate.side_effect = AssertionError('No provider generation')
    return RAGChain(HybridRetriever(dense, sparse), llm, top_k=1, reranker=LiteEvidenceSelector()), llm


@pytest.mark.parametrize('question', [QUESTION, 'Thông tin về ngành Khoa học máy tính PTIT',
    'Thông tin chi tiết về ngành Khoa học máy tính ở PTIT', 'nganh khoa hoc may tinh',
    'PTIT có ngành Khoa học máy tính không?', 'PTIT có đào tạo ngành Khoa học máy tính không?',
    'Cho mình thông tin về ngành khoa học máy tính năm 2026 tại cơ sở phía Bắc'])
def test_named_program_query_uses_source_names_and_all_available_fields(question):
    chain, llm = chain_for(family() + family('BVS'))
    answer = chain.answer(question)
    assert answer.answer_mode == 'evidence'
    assert '8480101TN' in answer.answer and '7480101' in answer.answer
    assert 'Khoa học dữ liệu' in answer.answer and 'Khoa học máy tính' in answer.answer
    assert '| 30 |' in answer.answer and '| 165 |' in answer.answer
    assert 'Tổ hợp xét tuyển: A00, A01, X06, X26' in answer.answer
    assert 'TÀI NĂNG' in answer.answer and 'ĐẠI TRÀ' in answer.answer
    assert 'Hà Nội' in answer.answer and '2026' in answer.answer
    assert '7480201' not in answer.answer
    llm.generate.assert_not_called()


@pytest.mark.parametrize('question', ['Học phí ngành Khoa học máy tính PTIT',
    'Ngành khoa học máy tính điểm chuẩn 2026', 'Ngành Khoa học máy tính học gì?',
    'Ngành Khoa học máy tính cơ hội việc làm', 'So sánh ngành Khoa học máy tính và Marketing'])
def test_specific_intents_keep_their_generation_path(question):
    assert named_program_query(question) is None


def test_display_name_query_does_not_expand_to_other_official_major_variants():
    chain, _ = chain_for(family() + family('BVS'))
    answer = chain.answer('ngành Khoa học dữ liệu')
    assert '7480101' in answer.answer and '| 165 |' in answer.answer
    assert '8480101TN' not in answer.answer and '| 30 |' not in answer.answer


@pytest.mark.parametrize('question', ['ngành Thú y', 'ngành Khoa học máy tính tại BVS',
                                      'ngành Khoa học máy tính năm 2027'])
def test_unproven_program_campus_or_year_abstains_without_negative_claim(question):
    chain, llm = chain_for(family() + family('BVS'))
    result = chain.answer(question)
    assert result.answer == FALLBACK_ANSWER and result.route_type == 'out_of_scope' and not result.citations
    llm.generate.assert_not_called()


def test_missing_table_part_cannot_produce_a_partial_named_profile():
    north = family()
    north.pop()
    chain, llm = chain_for(north + family('BVS'))
    assert chain.answer(QUESTION).route_type == 'out_of_scope'
    llm.generate.assert_not_called()


def test_conflicting_quota_in_equally_complete_sources_fails_closed():
    alternate = family(doc='other')
    alternate[-1].content = alternate[-1].content.replace('| 165 |', '| 999 |')
    assert named_program_catalog(QUESTION, [family(), family('BVS'), alternate, family('BVS', doc='other')]) is None


def test_source_records_are_never_mutated():
    chunks = family() + family('BVS')
    before = copy.deepcopy([c.__dict__ for c in chunks])
    chain, _ = chain_for(chunks)
    chain.answer(QUESTION)
    assert [c.__dict__ for c in chunks] == before


@pytest.fixture(scope='module')
def pinned():
    manifest = json.loads((Path(settings.INDEX_DIR) / 'manifest.json').read_text(encoding='utf-8'))
    return [Chunk(**row) for row in manifest['chunks']]


@pytest.mark.parametrize('question', [QUESTION, 'ngành Công nghệ thông tin', 'ngành Marketing',
                                      'ngành Khoa học dữ liệu', 'ngành Quản trị kinh doanh',
                                      'ngành CNTT', 'ngành ATTT'])
def test_other_real_program_names_are_handled_without_hardcoded_answers(pinned, question):
    chain, llm = chain_for(pinned)
    result = chain.answer(question)
    assert result.answer_mode == 'evidence' and result.route_type == 'general' and result.citations
    llm.generate.assert_not_called()


def test_pinned_computer_science_entries_match_original_source_rows(pinned):
    chain, _ = chain_for(pinned)
    result = chain.answer(QUESTION)
    source = Path('data/processed/ptit/admissions/ptit-2026-admissions-thong-tin-tuyen-sinh-dai-hoc-chinh-quy-nam-2026-v1.md')
    expected = []
    for line in source.read_text(encoding='utf-8').splitlines():
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) == 7 and cells[0].isdigit() and cells[1] in {'8480101TN', '7480101'}:
            expected.append(cells[1:])
    observed = [[c.strip() for c in line.strip('|').split('|')]
                for line in result.answer.splitlines() if line.startswith(('| 8480101TN |', '| 7480101 |'))]
    assert observed == expected and len(observed) == 2


class MemoryDB:
    async def get(self, *args): return None
    def add_all(self, records): pass
    async def flush(self): pass
    async def commit(self): pass
    async def rollback(self): raise AssertionError('Unexpected database failure')


@pytest.mark.asyncio
@pytest.mark.parametrize('accept', ['application/json', 'text/event-stream'])
async def test_named_program_json_sse_consistency(pinned, monkeypatch, accept):
    chain, llm = chain_for(pinned)
    monkeypatch.setattr(chat, 'rag_chain', chain)
    monkeypatch.setattr(chat, 'session_memory', SessionMemory())
    async def db(): yield MemoryDB()
    monkeypatch.setattr(app, 'dependency_overrides', {get_db: db})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/chat', json={'message': QUESTION}, headers={'Accept': accept})
    assert response.status_code == 200, response.text
    if accept == 'application/json':
        result = response.json()
    else:
        assert 'event: error' not in response.text
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        result = {**next(e for e in events if e.get('type') == 'metadata'),
                  'answer': ''.join(e.get('text', '') for e in events)}
    assert result['answer_mode'] == 'evidence' and result['session_id']
    assert '8480101TN' in result['answer'] and '7480101' in result['answer']
    assert '| 30 |' in result['answer'] and '| 165 |' in result['answer']
    assert result['citations'] and all(c['score'] is None for c in result['citations'])
    llm.generate.assert_not_called()
