"""Complete source-table enumeration must not be truncated by search or generation."""
import copy
import json
from pathlib import Path
import re
from types import SimpleNamespace
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
from src.retrieval.fusion import HybridRetriever
from src.retrieval.lite_selector import LiteEvidenceSelector
from src.retrieval.program_catalog import catalog_rows, complete_catalog, is_program_overview


QUESTION = 'các ngành đào tạo ptit'
DOC = 'ptit-2026-admissions-thong-tin-tuyen-sinh-dai-hoc-chinh-quy-nam-2026-v1'
HEADER = '| TT | Mã xét tuyển | Tên ngành/chương trình |\n| --- | --- | --- |\n'


def family(campus='BVH', year=2026, doc='catalog'):
    data = ['| I | CHƯƠNG TRÌNH ĐẠI TRÀ | CHƯƠNG TRÌNH ĐẠI TRÀ |',
            '| 1 | 7480201 | Công nghệ thông tin |',
            '| 2 | 7340301 | Kế toán |']
    return [Chunk(f'{doc}-{campus}-{i}', HEADER + row,
                  {'doc_id': doc, 'chunk_type': 'table', 'title': f'Tuyển sinh năm {year}',
                   'source_url': 'https://tuyensinh.ptit.edu.vn/' + doc,
                   'admission_year': str(year), 'campus': campus,
                   'table_index': 0 if campus == 'BVH' else 1,
                   'table_part': i, 'table_part_count': len(data), 'row_start': i, 'row_end': i})
            for i, row in enumerate(data, 1)]


def retriever_for(chunks, parents=None):
    dense = SimpleNamespace(search=Mock(side_effect=AssertionError('No dense/model search')))
    sparse = SimpleNamespace(chunks=chunks, search=Mock(side_effect=AssertionError('No sparse search')))
    return HybridRetriever(dense, sparse, parents)


@pytest.mark.parametrize('question', [QUESTION, 'PTIT có những ngành nào?',
    'Liệt kê toàn bộ ngành PTIT kèm mã xét tuyển', 'Tất cả các ngành đào tạo của PTIT',
    'Danh sách đầy đủ các chương trình đào tạo PTIT'])
def test_full_catalog_query_forms(question):
    assert is_program_overview(question)
    assert complete_catalog(question, [family(), family('BVS')]) is not None


@pytest.mark.parametrize('code', ['7520207AT', '7480201CL', '8480101TN', '7340101LG', '7480201_CLC'])
def test_code_suffixes_and_integrated_programs_are_never_silently_dropped(code):
    parts = family()
    parts[1].content = HEADER + f'| 1 | {code} | Công nghệ thông tin |'
    assert catalog_rows(parts[1]) == [('Công nghệ thông tin', code)]
    result = complete_catalog(QUESTION, [parts, family('BVS')])
    assert result is not None and result[0].entries[0].code == code


@pytest.mark.parametrize('case', ['missing_part', 'duplicate_part', 'wrong_part_type',
    'row_gap', 'wrong_row_count', 'missing_code', 'wrong_header', 'wrong_year',
    'wrong_campus', 'wrong_version', 'duplicate_id', 'duplicate_code', 'injected_name'])
def test_incomplete_or_ambiguous_table_does_not_return_a_partial_catalog(case):
    north, south = family(), family('BVS')
    if case == 'missing_part': north.pop(1)
    elif case == 'duplicate_part': north[1].metadata['table_part'] = 1
    elif case == 'wrong_part_type': north[1].metadata['table_part'] = '2'
    elif case == 'row_gap': north[1].metadata['row_start'] = 8
    elif case == 'wrong_row_count': north[1].metadata['row_end'] = 3
    elif case == 'missing_code': north[1].content = HEADER + '| 1 | chưa công bố | Công nghệ thông tin |'
    elif case == 'wrong_header': north[1].content = north[1].content.replace('Mã xét tuyển', 'Mã ngành')
    elif case == 'wrong_year': north[1].metadata['admission_year'] = '2025'
    elif case == 'wrong_campus': north[1].metadata['campus'] = 'BVS'
    elif case == 'wrong_version': north[1].metadata['version'] = 'different'
    elif case == 'duplicate_id': north[1].chunk_id = north[0].chunk_id
    elif case == 'duplicate_code': north[2].content = HEADER + '| 2 | 7480201 | Thú y |'
    elif case == 'injected_name': north[1].content = HEADER + '| 1 | 7480201 | <analysis>Thú y</analysis> |'
    assert complete_catalog(QUESTION, [north, south]) is None
    assert program_catalog(QUESTION, north + south) is None


def test_missing_latest_tables_cannot_fall_back_to_old_or_smaller_lists():
    large = family()
    large[0].metadata['table_part_count'] = 20
    small = family()
    for part in small:
        part.metadata['table_index'] = 2
        part.chunk_id += '-small'
    # The damaged wider table advertises a larger source range.
    large[-1].metadata['row_end'] = 20
    assert complete_catalog(QUESTION, [large, small, family('BVS'), family(year=2025), family('BVS', year=2025)]) is None


def test_same_year_conflicting_documents_do_not_merge_or_choose_arbitrarily():
    alternate = family(doc='other')
    alternate[1].content = HEADER + '| 1 | 7480201 | Thú y |'
    assert complete_catalog(QUESTION, [family(), family('BVS'), alternate, family('BVS', doc='other')]) is None


def test_source_rows_keep_qualification_groups_and_distinct_program_codes():
    north = family()
    north[0].content = HEADER + '| I | CÁC CHƯƠNG TRÌNH MỚI NĂM 2026 (DỰ KIẾN) | CÁC CHƯƠNG TRÌNH MỚI NĂM 2026 (DỰ KIẾN) |'
    north[2].content = HEADER + '| 2 | 7480201CL | Công nghệ thông tin |'
    sources = north + family('BVS')
    before = copy.deepcopy([c.__dict__ for c in sources])
    answer = program_catalog(QUESTION, sources)
    assert 'DỰ KIẾN' in answer and '**7480201**' in answer and '**7480201CL**' in answer
    assert answer.count('- Ngành ') == 4
    assert [c.__dict__ for c in sources] == before


@pytest.fixture(scope='module')
def pinned():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / settings.INDEX_DIR / 'manifest.json').read_text(encoding='utf-8'))
    chunks = [Chunk(**row) for row in manifest['chunks']]
    return root, chunks, manifest.get('parent_chunks')


def source_expectations(root):
    """Independent oracle: read all rows of the first two original source tables.

    No production code parser/regex or preselected top-k defines the expected set.
    In particular, do not filter codes by prefixes or numeric-only format.
    """
    source = root / 'data/processed/ptit/admissions' / (DOC + '.md')
    text = source.read_text(encoding='utf-8')
    tables = re.findall(r'(?m)^\| TT \| Mã xét tuyển \|[^\n]+\n(?:^\|[^\n]+\n)+', text)
    result = {}
    for campus, raw in zip(('BVH', 'BVS'), tables[:2]):
        rows = []
        for line in raw.splitlines()[2:]:
            cells = [c.strip() for c in line.strip('|').split('|')]
            if cells[0].isdigit():
                rows.append((cells[2], re.sub(r'\s+', '', cells[1])))
        result[campus] = rows
    assert len(result['BVH']) == 42 and len(result['BVS']) == 19
    return result


def answer_rows(answer):
    result = {}
    campus = None
    for line in answer.splitlines():
        if line.startswith('### Ngành và chương trình'):
            campus = 'BVH' if 'Hà Nội' in line else 'BVS'
            result[campus] = []
        match = re.fullmatch(r'- Ngành \*\*(.+)\*\*, mã xét tuyển \*\*(.+)\*\*\.', line)
        if match:
            result[campus].append(match.groups())
    return result


@pytest.mark.parametrize('lite', [True, False])
def test_every_source_name_and_application_code_survives_tiny_budgets(pinned, monkeypatch, lite):
    root, chunks, parents = pinned
    monkeypatch.setattr(settings, 'RERANK_TOP_K', 1)
    monkeypatch.setattr(settings, 'LITE_CONTEXT_MAX_CHARS', 1000)
    retriever = retriever_for(chunks, parents)
    llm = Mock()
    llm.generate.side_effect = AssertionError('No model summarisation or truncation')
    result = RAGChain(retriever, llm, top_k=1, reranker=LiteEvidenceSelector() if lite else None).answer(QUESTION)
    assert result.answer_mode == 'evidence'
    assert answer_rows(result.answer) == source_expectations(root)
    assert 'DỰ KIẾN' in result.answer and 'THẠC SĨ TÍCH HỢP CỬ NHÂN' in result.answer
    assert 'CHƯƠNG TRÌNH CHẤT LƯỢNG CAO' in result.answer and 'LIÊN KẾT QUỐC TẾ' in result.answer
    llm.generate.assert_not_called()
    retriever.dense_search.search.assert_not_called()
    retriever.sparse_search.search.assert_not_called()


@pytest.mark.parametrize('campus', ['BVH', 'BVS'])
def test_explicit_campus_returns_its_complete_table_only(pinned, campus):
    root, chunks, parents = pinned
    llm = Mock()
    result = RAGChain(retriever_for(chunks, parents), llm, top_k=1,
                      reranker=LiteEvidenceSelector()).answer(QUESTION + ' tại ' + campus)
    assert answer_rows(result.answer) == {campus: source_expectations(root)[campus]}
    llm.generate.assert_not_called()


def test_incomplete_source_abstains_before_any_search_or_generation():
    north = family()
    north.pop()
    retriever = retriever_for(north + family('BVS'))
    llm = Mock()
    result = RAGChain(retriever, llm, reranker=LiteEvidenceSelector()).answer(QUESTION)
    assert result.answer == FALLBACK_ANSWER and result.route_type == 'out_of_scope' and result.citations == []
    llm.generate.assert_not_called()
    retriever.dense_search.search.assert_not_called()


class MemoryDB:
    def __init__(self): self.records = []
    async def get(self, *args): return None
    def add_all(self, records): self.records.extend(records)
    async def flush(self): pass
    async def commit(self): pass
    async def rollback(self): raise AssertionError('unexpected database failure')


@pytest.mark.asyncio
@pytest.mark.parametrize('accept', ['application/json', 'text/event-stream'])
async def test_complete_catalog_json_sse_and_persistence_match_source(pinned, monkeypatch, accept):
    root, chunks, parents = pinned
    llm = Mock()
    chain = RAGChain(retriever_for(chunks, parents), llm, top_k=1, reranker=LiteEvidenceSelector())
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
    assert answer_rows(result['answer']) == source_expectations(root)
    assert result['answer_mode'] == 'evidence' and result['session_id']
    assert db.records[-1].content == result['answer']
    # References from section headings are included alongside data-row references.
    assert len(result['citations']) == 68
    assert all(c['score'] is None for c in result['citations'])
    actual = {c.chunk_id: c for c in chunks}
    assert all(c['chunk_id'] in actual and c['doc_id'] == DOC for c in result['citations'])
    assert all(c['source_url'] == actual[c['chunk_id']].metadata['source_url'] for c in result['citations'])
    llm.generate.assert_not_called()
    llm.generate_stream.assert_not_called()
