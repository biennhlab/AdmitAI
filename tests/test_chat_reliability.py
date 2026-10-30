from unittest.mock import Mock

import pytest

from src.generation.rag_chain import FALLBACK_ANSWER, RAGChain
from src.ingestion.chunker import Chunk
from src.retrieval.lite_selector import LiteEvidenceSelector


class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, top_k):
        return [(c, 1.0 / (i + 1)) for i, c in enumerate(self.chunks[:top_k])]


def chain_for(chunks, answer=''):
    llm = Mock()
    llm.generate.return_value = answer
    return RAGChain(Retriever(chunks), llm, reranker=LiteEvidenceSelector()), llm


@pytest.mark.parametrize('code', ['CN10', 'CN11', 'CN100'])
def test_exact_cn1_never_borrows_nearby_codes(code):
    source = Chunk('other', f'Điểm chuẩn {code} năm 2026 là 25,5.', {'title': 'Điểm chuẩn 2026', 'admission_year': '2026'})
    chain, llm = chain_for([source])
    result = chain.answer('điểm chuẩn cn1')
    assert result.route_type == 'out_of_scope'
    assert result.answer == FALLBACK_ANSWER and result.citations == []
    events = list(chain.answer_stream('điểm chuẩn cn1'))
    assert events[0]['route_type'] == 'out_of_scope'
    assert events[0]['citations'] == []
    llm.generate.assert_not_called()


def test_exact_cn1_requires_evidence_even_when_generic_cutoff_words_match():
    source = Chunk('unrelated', 'Điểm chuẩn Kỹ thuật viễn thông năm 2026 là 23,72.',
                   {'doc_id': 'cn1-technical-id', 'title': 'Điểm chuẩn 2026', 'admission_year': '2026'})
    chain, llm = chain_for([source])
    assert chain.answer('điểm chuẩn cn1').route_type == 'out_of_scope'
    llm.generate.assert_not_called()


def test_exact_cn1_still_generates_an_answer():
    source = Chunk('verified', 'Điểm chuẩn CN1 năm 2026 là 25,5.', {'title': 'Điểm chuẩn 2026', 'admission_year': '2026'})
    chain, llm = chain_for([source], 'Điểm chuẩn CN1 năm 2026 là 25,5 [1].')
    result = chain.answer('điểm chuẩn cn1')
    assert result.route_type == 'general'
    assert [c['chunk_id'] for c in result.citations] == ['verified']
    llm.generate.assert_called_once()


TUITION_QUESTION = 'Học phí PTIT năm học 2026–2027 là bao nhiêu?'
TUITION_TEXT = ('Học phí dự kiến năm học 2026-2027. Đại trà: 35,9 đến 40,7 triệu đồng/năm. '
                'Chất lượng cao: 44,8 đến 58,3 triệu đồng/năm. '
                'Liên kết quốc tế: 58,3 đến 67,2 triệu đồng/năm.')
TUITION_ANSWER = ('Học phí dự kiến năm học 2026-2027 [1]:\n\n'
                  '- Đại trà: 35,9 đến 40,7 triệu đồng/năm [1].\n'
                  '- Chất lượng cao: 44,8 đến 58,3 triệu đồng/năm [1].\n'
                  '- Liên kết quốc tế: 58,3 đến 67,2 triệu đồng/năm [1].')


def test_broad_tuition_keeps_all_program_claims_and_json_sse_mapping():
    source = Chunk('tuition', TUITION_TEXT, {'heading': 'Học phí', 'academic_year': '2026-2027'})
    chain, _ = chain_for([source], TUITION_ANSWER)
    result = chain.answer(TUITION_QUESTION)
    events = list(chain.answer_stream(TUITION_QUESTION))
    assert result.route_type == 'general'
    assert ''.join(e.get('text', '') for e in events) == result.answer
    assert next(e['citations'] for e in events if e['type'] == 'metadata') == result.citations
    assert [c['marker'] for c in result.citations] == [1]


def test_tuition_prompt_uses_optional_source_references():
    source = Chunk('tuition', TUITION_TEXT, {'heading': 'Học phí', 'academic_year': '2026-2027'})
    chain, llm = chain_for([source], TUITION_ANSWER)
    chain.answer(TUITION_QUESTION)
    prompt = llm.generate.call_args.kwargs['system_prompt']
    assert 'source_ids là trường tùy chọn' in prompt
    assert 'Có thể để source_ids=[] hoặc bỏ trường này' in prompt


def test_multiple_sources_keep_relational_citation_mapping_in_both_transports():
    chunks = [Chunk('standard', 'Học phí đại trà 35,9 triệu/năm.', {'heading': 'Học phí'}),
              Chunk('quality', 'Học phí chất lượng cao 58,3 triệu/năm.', {'heading': 'Học phí'})]
    answer = ('Học phí đại trà 35,9 triệu/năm [1]. '
              'Học phí chất lượng cao 58,3 triệu/năm [2].')
    chain, _ = chain_for(chunks, answer)
    result = chain.answer('Học phí đại trà và chất lượng cao?')
    events = list(chain.answer_stream('Học phí đại trà và chất lượng cao?'))
    assert result.answer == answer.replace(' [1]', '').replace(' [2]', '')
    assert ''.join(event.get('text', '') for event in events) == answer.replace(' [1]', '').replace(' [2]', '')
    assert next(event['citations'] for event in events if event['type'] == 'metadata') == result.citations
    assert [(citation['marker'], citation['chunk_id']) for citation in result.citations] == [(1, 'standard'), (2, 'quality')]
