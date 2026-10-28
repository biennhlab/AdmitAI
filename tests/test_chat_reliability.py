from unittest.mock import Mock

import pytest

from src.generation.rag_chain import FALLBACK_ANSWER, RAGChain
from src.generation.grounding import RAGGroundingError
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


def test_verified_exact_cn1_still_generates_a_grounded_answer():
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


@pytest.mark.parametrize('unsupported', [TUITION_ANSWER.replace('67,2', '99,9'), TUITION_ANSWER.replace('[1]', '[99]'), TUITION_ANSWER.replace(' [1]', '')])
def test_broad_tuition_rejects_fabricated_figures_markers_and_missing_citations(unsupported):
    source = Chunk('tuition', TUITION_TEXT, {'heading': 'Học phí', 'academic_year': '2026-2027'})
    chain, _ = chain_for([source], unsupported)
    with pytest.raises(RAGGroundingError):
        chain.answer(TUITION_QUESTION)
    with pytest.raises(RAGGroundingError):
        next(chain.answer_stream(TUITION_QUESTION))


def test_live_tuition_factual_introduction_requires_its_own_citation():
    introduction = 'Học phí dự kiến cho năm học 2026–2027 tại PTIT sẽ tùy thuộc vào chương trình học mà bạn lựa chọn:'
    source = Chunk('tuition', TUITION_TEXT, {'heading': 'Học phí', 'academic_year': '2026-2027'})
    answer = introduction + '\n\n- Đại trà: 35,9 đến 40,7 triệu đồng/năm [1].'
    chain, _ = chain_for([source], answer)
    with pytest.raises(RAGGroundingError) as caught:
        chain.answer(TUITION_QUESTION)
    assert caught.value.reason == 'missing_citation'
    assert caught.value.claim_index == 1
    assert caught.value.citation_present is False
    with pytest.raises(RAGGroundingError):
        next(chain.answer_stream(TUITION_QUESTION))


def test_tuition_prompt_explicitly_requires_citations_on_factual_introductions():
    source = Chunk('tuition', TUITION_TEXT, {'heading': 'Học phí', 'academic_year': '2026-2027'})
    chain, llm = chain_for([source], TUITION_ANSWER)
    chain.answer(TUITION_QUESTION)
    prompt = llm.generate.call_args.kwargs['system_prompt']
    assert 'câu dẫn cũng phải có marker từ tài liệu hỗ trợ' in prompt
    assert 'Citation ở các mục bên dưới không thay thế citation' in prompt


@pytest.mark.parametrize('source,metadata,question,supported,swapped', [
    ('Học phí đại trà 35,9 triệu/năm. Học phí chất lượng cao 58,3 triệu/năm.',
     {'heading': 'Học phí'}, 'Học phí đại trà?',
     'Học phí đại trà 35,9 triệu/năm [1].', 'Học phí đại trà 58,3 triệu/năm [1].'),
    ('Điểm SAT tối thiểu 1130/1600 hoặc ACT tối thiểu 25/36.',
     {'heading': 'Điều kiện xét tuyển'}, 'Điểm SAT tối thiểu?',
     'Điểm SAT tối thiểu 1130/1600 [1].', 'Điểm SAT tối thiểu 25/36 [1].'),
    ('Điểm chuẩn Hà Nội 27,5, TP.HCM 25,5.',
     {'heading': 'Điểm chuẩn', 'campus': 'BVH BVS'}, 'Điểm chuẩn Hà Nội?',
     'Điểm chuẩn Hà Nội 27,5 [1].', 'Điểm chuẩn Hà Nội 25,5 [1].'),
    ('Học phí đại trà năm 2025 là 35,9 triệu. Học phí đại trà năm 2026 là 40,7 triệu.',
     {'heading': 'Học phí', 'admission_year': '2026'}, 'Học phí đại trà năm 2026?',
     'Học phí đại trà năm 2026 là 40,7 triệu [1].', 'Học phí đại trà năm 2026 là 35,9 triệu [1].'),
])
def test_relational_grounding_is_identical_before_json_and_sse_output(
    source, metadata, question, supported, swapped,
):
    chunk = Chunk('relation', source, metadata)
    chain, llm = chain_for([chunk], supported)
    result = chain.answer(question)
    events = list(chain.answer_stream(question))
    assert result.answer == supported.replace(' [1]', '')
    assert ''.join(event.get('text', '') for event in events) == supported.replace(' [1]', '')
    assert next(event['citations'] for event in events if event['type'] == 'metadata') == result.citations
    assert [citation['chunk_id'] for citation in result.citations] == ['relation']
    llm.generate.return_value = swapped
    with pytest.raises(RAGGroundingError) as direct:
        chain.answer(question)
    with pytest.raises(RAGGroundingError) as streaming:
        next(chain.answer_stream(question))
    assert direct.value.reason == streaming.value.reason == 'unsupported_relationship'


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


def _safe_abstention_outputs():
    import unicodedata

    return [
        FALLBACK_ANSWER,
        FALLBACK_ANSWER.removesuffix('.'),
        FALLBACK_ANSWER.upper(),
        '\n\t' + FALLBACK_ANSWER.replace(' ', ' \n\t') + '\n',
        unicodedata.normalize('NFD', FALLBACK_ANSWER),
        'Mình chưa có đủ thông tin trong dữ liệu tuyển sinh hiện có để trả lời câu hỏi này.',
        'Mình chưa có đủ bằng chứng trong dữ liệu tuyển sinh hiện có để trả lời câu hỏi này.',
        f'<answer>{FALLBACK_ANSWER}</answer>',
        f'<response>\n{FALLBACK_ANSWER}\n</response>',
        f'<assistant_answer>{FALLBACK_ANSWER}</assistant_answer>',
        f'<final>{FALLBACK_ANSWER}</final>',
        f'<ANSWER>{FALLBACK_ANSWER}</ANSWER>',
    ]


@pytest.mark.parametrize('transport', ['json', 'sse'])
@pytest.mark.parametrize('output', _safe_abstention_outputs())
def test_narrow_pure_abstentions_return_only_canonical_fallback(output, transport):
    from unittest.mock import patch

    chain, llm = chain_for([Chunk('sat', 'Điểm SAT tối thiểu 1130/1600.', {})], output)
    with patch('src.generation.rag_chain.validate_answer', side_effect=AssertionError('Pure abstention is not a factual claim')):
        if transport == 'json':
            result = chain.answer('Điểm SAT tối thiểu?')
            assert (result.answer, result.citations, result.route_type) == (FALLBACK_ANSWER, [], 'out_of_scope')
        else:
            assert list(chain.answer_stream('Điểm SAT tối thiểu?')) == [
                {'type': 'metadata', 'citations': [], 'route_type': 'out_of_scope'},
                {'type': 'chunk', 'text': FALLBACK_ANSWER},
            ]
    llm.generate.assert_called_once()
    llm.generate_stream.assert_not_called()


_UNSAFE_ABSTENTION_OUTPUTS = [
    pytest.param(FALLBACK_ANSWER + ' Học phí đại trà 99,9 triệu/năm [1].', id='unsupported-tuition'),
    pytest.param(FALLBACK_ANSWER + '\n\nNgành Thú y được tuyển sinh [1].', id='unsupported-program'),
    pytest.param(FALLBACK_ANSWER + ' Điểm SAT tối thiểu 9999 [1].', id='unsupported-score'),
    pytest.param('Điểm SAT tối thiểu 9999 [1]. ' + FALLBACK_ANSWER, id='fact-before-abstention'),
    pytest.param(FALLBACK_ANSWER + ' [99]', id='invented-citation'),
    pytest.param(FALLBACK_ANSWER + ' [1]', id='citation-appended'),
    pytest.param('Xin lỗi, ' + FALLBACK_ANSWER, id='non-allowlisted-prefix'),
    pytest.param(FALLBACK_ANSWER + ' Bạn hãy thử lại.', id='extra-sentence'),
    pytest.param(FALLBACK_ANSWER + '..', id='extra-punctuation'),
    pytest.param(FALLBACK_ANSWER.replace('tuyển sinh', 'tuyển sinh năm 2026'), id='year-inserted'),
    pytest.param('Minh chua tim thay thong tin nay trong du lieu tuyen sinh hien co.', id='accentless-near-match'),
    pytest.param('<|channel|analysis>' + FALLBACK_ANSWER, id='provider-channel'),
    pytest.param(FALLBACK_ANSWER + '<|turn|>', id='provider-turn'),
    pytest.param('<analysis>PRIVATE_INTERNAL</analysis>' + FALLBACK_ANSWER, id='hidden-analysis'),
    pytest.param('<thought>PRIVATE_INTERNAL</thought>' + FALLBACK_ANSWER, id='hidden-thought'),
    pytest.param('<reasoning>PRIVATE_INTERNAL</reasoning>' + FALLBACK_ANSWER, id='hidden-reasoning'),
    pytest.param('<answer>' + FALLBACK_ANSWER, id='unclosed-wrapper'),
    pytest.param(FALLBACK_ANSWER + '</answer>', id='stray-close'),
    pytest.param('<answer>' + FALLBACK_ANSWER + '</final>', id='mismatched-wrapper'),
    pytest.param('<answer><final>' + FALLBACK_ANSWER + '</final></answer>', id='nested-wrapper'),
    pytest.param('<analysis>PRIVATE_INTERNAL</analysis>', id='only-internal'),
    pytest.param('<analysis>PRIVATE_INTERNAL ' + FALLBACK_ANSWER, id='unclosed-internal'),
    pytest.param('```\n' + FALLBACK_ANSWER + '\n```', id='code-fence'),
    pytest.param('**' + FALLBACK_ANSWER + '**', id='markdown-emphasis'),
    pytest.param('\x00' + FALLBACK_ANSWER, id='null-control'),
    pytest.param('\x85' + FALLBACK_ANSWER, id='stripped-control'),
    pytest.param('\u200b' + FALLBACK_ANSWER, id='zero-width-control'),
    pytest.param('', id='empty'),
    pytest.param(' \n\t', id='blank'),
    pytest.param(None, id='none'),
    pytest.param(123, id='number'),
    pytest.param({}, id='object'),
    pytest.param([], id='array'),
    pytest.param(b'not text', id='bytes'),
]


@pytest.mark.parametrize('transport', ['json', 'sse'])
@pytest.mark.parametrize('output', _UNSAFE_ABSTENTION_OUTPUTS)
def test_mixed_control_and_malformed_outputs_never_receive_abstention_exemption(output, transport):
    chain, llm = chain_for([Chunk('sat', 'Điểm SAT tối thiểu 1130/1600.', {})], output)
    with pytest.raises(RAGGroundingError):
        if transport == 'json':
            chain.answer('Điểm SAT tối thiểu?')
        else:
            next(chain.answer_stream('Điểm SAT tối thiểu?'))
    llm.generate.assert_called_once()
    llm.generate_stream.assert_not_called()


@pytest.mark.parametrize('transport', ['json', 'sse'])
@pytest.mark.parametrize('supported', [True, False])
def test_factual_answers_always_use_unchanged_relational_grounding(transport, supported):
    from unittest.mock import patch
    from src.generation.grounding import validate_answer

    chunk = Chunk('tuition', 'Học phí đại trà 35,9 triệu/năm. Học phí chất lượng cao 58,3 triệu/năm.', {})
    answer = f'Học phí đại trà {"35,9" if supported else "58,3"} triệu/năm [1].'
    chain, llm = chain_for([chunk], answer)
    with patch('src.generation.rag_chain.validate_answer', wraps=validate_answer) as validate:
        def invoke():
            return chain.answer('Học phí đại trà?') if transport == 'json' else list(chain.answer_stream('Học phí đại trà?'))
        if supported:
            result = invoke()
            if transport == 'json':
                assert result.answer == answer.replace(' [1]', '').replace(' [2]', '') and result.route_type == 'general'
            else:
                assert next(event for event in result if event['type'] == 'metadata')['route_type'] == 'general'
        else:
            with pytest.raises(RAGGroundingError) as error:
                invoke()
            assert error.value.reason == 'unsupported_relationship'
        validate.assert_called_once()
        assert validate.call_args.args[0] == answer
    llm.generate.assert_called_once()
