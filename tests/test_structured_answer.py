"""Rendering checks concern JSON shape, never factual support or citation gaps."""
import json

import pytest

from src.generation.structured_answer import GenerationFormatError, render_structured_answer


def answer(block):
    return json.dumps({'status': 'answer', 'blocks': [block]}, ensure_ascii=False)


@pytest.mark.parametrize('kind', ['paragraph', 'bullet', 'numbered_item', 'heading'])
@pytest.mark.parametrize('ids', [None, [], [1], [99], [True], ['1'], [1, 1], '1'])
def test_source_references_are_optional_and_never_verify_content(kind, ids):
    block = {'kind': kind, 'text': 'Điểm SAT tối đa 9999.'}
    if ids is not None: block['source_ids'] = ids
    result = render_structured_answer(answer(block), 1)
    assert 'Điểm SAT tối đa 9999.' in result.text
    assert '[99]' not in result.text
    expected = isinstance(ids, list) and any(type(value) is int and value == 1 for value in ids)
    assert result.text.count('[1]') == int(expected)


def test_table_rows_render_without_sources_and_do_not_check_values_against_evidence():
    data = {'kind': 'table', 'columns': ['Kỳ thi', 'Điểm'], 'rows': [
        {'cells': ['SAT', '9999']}, {'cells': ['ACT', '1130'], 'source_ids': [2]},
    ]}
    result = render_structured_answer(answer(data), 2)
    assert '| SAT | 9999 |' in result.text
    assert '| ACT | 1130 [2] |' in result.text


@pytest.mark.parametrize('raw', [
    '{"status":"answer","blocks":[',
    '{"status":"abstain","status":"answer","blocks":[]}',
    '{"status":"answer","blocks":[],"extra":NaN}',
    '{"status":"abstain","blocks":[]} trailing text',
    '```json\n{"status":"abstain","blocks":[]}',
])
def test_unrenderable_json_is_a_format_error(raw):
    with pytest.raises(GenerationFormatError) as error:
        render_structured_answer(raw, 1)
    assert error.value.code == 'INVALID_GENERATION'


@pytest.mark.parametrize('block', [
    {'kind': 'paragraph', 'text': ''},
    {'kind': 'unknown', 'text': 'Answer'},
    {'kind': ['paragraph'], 'text': 'Answer'},
    {'kind': 'table', 'columns': ['Only one column'], 'rows': []},
    {'kind': 'table', 'columns': ['A', 'B'], 'rows': [{'cells': ['A']}]},
])
def test_invalid_block_shapes_remain_format_errors(block):
    with pytest.raises(GenerationFormatError): render_structured_answer(answer(block), 2)


def test_plain_text_and_explicit_abstention_render_without_a_verification_step():
    assert render_structured_answer('Uncited answer.', 1) is None
    result = render_structured_answer('{"status":"abstain","blocks":[]}', 0)
    assert result.abstained and result.text == ''


def test_complete_json_fence_and_ordered_lists():
    raw = json.dumps({'status': 'answer', 'blocks': [
        {'kind': 'numbered_item', 'text': 'First'}, {'kind': 'numbered_item', 'text': 'Second'},
    ]})
    result = render_structured_answer('```json\n' + raw + '\n```', 0)
    assert result.text == '1. First\n2. Second'
