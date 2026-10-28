import logging

import pytest

from src.generation.grounding import RAGGroundingError, logical_claims, validate_answer
from src.ingestion.chunker import Chunk


SOURCES = [
    Chunk("sat", "Điều kiện xét tuyển năm 2026 tại cơ sở Hà Nội: điểm SAT tối thiểu 1130/1600.",
          {"campus": "BVH", "admission_year": 2026}),
    Chunk("act", "Điều kiện xét tuyển năm 2026 tại cơ sở Hà Nội: điểm ACT tối thiểu 25/36.",
          {"campus": "BVH", "admission_year": 2026}),
]


@pytest.mark.parametrize("answer", [
    "Điểm SAT tối thiểu\nlà 1130/1600 [1].",
    "Điểm SAT tối thiểu 1130/1600.\nĐiểm ACT tối thiểu 25/36 [1] [2].",
    "Điểm SAT tối thiểu 1130/1600. Điểm ACT tối thiểu 25/36. [1] [2]",
    "Điểm SAT tối thiểu 1130/1600\n[1].",
    "## Điều kiện xét tuyển\n\n- Điểm SAT tối thiểu\n  1130/1600 [1].\n- Điểm ACT tối thiểu 25/36 [2].",
    "**Ngưỡng điểm xét tuyển**\n\n- Điểm SAT tối thiểu 1130/1600 [1].\n- Điểm ACT tối thiểu 25/36 [2].",
    "__Điều kiện xét tuyển__\n\nĐiểm SAT tối thiểu 1130/1600 [1].",
    "**Điều kiện xét tuyển:**\n1. Điểm SAT tối thiểu 1130/1600 [1].\n2. Điểm ACT tối thiểu 25/36 [2].",
    "- SAT:\n  - Điểm tối thiểu 1130/1600 [1].\n- ACT:\n  - Điểm tối thiểu 25/36 [2].",
    "| Phương thức | Điểm tối thiểu |\n| --- | ---: |\n| SAT | 1130/1600 [1] |\n| ACT | 25/36 [2] |",
    "Phương thức | Điểm tối thiểu\n--- | ---\nSAT | 1130/1600 [1]\nACT | 25/36 [2]",
    "## Cơ sở Hà Nội năm 2026\n\nĐiểm SAT tối thiểu 1130/1600 [1].",
    "Chào bạn!\n\nĐiểm SAT tối thiểu 1130/1600 [1].",
    "Bạn cần đạt một trong các ngưỡng điểm sau:\n- SAT 1130/1600 [1].\n- ACT 25/36 [2].",
    "Ngưỡng điểm tối thiểu để đăng ký xét tuyển là:\n- SAT 1130/1600 [1].\n- ACT 25/36 [2].",
    "# Điểm SAT tối thiểu 1130/1600 [1]\n\nĐiểm ACT tối thiểu 25/36 [2].",
])
def test_markdown_logical_claims_with_verified_sources(answer):
    validate_answer(answer, SOURCES)


@pytest.mark.parametrize("answer,reason", [
    ("Điểm SAT tối thiểu 1130/1600.", "missing_citation"),
    ("Điểm SAT tối thiểu 1130/1600 [99].", "invalid_marker"),
    ("Điểm SAT tối thiểu 1130/1600 [0].", "invalid_marker"),
    ("Điểm SAT tối thiểu 1130/1600.\n\nĐiểm ACT tối thiểu 25/36 [2].", "missing_citation"),
    ("- Điểm SAT tối thiểu 1130/1600\n- Điểm ACT tối thiểu 25/36 [1] [2].", "missing_citation"),
    ("| SAT | 1130/1600 |\n| ACT | 25/36 [1] [2] |", "missing_citation"),
    ("Điểm SAT tối thiểu 1130/1600 [1].\n\nĐiểm ACT tối thiểu 25/36.", "missing_citation"),
    ("Điểm SAT tối thiểu 9999 [1].", "unsupported_figures"),
    ("Điểm SAT tối thiểu 1130/1600 [2].", "unsupported_figures"),
    ("## Năm 2027\n\nĐiểm SAT tối thiểu 1130/1600 [1].", "unsupported_figures"),
    ("| Điểm năm 2027 |\n| --- |\n| SAT 1130/1600 [1] |", "unsupported_figures"),
    ("| Bảo đảm việc làm |\n| --- |\n| SAT 1130/1600 [1] |", "missing_citation"),
    ("## Cơ sở TP.HCM\n\nĐiểm SAT tối thiểu 1130/1600 [1].", "unsupported_campus"),
    ("Điểm SAT tối thiểu CN10 1130/1600 [1].", "unsupported_code"),
    ("Điểm TOEFL tối thiểu 25/36 [2].", "unsupported_exam"),
    ("# Điểm SAT tối thiểu 9999\n\nĐiểm SAT tối thiểu 1130/1600 [1].", "missing_citation"),
    ("**Điểm SAT tối thiểu 9999**\n\nĐiểm SAT tối thiểu 1130/1600 [1].", "missing_citation"),
    ("Bạn không cần đạt ngưỡng điểm sau:\n- SAT 1130/1600 [1].", "missing_citation"),
    ("Điểm SAT tối thiểu 1130/1600 [1]. Bảo đảm việc làm cho mọi sinh viên.", "missing_citation"),
    ("Bảo đảm việc làm cho mọi sinh viên. Điểm SAT tối thiểu 1130/1600 [1].", "insufficient_textual_support"),
])
def test_missing_evidence_and_fabricated_markers_remain_rejected(answer, reason):
    with pytest.raises(RAGGroundingError) as caught:
        validate_answer(answer, SOURCES)
    assert caught.value.reason == reason


def test_paragraph_scope_checks_each_sentence_without_reassigning_explicit_markers():
    claims = logical_claims("Điểm SAT 1130 [1]. Điểm ACT 25 [2].")
    assert [claim.markers for claim in claims] == [(1,), (2,)]
    with pytest.raises(RAGGroundingError):
        validate_answer("Điểm SAT 1130 [2]. Điểm ACT 25 [1].", SOURCES)


def test_diagnostics_separate_missing_citation_from_verified_wrapping(caplog):
    with caplog.at_level(logging.INFO, logger="src.generation.grounding"):
        validate_answer("Điểm SAT tối thiểu\n1130/1600 [1].", SOURCES)
        with pytest.raises(RAGGroundingError) as caught:
            validate_answer("Điểm SAT 1130 [1].\n\nprivate-user-message-unique", SOURCES)
    assert caught.value.claim_index == 2
    assert caught.value.citation_present is False
    assert caught.value.source_count == 2
    assert any("reason=logical_wrapping" in message for message in caplog.messages)
    assert any("reason=missing_citation claim_index=2" in message for message in caplog.messages)
    assert "private-user-message-unique" not in caplog.text


def test_live_sat_act_referential_introduction_is_checked_with_each_cited_item():
    answer = (
        "Để đăng ký xét tuyển vào PTIT năm 2026 theo phương thức xét tuyển dựa vào "
        "kết quả Chứng chỉ đánh giá năng lực quốc tế, bạn cần đạt mức điểm tối thiểu như sau:\n\n"
        "* **Chứng chỉ SAT:** từ **1130/1600** trở lên [1], [5].\n\n"
        "* **Chứng chỉ ACT:** từ **25/36** trở lên [1], [5].\n\n"
        "Lưu ý là các chứng chỉ này phải còn trong thời hạn **02 năm** tính đến ngày bạn đăng ký dự tuyển [1], [5]."
    )
    content = (
        "Tuyển sinh năm 2026. Chứng chỉ đánh giá năng lực quốc tế SAT từ 1130/1600 "
        "trở lên hoặc ACT từ 25/36 trở lên, trong thời hạn 02 năm tính đến ngày đăng ký dự tuyển."
    )
    sources = [Chunk(str(index), content, {"admission_year": "2026"}) for index in range(5)]
    validate_answer(answer, sources)
    claims = logical_claims(answer)
    assert len(claims) == 3
    assert all(claim.markers == (1, 5) for claim in claims)
    assert all("2026" in " ".join(claim.context) for claim in claims[:2])
    with pytest.raises(RAGGroundingError) as caught:
        validate_answer(answer.replace("năm 2026", "năm 2027"), sources)
    assert caught.value.reason == "unsupported_figures"
    with pytest.raises(RAGGroundingError):
        validate_answer(answer.replace("1130/1600", "9999/1600"), sources)


def test_comma_separated_paragraph_citations_preserve_all_source_markers():
    claims = logical_claims("Điểm SAT 1130. Điểm ACT 25 [1], [2].")
    assert [claim.markers for claim in claims] == [(1, 2), (1, 2)]
    validate_answer("Điểm SAT 1130. Điểm ACT 25 [1], [2].", SOURCES)


METHODS = [
    "Xét tuyển tài năng",
    "Xét tuyển dựa vào chứng chỉ SAT hoặc ACT",
    "Xét tuyển kết quả đánh giá năng lực hoặc đánh giá tư duy",
    "Xét tuyển kết hợp IELTS, TOEFL với kết quả học tập THPT",
    "Xét tuyển kết quả thi tốt nghiệp THPT năm 2026",
]
METHOD_SOURCES = [
    Chunk(str(i), ("Năm 2026 sử dụng **0****5** phương thức tuyển sinh. " if i == 1 else "") + method,
          {"admission_year": "2026", "title": "Tuyển sinh năm 2026"})
    for i, method in enumerate(METHODS, 1)
]
METHOD_ANSWER = "PTIT sử dụng 05 phương thức tuyển sinh năm 2026 như sau [1]:\n\n" + "\n\n".join(
    f"{i}. **{method}** [{i}]." for i, method in enumerate(METHODS, 1)
)


def test_five_method_list_validates_count_and_each_item_with_its_own_source():
    validate_answer(METHOD_ANSWER, METHOD_SOURCES)
    claims = logical_claims(METHOD_ANSWER)
    assert len(claims) == 6
    assert claims[0].kind == "list_introduction"
    assert [claim.markers for claim in claims[1:]] == [(i,) for i in range(1, 6)]
    assert all("2026" in " ".join(claim.context) for claim in claims[1:])


@pytest.mark.parametrize("answer,reason", [
    (METHOD_ANSWER.replace("như sau [1]:", "như sau:"), "missing_citation"),
    (METHOD_ANSWER.replace("05 phương thức", "06 phương thức"), "unsupported_figures"),
    (METHOD_ANSWER.replace("năm 2026 như sau", "năm 2027 như sau"), "unsupported_figures"),
    (METHOD_ANSWER.replace("Xét tuyển tài năng", "Xét tuyển bằng bốc thăm ngẫu nhiên"), "insufficient_textual_support"),
    (METHOD_ANSWER.replace("Xét tuyển tài năng** [1]", "Xét tuyển học bạ THPT** [5]"), "unsupported_method"),
    (METHOD_ANSWER.replace("tài năng** [1]", "tài năng** [5]"), "unsupported_method"),
    (METHOD_ANSWER.replace("SAT hoặc ACT** [2]", "SAT hoặc ACT** [3]"), "unsupported_exam"),
    (METHOD_ANSWER.replace("như sau [1]:", "như sau [99]:"), "invalid_marker"),
    (METHOD_ANSWER.replace("tài năng** [1]", "tài năng**"), "missing_citation"),
])
def test_five_method_list_rejects_missing_or_wrong_evidence(answer, reason):
    with pytest.raises(RAGGroundingError) as caught:
        validate_answer(answer, METHOD_SOURCES)
    assert caught.value.reason == reason


def test_method_list_year_scope_cannot_be_borrowed_from_intro_source():
    # The intro and talent item are supported for 2027, but the SAT item is not.
    changed = [Chunk("1", METHOD_SOURCES[0].content.replace("2026", "2027"), {"admission_year": "2027"}), *METHOD_SOURCES[1:]]
    with pytest.raises(RAGGroundingError) as caught:
        validate_answer(METHOD_ANSWER.replace("năm 2026 như sau", "năm 2027 như sau"), changed)
    assert caught.value.reason == "unsupported_figures"
    assert caught.value.claim_index == 3


def test_method_list_preserves_document_order_when_marker_is_not_method_number():
    sources = [METHOD_SOURCES[0], METHOD_SOURCES[1], METHOD_SOURCES[3], METHOD_SOURCES[2], METHOD_SOURCES[4]]
    answer = METHOD_ANSWER.replace("[3]", "[swapped]").replace("[4]", "[3]").replace("[swapped]", "[4]")
    validate_answer(answer, sources)
    with pytest.raises(RAGGroundingError) as caught:
        validate_answer(METHOD_ANSWER, sources)
    assert caught.value.reason == "unsupported_method"


RELATIONAL_CASES = [
    pytest.param(
        "Học phí đại trà 35,9 triệu đồng/năm. Học phí chất lượng cao 58,3 triệu đồng/năm.", {},
        "Học phí đại trà 35,9 triệu đồng/năm [1].",
        "Học phí đại trà 58,3 triệu đồng/năm [1].", id="audit-tuition-inversion",
    ),
    pytest.param(
        "Học phí đại trà 35,9 triệu đồng/năm, chất lượng cao 58,3 triệu đồng/năm.", {},
        "Học phí chất lượng cao 58,3 triệu đồng/năm [1].",
        "Học phí chất lượng cao 35,9 triệu đồng/năm [1].", id="same-sentence-tuition",
    ),
    pytest.param(
        "Tuition: standard 35.9; high-quality 58.3.", {},
        "Tuition standard 35.9 [1].", "Tuition standard 58.3 [1].", id="english-tuition",
    ),
    pytest.param(
        "Tuition: standard 35.9; high-quality 58.3.", {},
        "Tuition high-quality 58.3 [1].", "Tuition high-quality 35.9 [1].", id="shared-semicolon-prefix",
    ),
    pytest.param(
        "Điểm SAT tối thiểu 1130/1600 hoặc ACT tối thiểu 25/36.", {},
        "Điểm SAT tối thiểu 1130/1600 [1].",
        "Điểm SAT tối thiểu 25/36 [1].", id="same-sentence-sat-act",
    ),
    pytest.param(
        "Điểm SAT tối thiểu 1130/1600. Điểm ACT tối thiểu 25/36.", {},
        "Điểm ACT tối thiểu 25/36 [1].",
        "Điểm ACT tối thiểu 1130/1600 [1].", id="exam-sentences",
    ),
    pytest.param(
        "Điểm chuẩn Hà Nội 27,5, TP.HCM 25,5.", {"campus": "BVH BVS"},
        "Điểm chuẩn Hà Nội 27,5 [1].", "Điểm chuẩn Hà Nội 25,5 [1].", id="campus-cutoff",
    ),
    pytest.param(
        "Điểm chuẩn CN1 25,5, CN2 27,5.", {},
        "Điểm chuẩn CN1 25,5 [1].", "Điểm chuẩn CN1 27,5 [1].", id="program-code",
    ),
    pytest.param(
        "Học phí đại trà năm 2025 là 35,9 triệu. Học phí đại trà năm 2026 là 40,7 triệu.", {},
        "Học phí đại trà năm 2026 là 40,7 triệu [1].",
        "Học phí đại trà năm 2026 là 35,9 triệu [1].", id="year-mismatch",
    ),
    pytest.param(
        "Học phí ngành Công nghệ thông tin 35,9 triệu. Học phí ngành An toàn thông tin 58,3 triệu.", {},
        "Học phí ngành Công nghệ thông tin 35,9 triệu [1].",
        "Học phí ngành Công nghệ thông tin 58,3 triệu [1].", id="named-major",
    ),
    pytest.param(
        "## Học phí năm 2026\n\n- Đại trà: 35,9 triệu/năm.\n- Chất lượng cao: 58,3 triệu/năm.", {},
        "Học phí đại trà năm 2026 là 35,9 triệu/năm [1].",
        "Học phí đại trà năm 2026 là 58,3 triệu/năm [1].", id="markdown-evidence-list",
    ),
    pytest.param(
        "| Chương trình | Học phí |\n| --- | --- |\n| Đại trà | 35,9 triệu/năm |\n| Chất lượng cao | 58,3 triệu/năm |", {},
        "Học phí đại trà 35,9 triệu/năm [1].",
        "Học phí đại trà 58,3 triệu/năm [1].", id="markdown-evidence-table",
    ),
    pytest.param(
        "| Chương trình | 2025 | 2026 |\n| --- | --- | --- |\n| Đại trà | 35,9 | 40,7 |\n| Chất lượng cao | 58,3 | 60 |",
        {"heading": "Học phí"},
        "Học phí đại trà năm 2026 là 40,7 [1].",
        "Học phí đại trà năm 2026 là 35,9 [1].", id="year-column-binding",
    ),
    pytest.param(
        "| Năm | Đại trà | Chất lượng cao |\n| --- | --- | --- |\n| 2026 | 35,9 | 58,3 |",
        {"heading": "Học phí"},
        "Học phí đại trà năm 2026 là 35,9 [1].",
        "Học phí đại trà năm 2026 là 58,3 [1].", id="program-column-binding",
    ),
    pytest.param(
        "## Học phí năm 2025\n\nĐại trà: 35,9 triệu.\n\n## Học phí năm 2026\n\nĐại trà: 40,7 triệu.", {},
        "Học phí đại trà năm 2026 là 40,7 triệu [1].",
        "Học phí đại trà năm 2026 là 35,9 triệu [1].", id="section-year-binding",
    ),
]


@pytest.mark.parametrize("content,metadata,positive,negative", RELATIONAL_CASES)
def test_supported_entity_value_relationships(content, metadata, positive, negative):
    validate_answer(positive, [Chunk("relational", content, metadata)])


@pytest.mark.parametrize("content,metadata,positive,negative", RELATIONAL_CASES)
def test_swapped_entity_value_relationships_are_rejected(content, metadata, positive, negative):
    with pytest.raises(RAGGroundingError) as caught:
        validate_answer(negative, [Chunk("relational", content, metadata)])
    assert caught.value.reason == "unsupported_relationship"


@pytest.mark.parametrize("answer", [
    "Học phí đại trà 35,9 và chất lượng cao 58,3 triệu/năm [1] [2].",
    "Học phí đại trà 35,9 triệu/năm [1]. Học phí chất lượng cao 58,3 triệu/năm [2].",
    "## Học phí\n\n- Đại trà: 35,9 triệu/năm [1].\n- Chất lượng cao: 58,3 triệu/năm [2].",
    "| Chương trình | Học phí |\n| --- | --- |\n| Đại trà | 35,9 triệu/năm [1] |\n| Chất lượng cao | 58,3 triệu/năm [2] |",
    "Học phí đại trà 35,9 triệu/năm. Học phí chất lượng cao 58,3 triệu/năm [1], [2].",
])
def test_multiple_citations_preserve_independently_supported_relationships(answer):
    sources = [Chunk("standard", "Học phí đại trà 35,9 triệu/năm.", {}),
               Chunk("quality", "Học phí chất lượng cao 58,3 triệu/năm.", {})]
    validate_answer(answer, sources)


@pytest.mark.parametrize("answer", [
    "Học phí đại trà 58,3 và chất lượng cao 35,9 triệu/năm [1] [2].",
    "## Học phí\n\n- Đại trà: 58,3 triệu/năm [1] [2].",
    "| Chương trình | Học phí |\n| --- | --- |\n| Đại trà | 58,3 triệu/năm [1] [2] |",
])
def test_multiple_citations_cannot_pool_subjects_and_values(answer):
    sources = [Chunk("standard", "Học phí đại trà 35,9 triệu/năm.", {}),
               Chunk("quality", "Học phí chất lượng cao 58,3 triệu/năm.", {})]
    with pytest.raises(RAGGroundingError) as caught:
        validate_answer(answer, sources)
    assert caught.value.reason == "unsupported_relationship"


def test_year_and_program_must_coexist_in_one_supported_relationship():
    sources = [Chunk("old", "Học phí đại trà năm 2025 là 35,9 triệu.", {}),
               Chunk("new", "Học phí chất lượng cao năm 2026 là 58,3 triệu.", {})]
    with pytest.raises(RAGGroundingError) as caught:
        validate_answer("Học phí đại trà năm 2026 là 35,9 triệu [1] [2].", sources)
    assert caught.value.reason == "unsupported_relationship"


@pytest.mark.parametrize("answer", [
    "Học phí 58,3 triệu/năm [1].",
    "Học phí đại trà và chất lượng cao 58,3 triệu/năm [1].",
])
def test_ambiguous_program_relationships_fail_closed(answer):
    source = Chunk("ambiguous", "Học phí đại trà 35,9 triệu/năm, chất lượng cao 58,3 triệu/năm.", {})
    with pytest.raises(RAGGroundingError) as caught:
        validate_answer(answer, [source])
    assert caught.value.reason == "unsupported_relationship"


def test_exam_validity_cannot_be_reused_as_an_exam_score():
    source = Chunk("sat", "Điểm SAT tối thiểu 1130/1600, thời hạn 02 năm.", {})
    validate_answer("Chứng chỉ có thời hạn 02 năm [1].", [source])
    with pytest.raises(RAGGroundingError) as caught:
        validate_answer("Điểm SAT tối thiểu 02 [1].", [source])
    assert caught.value.reason == "unsupported_relationship"


@pytest.mark.parametrize("answer", [
    "Học phí đại trà 35,9 và chất lượng cao 58,3, 67,2 triệu/năm [1].",
    "Học phí đại trà 35,9 và chất lượng cao, liên kết quốc tế 58,3 triệu/năm [1].",
    "Học phí đại trà 35,9 triệu/năm và chất lượng cao [1].",
])
def test_partial_or_ambiguous_relation_parsing_cannot_accept_the_rest(answer):
    source = Chunk("partial", "Học phí đại trà 35,9 triệu/năm. Học phí chất lượng cao 58,3 triệu/năm. "
                   "Học phí liên kết quốc tế 67,2 triệu/năm.", {})
    with pytest.raises(RAGGroundingError) as caught:
        validate_answer(answer, [source])
    assert caught.value.reason == "unsupported_relationship"


def test_coordinated_program_claims_keep_the_explicit_year_scope():
    sources = [Chunk("standard", "Học phí đại trà năm 2026 là 35,9 triệu/năm.", {}),
               Chunk("quality", "Học phí chất lượng cao năm 2025 là 58,3 triệu/năm.", {})]
    answer = "Học phí đại trà năm 2026 là 35,9 và chất lượng cao 58,3 triệu/năm [1] [2]."
    with pytest.raises(RAGGroundingError) as caught:
        validate_answer(answer, sources)
    assert caught.value.reason == "unsupported_relationship"


@pytest.mark.parametrize("content", [
    "Học phí đại trà và chất lượng cao là 35,9 và 58,3 triệu/năm.",
    "| Chương trình | Học phí |\n| --- | --- |\n| Đại trà | 35,9 | 58,3 |",
    "| Đại trà | 35,9 | 58,3 |",
])
def test_ambiguous_evidence_cannot_supply_a_program_value_pair(content):
    with pytest.raises(RAGGroundingError) as caught:
        validate_answer("Học phí đại trà 58,3 triệu/năm [1].",
                        [Chunk("ambiguous", content, {"heading": "Học phí triệu/năm"})])
    assert caught.value.reason == "unsupported_relationship"


def test_metadata_cannot_supply_a_numeric_payload_from_another_entity():
    source = Chunk("metadata", "Học phí chất lượng cao 58,3 triệu/năm.",
                   {"title": "Học phí đại trà 35,9 triệu/năm"})
    with pytest.raises(RAGGroundingError) as caught:
        validate_answer("Học phí đại trà 35,9 triệu/năm [1].", [source])
    assert caught.value.reason == "unsupported_relationship"


@pytest.mark.parametrize("supported", [True, False])
@pytest.mark.parametrize("table", [True, False])
def test_year_itself_is_bound_to_the_program_even_without_an_amount(supported, table):
    text = ("| Chương trình | Năm |\n| --- | --- |\n| Đại trà | 2025 |\n| Chất lượng cao | 2026 |"
            if table else "Chương trình đại trà năm 2025. Chương trình chất lượng cao năm 2026.")
    year = "2025" if supported else "2026"
    answer = f"Chương trình đại trà năm {year} [1]."
    if supported:
        validate_answer(answer, [Chunk("year", text, {})])
    else:
        with pytest.raises(RAGGroundingError) as caught:
            validate_answer(answer, [Chunk("year", text, {})])
        assert caught.value.reason == "unsupported_relationship"


@pytest.mark.parametrize("text", [
    "Mình chưa tìm thấy thông tin này trong dữ liệu tuyển sinh hiện có.",
    "Mình chưa có đủ thông tin trong dữ liệu tuyển sinh hiện có để trả lời câu hỏi này.",
    "Mình chưa có đủ bằng chứng trong dữ liệu tuyển sinh hiện có để trả lời câu hỏi này.",
])
def test_strict_grounding_has_no_blanket_abstention_or_citation_exemption(text):
    # Only the chain's exact whole-output acceptance gate handles abstentions.
    # Direct callers retain the same strict validator, including F01 checks.
    with pytest.raises(RAGGroundingError) as error:
        validate_answer(text, SOURCES)
    assert error.value.reason == "missing_citation"
