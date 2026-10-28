from scripts.phase_a_smoke_contract import response_acceptance


def test_out_of_scope_case_requires_abstention_route_and_no_evidence_refs():
    result = response_acceptance(
        answer="Mình chưa tìm thấy thông tin này trong dữ liệu hiện có.",
        citations=[],
        inline_markers=[],
        route_type="out_of_scope",
        expected_route="out_of_scope",
        expected_evidence_count=0,
        required_terms_present=True,
        marker_mapping_valid=True,
        citation_support_valid=False,
    )
    assert result["acceptance_valid"] is True


def test_out_of_scope_case_rejects_answer_and_citation_even_if_metadata_matches():
    result = response_acceptance(
        answer="PTIT có tuyển ngành này năm 2026 [1].",
        citations=[{"marker": 1, "chunk_id": "unrelated"}],
        inline_markers=[1],
        route_type="general",
        expected_route="out_of_scope",
        expected_evidence_count=0,
        required_terms_present=True,
        marker_mapping_valid=True,
        citation_support_valid=True,
    )
    assert result["acceptance_valid"] is False


def test_evidence_question_rejects_empty_markers_and_citations():
    result = response_acceptance(
        answer="Các mức phí lần lượt là 35,9; 44,8 và 58,3.",
        citations=[],
        inline_markers=[],
        route_type="general",
        expected_route="general",
        expected_evidence_count=3,
        required_terms_present=True,
        marker_mapping_valid=True,
        citation_support_valid=False,
    )
    assert result["acceptance_valid"] is False


def test_evidence_question_requires_supported_marker_mapping():
    result = response_acceptance(
        answer="Ngưỡng là SAT từ 1130/1600 trở lên [1].",
        citations=[{"marker": 1, "chunk_id": "expected"}],
        inline_markers=[1],
        route_type="general",
        expected_route="general",
        expected_evidence_count=1,
        required_terms_present=True,
        marker_mapping_valid=True,
        citation_support_valid=False,
    )
    assert result["acceptance_valid"] is False
