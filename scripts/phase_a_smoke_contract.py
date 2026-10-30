"""Acceptance rules for Phase A JSON/SSE smoke responses."""

from __future__ import annotations


_ABSTENTION_PHRASES = (
    "chưa tìm thấy",
    "không tìm thấy",
    "không có thông tin",
    "chưa có thông tin",
    "không đủ thông tin",
    "không đủ dữ liệu",
    "không thể xác nhận",
)


def response_acceptance(
    *,
    answer: str,
    citations: list[dict],
    inline_markers: list[int],
    route_type: str | None,
    expected_route: str,
    expected_evidence_count: int,
    required_terms_present: bool,
    marker_mapping_valid: bool,
    citation_support_valid: bool,
) -> dict[str, bool]:
    """Validate one transport response independently; never infer support from hashes."""
    answer_nonempty = bool(answer.strip())
    route_matches = route_type == expected_route
    no_markers = not inline_markers
    no_citations = not citations

    if expected_route == "out_of_scope":
        folded = " ".join(answer.casefold().split())
        abstention = any(phrase in folded for phrase in _ABSTENTION_PHRASES)
        valid = (
            route_matches
            and answer_nonempty
            and abstention
            and no_markers
            and no_citations
        )
        return {
            "route_matches_expected": route_matches,
            "answer_nonempty": answer_nonempty,
            "answer_is_abstention": abstention,
            "no_inline_markers": no_markers,
            "no_citations": no_citations,
            "acceptance_valid": valid,
        }

    evidence_required = expected_evidence_count > 0
    valid = route_matches and answer_nonempty and required_terms_present
    if evidence_required:
        valid = (
            valid
            and bool(inline_markers)
            and bool(citations)
            and marker_mapping_valid
            and citation_support_valid
        )
    return {
        "route_matches_expected": route_matches,
        "answer_nonempty": answer_nonempty,
        "required_terms_present": required_terms_present,
        "evidence_required": evidence_required,
        "markers_present": bool(inline_markers),
        "citations_present": bool(citations),
        "marker_mapping_valid": marker_mapping_valid,
        "citation_support_valid": citation_support_valid,
        "acceptance_valid": valid,
    }
