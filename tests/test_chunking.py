from __future__ import annotations

from collections import Counter

import pytest

from src.ingestion.chunker import (
    combined_chunk,
    naive_chunk,
    parent_child_chunk,
    structure_aware_chunk,
    table_aware_chunk,
)
from src.ingestion.parser import Heading, ParsedPage, Table


BASE_METADATA = {
    "doc_id": "ptit-doc-01",
    "source": "Đề án tuyển sinh PTIT",
    "source_file": "de-an.pdf",
    "source_url": "https://example.test/de-an.pdf",
    "section": "Toàn văn",
}


def page(
    text: str = "",
    *,
    number: int = 1,
    headings: list[Heading] | None = None,
    tables: list[Table] | None = None,
) -> ParsedPage:
    return ParsedPage(number, text, tables=tables or [], headings=headings or [])


def test_naive_chunk_remains_backward_compatible() -> None:
    chunks = naive_chunk("0123456789a", chunk_size=6, overlap=2, metadata={"doc_id": "doc"})

    assert [chunk.content for chunk in chunks] == ["012345", "456789", "89a"]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert [chunk.chunk_id for chunk in chunks] == [
        chunk.chunk_id
        for chunk in naive_chunk("0123456789a", chunk_size=6, overlap=2, metadata={"doc_id": "doc"})
    ]


def test_structure_aware_never_combines_two_heading_sections() -> None:
    first = Heading("I. Phương thức tuyển sinh", 1, 1)
    second = Heading("II. Chỉ tiêu", 1, 1)
    source = page(
        f"{first.text}\nNội dung phương thức.\n\n{second.text}\nNội dung chỉ tiêu.",
        headings=[first, second],
    )

    chunks = structure_aware_chunk([source], max_size=500, metadata=BASE_METADATA)

    assert len(chunks) == 2
    assert chunks[0].content.startswith(first.text)
    assert chunks[1].content.startswith(second.text)
    assert all(not (first.text in chunk.content and second.text in chunk.content) for chunk in chunks)


def test_structure_aware_keeps_paragraph_boundary_when_possible() -> None:
    paragraph_one = "Đoạn thứ nhất có dữ liệu tuyển sinh."
    paragraph_two = "Đoạn thứ hai có thông tin học phí."
    chunks = structure_aware_chunk([page(f"{paragraph_one}\n\n{paragraph_two}")], max_size=200)

    assert len(chunks) == 1
    assert chunks[0].content == f"{paragraph_one}\n\n{paragraph_two}"


def test_structure_aware_carries_heading_context_to_next_page() -> None:
    heading = Heading("III. Học phí", 2, 3)
    chunks = structure_aware_chunk(
        [
            page(f"{heading.text}\nMức học phí cơ bản.", number=3, headings=[heading]),
            page("Thông tin tiếp tục ở trang sau.", number=4),
        ],
        max_size=100,
        metadata=BASE_METADATA,
    )

    second_page = [chunk for chunk in chunks if chunk.metadata["page_number"] == 4]
    assert second_page
    assert second_page[0].content.startswith(heading.text)
    assert second_page[0].metadata["heading"] == heading.text


def test_table_chunk_preserves_header_and_heading_context() -> None:
    heading = Heading("Điểm chuẩn năm 2025", 2, 2)
    table = Table([["Ngành", "Điểm"], ["CNTT", "27.0"]], page_number=2)

    chunks = table_aware_chunk(
        [page(f"{heading.text}\nNgành Điểm", number=2, headings=[heading], tables=[table])],
        metadata=BASE_METADATA,
    )

    assert len(chunks) == 1
    assert "| Ngành | Điểm |" in chunks[0].content
    assert "| CNTT | 27.0 |" in chunks[0].content
    assert chunks[0].metadata["heading"] == heading.text
    assert chunks[0].metadata["table_title"] == heading.text
    assert chunks[0].metadata["page_number"] == 2


def test_multiple_tables_use_the_heading_immediately_above_each_table() -> None:
    first_heading = Heading("Chỉ tiêu miền Bắc", 2, 1)
    second_heading = Heading("Chỉ tiêu miền Nam", 2, 1)
    first_table = Table([["Ngành", "BVH"], ["CNTT", "500"]], 1)
    second_table = Table([["Ngành", "BVS"], ["CNTT", "300"]], 1)
    text = (
        f"{first_heading.text}\n| Ngành | BVH |\n| --- | --- |\n| CNTT | 500 |\n\n"
        f"{second_heading.text}\n| Ngành | BVS |\n| --- | --- |\n| CNTT | 300 |"
    )

    chunks = table_aware_chunk(
        [page(text, headings=[first_heading, second_heading], tables=[first_table, second_table])],
        metadata=BASE_METADATA,
    )

    assert [chunk.metadata["heading"] for chunk in chunks] == [first_heading.text, second_heading.text]


def test_large_table_splits_by_rows_and_repeats_header() -> None:
    rows = [["Mã ngành", "Tên ngành"]] + [
        [f"74802{i:02d}", f"Chương trình đào tạo số {i} với mô tả"] for i in range(12)
    ]
    table = Table(rows, page_number=1)

    chunks = table_aware_chunk([page(tables=[table])], max_size=180, metadata=BASE_METADATA)

    assert len(chunks) > 1
    assert all("| Mã ngành | Tên ngành |" in chunk.content for chunk in chunks)
    assert all(chunk.metadata["table_part_count"] == len(chunks) for chunk in chunks)
    for row in rows[1:]:
        rendered = f"| {row[0]} | {row[1]} |"
        assert sum(rendered in chunk.content for chunk in chunks) == 1


def test_parent_child_mapping_is_bidirectional_and_not_orphaned() -> None:
    heading = Heading("IV. Đối tượng", 1, 1)
    body = "\n\n".join(f"Đoạn {index}: " + "nội dung " * 12 for index in range(8))
    hierarchy = parent_child_chunk(
        [page(f"{heading.text}\n{body}", headings=[heading])],
        parent_size=300,
        child_size=110,
        metadata=BASE_METADATA,
    )
    parents = {chunk.chunk_id: chunk for chunk in hierarchy if chunk.metadata["chunk_type"] == "parent"}
    children = [chunk for chunk in hierarchy if chunk.metadata["chunk_type"] == "child"]

    assert len(parents) > 1
    assert children
    assert all(child.metadata["parent_chunk_id"] in parents for child in children)
    for parent_id, parent in parents.items():
        expected = [child.chunk_id for child in children if child.metadata["parent_chunk_id"] == parent_id]
        assert parent.metadata["child_chunk_ids"] == expected
        assert parent.metadata["child_count"] == len(expected)


def test_parent_and_child_ids_are_unique_and_deterministic() -> None:
    pages = [page("Một hai ba bốn năm sáu bảy tám chín mười. " * 20)]
    first = parent_child_chunk(pages, parent_size=180, child_size=70, metadata=BASE_METADATA)
    second = parent_child_chunk(pages, parent_size=180, child_size=70, metadata=BASE_METADATA)

    first_ids = [chunk.chunk_id for chunk in first]
    assert len(first_ids) == len(set(first_ids))
    assert first_ids == [chunk.chunk_id for chunk in second]
    child_parent_pairs = [
        (chunk.chunk_id, chunk.metadata["parent_chunk_id"])
        for chunk in first
        if chunk.metadata["chunk_type"] == "child"
    ]
    assert len(child_parent_pairs) == len(set(child_parent_pairs))


def test_combined_returns_only_children_and_tables_and_no_duplicate_content() -> None:
    heading = Heading("V. Chỉ tiêu", 1, 1)
    table = Table([["Ngành", "Chỉ tiêu"], ["CNTT", "500"]], 1)
    markdown_table = "| Ngành | Chỉ tiêu |\n| --- | --- |\n| CNTT | 500 |"
    source = page(
        f"{heading.text}\nNội dung mô tả.\n\n{markdown_table}",
        headings=[heading],
        tables=[table],
    )
    parents = []

    chunks = combined_chunk([source], metadata=BASE_METADATA, parent_chunks=parents)

    assert parents
    assert {chunk.metadata["chunk_type"] for chunk in chunks} <= {"child", "table"}
    normalized = [" ".join(chunk.content.split()).casefold() for chunk in chunks]
    assert len(normalized) == len(set(normalized))
    text_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "child"]
    assert all("| CNTT | 500 |" not in chunk.content for chunk in text_chunks)


def test_combined_metadata_and_parent_registry_are_consistent() -> None:
    heading = Heading("VI. Học bổng", 1, 7)
    source = page(
        f"{heading.text}\nHọc bổng dành cho thí sinh xuất sắc.",
        number=7,
        headings=[heading],
    )
    parents = []
    chunks = combined_chunk([source], metadata=BASE_METADATA, parent_chunks=parents)
    parent_ids = {parent.chunk_id for parent in parents}

    assert chunks
    for chunk in chunks:
        assert chunk.metadata["doc_id"] == BASE_METADATA["doc_id"]
        assert chunk.metadata["source"] == BASE_METADATA["source"]
        assert chunk.metadata["source_file"] == BASE_METADATA["source_file"]
        assert chunk.metadata["source_url"] == BASE_METADATA["source_url"]
        assert chunk.metadata["page_number"] == 7
        assert chunk.metadata["heading"] == heading.text
        assert chunk.metadata["parent_chunk_id"] in parent_ids


@pytest.mark.parametrize(
    "strategy",
    [structure_aware_chunk, table_aware_chunk, parent_child_chunk, combined_chunk],
)
def test_empty_page_produces_no_chunks(strategy) -> None:
    assert strategy([page()], metadata=BASE_METADATA) == []


def test_page_with_only_a_table_remains_retrievable() -> None:
    table = Table([["Cơ sở", "Chỉ tiêu"], ["Hà Nội", "1000"]], 5)
    parents = []

    chunks = combined_chunk(
        [page(number=5, tables=[table])],
        metadata=BASE_METADATA,
        parent_chunks=parents,
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_type"] == "table"
    assert chunks[0].metadata["parent_chunk_id"] == parents[0].chunk_id
    assert chunks[0].chunk_id in parents[0].metadata["child_chunk_ids"]


def test_page_with_only_a_heading_produces_context_chunk() -> None:
    heading = Heading("VII. Điều khoản thi hành", 1, 9)
    source = page(heading.text, number=9, headings=[heading])

    structured = structure_aware_chunk([source], metadata=BASE_METADATA)
    parents = []
    combined = combined_chunk([source], metadata=BASE_METADATA, parent_chunks=parents)

    assert [chunk.content for chunk in structured] == [heading.text]
    assert [chunk.content for chunk in combined] == [heading.text]
    assert combined[0].metadata["parent_chunk_id"] == parents[0].chunk_id


def test_detected_heading_absent_from_normalized_page_text_is_not_lost() -> None:
    heading = Heading("IX. Hồ sơ đăng ký", 1, 3)

    chunks = structure_aware_chunk(
        [page("Nội dung đã được parser chuẩn hóa.", number=3, headings=[heading])],
        metadata=BASE_METADATA,
    )

    assert chunks[0].content.startswith(heading.text)
    assert chunks[0].metadata["heading"] == heading.text


def test_vietnamese_unicode_is_preserved() -> None:
    heading = Heading("Thông tin tuyển sinh", 1, 1)
    content = "Thí sinh được xét tuyển thẳng nếu đáp ứng điều kiện. 🇻🇳"

    chunks = structure_aware_chunk(
        [page(f"{heading.text}\n{content}", headings=[heading])],
        max_size=90,
        metadata=BASE_METADATA,
    )

    assert content in " ".join(chunk.content for chunk in chunks)
    assert all("�" not in chunk.content for chunk in chunks)


def test_structure_chunk_emits_final_remainder_without_overlap() -> None:
    words = [f"mục{i}" for i in range(30)]
    chunks = structure_aware_chunk([page(" ".join(words))], max_size=45)

    assert len(chunks) > 1
    observed = Counter(word for chunk in chunks for word in chunk.content.split())
    assert observed == Counter(words)
    assert words[-1] in chunks[-1].content


def test_chunk_boundary_uses_available_max_size() -> None:
    exact = "x" * 80
    near = structure_aware_chunk([page(exact)], max_size=80)
    oversized = structure_aware_chunk([page(exact + "y")], max_size=80)

    assert [len(chunk.content) for chunk in near] == [80]
    assert [len(chunk.content) for chunk in oversized] == [80, 1]
    assert all(len(chunk.content) <= 80 for chunk in oversized)


def test_combined_ids_are_deterministic_for_children_tables_and_parents() -> None:
    heading = Heading("VIII. Mã ngành", 1, 2)
    table = Table([["Mã", "Tên"], ["7480201", "Công nghệ thông tin"]], 2)
    pages = [page(f"{heading.text}\nThông tin mô tả.", number=2, headings=[heading], tables=[table])]
    parents_one: list = []
    parents_two: list = []

    first = combined_chunk(pages, metadata=BASE_METADATA, parent_chunks=parents_one)
    second = combined_chunk(pages, metadata=BASE_METADATA, parent_chunks=parents_two)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.chunk_id for chunk in parents_one] == [chunk.chunk_id for chunk in parents_two]


def test_size_validation_rejects_invalid_advanced_configuration() -> None:
    with pytest.raises(ValueError, match="max_size"):
        structure_aware_chunk([page("text")], max_size=0)
    with pytest.raises(ValueError, match="child_size"):
        parent_child_chunk([page("text")], parent_size=100, child_size=101)
