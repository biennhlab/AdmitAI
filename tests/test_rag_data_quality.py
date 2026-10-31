"""Regressions for source fidelity and incomplete RAG index publication."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import numpy as np
import pytest

from scripts import ingest
from scripts.audit_rag_data import audit_documents
from src.ingestion import Chunk, combined_chunk, load_documents, parse_markdown_document, structure_aware_chunk
from src.ingestion.markdown import parse_row


def source(tmp_path: Path, body: str, *, doc_id: str = "test-doc", extra: str = ""):
    path = tmp_path / f"{doc_id}.md"
    path.write_text(f'---\ndoc_id: "{doc_id}"\ntitle: "Test evidence"\n{extra}---\n{body}', encoding="utf-8")
    return parse_markdown_document(path)


def test_frontmatter_delimiters_inside_values_and_bom_are_safe(tmp_path):
    path = tmp_path / "bom.md"
    path.write_text('\ufeff---\ndoc_id: "bom"\ntitle: "Before---after"\n---\nActual evidence.', encoding="utf-8")
    document = parse_markdown_document(path)
    assert document.metadata["title"] == "Before---after"
    assert document.content == "Actual evidence."


def test_content_edit_invalidates_fingerprint_even_with_stale_frontmatter_hash(tmp_path):
    first = source(tmp_path, "Tuition 2025: 30 million.", extra='content_hash: "stale"\n')
    second = source(tmp_path, "Tuition 2025: 40 million.", extra='content_hash: "stale"\n')
    assert first.metadata["content_hash"] != second.metadata["content_hash"]
    assert ingest.corpus_fingerprint([first], 3) != ingest.corpus_fingerprint([second], 3)


@pytest.mark.parametrize("target", ["missing", "canonical"])
def test_duplicate_marker_never_discards_unique_or_unresolvable_evidence(tmp_path, target):
    source(tmp_path, "Original evidence.", doc_id="canonical")
    source(tmp_path, "Additional admission condition.", doc_id="alias", extra=f'duplicate_of: "{target}"\n')
    loaded = load_documents(tmp_path)
    assert len(loaded.documents) == 2
    assert loaded.skipped_duplicates == []
    assert loaded.warnings[0]["code"] == "invalid_duplicate_reference"


def test_duplicate_cycles_preserve_evidence(tmp_path):
    source(tmp_path, "Same evidence.", doc_id="a", extra='duplicate_of: "b"\n')
    source(tmp_path, "Same evidence.", doc_id="b", extra='duplicate_of: "a"\n')
    assert len(load_documents(tmp_path).documents) == 2


def test_duplicate_text_keeps_distinct_source_provenance(tmp_path):
    source(tmp_path, "Same evidence.", doc_id="a", extra='source_url: "https://example.test/a"\n')
    source(tmp_path, "Same evidence.", doc_id="b", extra='source_url: "https://example.test/b"\nduplicate_of: "a"\n')
    loaded = load_documents(tmp_path)
    assert len(loaded.documents) == 1
    assert loaded.documents[0].metadata["duplicate_sources"][0]["source_url"] == "https://example.test/b"


def test_heading_mentioned_in_prose_is_not_used_as_section_start(tmp_path):
    document = source(tmp_path, "See Tuition below for details.\n\n## Tuition\nFee is 35 million.")
    chunks = structure_aware_chunk(document.pages)
    assert chunks[0].content == "See Tuition below for details."
    assert chunks[1].content == "Tuition\n\nFee is 35 million."


def test_table_removal_preserves_identical_prose_and_offsets(tmp_path):
    document = source(tmp_path, "## Fees\n\nProgramme Fee\n\n| Programme | Fee |\n| --- | --- |\n| CNTT | 35 |\n\n## Eligibility\nRequires graduation.")
    chunks = combined_chunk(document.pages, metadata=document.metadata)
    text = "\n".join(chunk.content for chunk in chunks if chunk.metadata["chunk_type"] == "child")
    assert "Programme Fee" in text
    assert "| CNTT |" not in text
    assert any(chunk.metadata["heading"] == "Eligibility" and "Requires graduation." in chunk.content for chunk in chunks)


def test_text_and_table_indexes_never_overwrite_each_other_in_neighbor_map(tmp_path):
    from src.retrieval import BM25Search, HybridRetriever
    document = source(tmp_path, "## Admissions\n\nConditions apply.\n\n| Programme | Quota |\n| --- | --- |\n| CNTT | 500 |")
    chunks = combined_chunk(document.pages, metadata=document.metadata)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    sparse = BM25Search()
    sparse.index(chunks)
    hybrid = HybridRetriever(SimpleNamespace(search=lambda *args, **kwargs: []), sparse)
    assert len(hybrid.chunk_by_doc_and_index) == len(chunks)


def test_markdown_tables_without_outer_pipes_and_fenced_examples(tmp_path):
    document = source(tmp_path, "```md\n# Example heading\nA | B\n--- | ---\n1 | 2\n```\n\n## Real\nA | B\n--- | ---\n3 | 4")
    assert [heading.text for heading in document.pages[0].headings] == ["Real"]
    assert len(document.pages[0].tables) == 1
    assert document.pages[0].tables[0].rows[-1] == ["3", "4"]


def test_multitier_headers_keep_score_methods_and_scales_in_every_part(tmp_path):
    body = ("## North BVH\n\n| TT | Programme | Cutoff | Cutoff |\n| --- | --- | --- | --- |\n"
            "| TT | Programme | SAT (1600) | HSA (150) |\n" +
            "\n".join(f"| {index} | Programme {index} | 1300 | 95 |" for index in range(8)))
    document = source(tmp_path, body)
    parents = []
    chunks = combined_chunk(document.pages, max_size=180, metadata=document.metadata, parent_chunks=parents)
    tables = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "table"]
    assert len(tables) > 1
    assert all("SAT (1600)" in chunk.content and "HSA (150)" in chunk.content for chunk in tables)
    assert all(chunk.metadata["header_row_count"] == 2 for chunk in tables)
    assert audit_documents([document], chunks, parents)["errors"] == []


def test_wide_tables_split_columns_without_losing_score_labels_or_cells(tmp_path):
    columns = ["TT", "Programme", "Code", *[f"Method {index} (scale 100)" for index in range(8)]]
    from src.ingestion.chunker import _markdown_row
    rows = [[str(index), f"Programme {index}", f"7480{index}", *[str(70 + value) for value in range(8)]] for index in range(3)]
    document = source(tmp_path, _markdown_row(columns) + "\n" + _markdown_row(["---"] * len(columns)) + "\n" + "\n".join(_markdown_row(row) for row in rows))
    parents = []
    chunks = combined_chunk(document.pages, max_size=240, metadata=document.metadata, parent_chunks=parents)
    assert len({tuple(chunk.metadata["column_indexes"]) for chunk in chunks}) > 1
    assert all("| TT | Programme | Code |" in chunk.content for chunk in chunks)
    assert all(len(chunk.content) <= 240 for chunk in chunks)
    assert audit_documents([document], chunks, parents)["errors"] == []


def test_table_before_first_heading_is_not_attached_to_later_section(tmp_path):
    document = source(tmp_path, "| Programme | Fee |\n| --- | --- |\n| CNTT | 35 |\n\n## Scholarships\nScholarship rules.")
    chunks = combined_chunk(document.pages, metadata=document.metadata)
    table = next(chunk for chunk in chunks if chunk.metadata["chunk_type"] == "table")
    assert table.metadata["heading"] != "Scholarships"


def test_pdf_table_without_position_does_not_invent_a_campus_heading():
    from src.ingestion import Heading, ParsedPage, Table, table_aware_chunk
    page = ParsedPage(1, "North BVH\nRules.\n\nSouth BVS\nOther rules.",
                      tables=[Table([["Programme", "Quota"], ["CNTT", "500"]], 1)],
                      headings=[Heading("North BVH", 2, 1), Heading("South BVS", 2, 1)])
    table = table_aware_chunk([page])[0]
    assert table.metadata["heading"] == ""


def test_table_context_keeps_full_hierarchy_and_caption(tmp_path):
    document = source(tmp_path, "# Admissions 2026\n\n## North BVH\n\nUnit: million VND/year\n\n| Programme | Fee |\n| --- | --- |\n| CNTT | 35 |")
    table = next(chunk for chunk in combined_chunk(document.pages) if chunk.metadata["chunk_type"] == "table")
    assert table.metadata["heading_path"] == ["Admissions 2026", "North BVH"]
    assert "Unit: million VND/year" in table.content


def test_marked_footnote_reaches_generation_without_polluting_table_embeddings(tmp_path):
    from src.generation.assembler import ContextAssembler
    document = source(tmp_path, "| Programme | Tie-break (*) |\n| --- | --- |\n| A | NV1 |\n\n**(\\*)** Prefer the earlier admission choice.")
    table = next(chunk for chunk in combined_chunk(document.pages) if chunk.metadata["chunk_type"] == "table")
    assert table.metadata["table_notes"]
    original = table.content
    assembled = ContextAssembler().assemble([(table, 1.)])
    assert "Prefer the earlier admission choice." in assembled[0][0].content
    assert table.content == original
    assert "Prefer the earlier admission choice." not in original


def test_table_notes_are_counted_against_context_budget():
    from src.generation.assembler import ContextAssembler
    table = Chunk("t", "| A | B |", {"chunk_type": "table", "table_notes": ["Long note " * 30]})
    assert ContextAssembler(max_chars=30).assemble([(table, 1.)]) == []


def test_repeated_table_notes_and_identical_hits_do_not_consume_context_twice():
    from src.generation.assembler import ContextAssembler
    note = "A common applicability note."
    first = Chunk("t1", "| A | 20 |", {"chunk_type": "table", "table_notes": [note]})
    second = Chunk("t2", "| B | 30 |", {"chunk_type": "table", "table_notes": [note]})
    context = ContextAssembler().assemble([(first, 1.), (first, .9), (second, .8)])
    assert len(context) == 2
    assert "\n".join(chunk.content for chunk, _score in context).count(note) == 1


def test_historical_faq_is_not_mislabeled_as_current_policy_and_keeps_pairs(tmp_path):
    from src.generation.assembler import ContextAssembler
    document = source(tmp_path, "## FAQ\n\nWhat are the conditions?\n\nRequires IELTS 6.5.\n\nWhat are the fees?\n\nFees depend on the programme.",
                      doc_id="cau-hoi-thuong-gap", extra='published_at: "2024-08-03"\nadmission_year: "2026"\nacademic_year: "2026-2027"\n')
    assert document.metadata["collection_admission_year"] == "2026"
    assert document.metadata["admission_year"] == ""
    assert document.metadata["academic_year"] == ""
    chunks = combined_chunk(document.pages, metadata=document.metadata)
    answer = next(chunk for chunk in chunks if "IELTS 6.5" in chunk.content)
    assert "What are the conditions?" in answer.content
    assert "What are the fees?" not in answer.content
    context = ContextAssembler().assemble([(answer, 1.)])
    assert document.metadata["applicability_note"] in context[0][0].content


def test_explicit_historical_faq_year_is_used_instead_of_collection_year(tmp_path):
    document = source(tmp_path, "## FAQ\nTHPT 2024 admission conditions apply.", doc_id="cau-hoi-thuong-gap",
                      extra='published_at: "2024-08-03"\nadmission_year: "2026"\n')
    assert document.metadata["admission_year"] == "2024"
    assert document.metadata["collection_admission_year"] == "2026"


def test_long_heading_never_produces_unbounded_children_or_drops_answer(tmp_path):
    heading = "Eligibility conditions " * 25
    document = source(tmp_path, f"## {heading}\n\nAn applicant needs IELTS 6.5 and SAT 1200.")
    parents = []
    chunks = combined_chunk(document.pages, child_size=90, parent_size=180, parent_chunks=parents, metadata=document.metadata)
    assert all(len(chunk.content) <= 90 for chunk in chunks)
    assert "IELTS 6.5" in " ".join(chunk.content for chunk in chunks)
    assert audit_documents([document], chunks, parents)["errors"] == []


def test_raw_html_fallback_keeps_headings_and_tables(tmp_path):
    path = tmp_path / "raw.html"
    path.write_text("<main><h2>Fees 2026</h2><p>Annual tuition applies to all incoming students under the conditions below.</p><table><tr><th>Programme</th><th>Fee</th></tr><tr><td>CNTT</td><td>35</td></tr></table></main>", encoding="utf-8")
    loaded = load_documents(tmp_path, prefer_processed=False)
    assert loaded.errors == []
    assert loaded.documents[0].pages[0].headings
    assert loaded.documents[0].pages[0].tables[0].rows[-1] == ["CNTT", "35"]


def test_media_sources_survive_collection_and_accordion_questions_become_headings():
    from scripts.collect_ptit_admissions import clean_content_html, html_to_md
    cleaned, links, _stats = clean_content_html('<div><span class="elementor-accordion-title">Who is eligible?</span><p>Graduates.</p><iframe src="/programme-video"></iframe><img src="/cutoff.png"></div>', 'https://example.test')
    markdown = html_to_md(cleaned)
    assert "### Who is eligible?" in markdown
    assert "https://example.test/programme-video" in links
    assert "https://example.test/cutoff.png" in markdown


def test_escaped_pipes_and_literal_backslashes_round_trip():
    from src.ingestion.chunker import _markdown_row
    row = ["A|B", "C:\\folder\\", "6.5"]
    assert parse_row(_markdown_row(row)) == row


@pytest.mark.parametrize("bad", [np.array([[np.nan, 1.]]), np.zeros((1, 2)), np.ones((2, 2)), np.ones((1, 0))])
def test_invalid_single_embedding_is_not_added_to_successful_rows(bad):
    chunks, vectors, errors = ingest.embed_chunks(SimpleNamespace(embed=lambda texts: bad), [Chunk("bad", "Evidence")])
    assert chunks == []
    assert vectors.shape == (0, 0)
    assert errors[0]["chunk_id"] == "bad"


def test_embedding_batch_failure_retries_without_misaligning_rows():
    def embed(texts):
        if len(texts) > 1:
            raise RuntimeError("batch failed")
        if texts[0] == "Bad":
            return np.full((1, 2), np.nan)
        return np.array([[len(texts[0]), 1.]])
    successful, vectors, errors = ingest.embed_chunks(SimpleNamespace(embed=embed), [Chunk("a", "Good"), Chunk("b", "Bad"), Chunk("c", "Longer")])
    assert [chunk.chunk_id for chunk in successful] == ["a", "c"]
    np.testing.assert_array_equal(vectors, [[4., 1.], [6., 1.]])
    assert errors[0]["chunk_id"] == "b"


def test_source_errors_abort_before_models_or_index_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ingest.py", "--module", "2", "--index-dir", str(tmp_path / "candidate")])
    monkeypatch.setattr(ingest, "load_documents", lambda *args, **kwargs: SimpleNamespace(errors=[{"file": "broken", "error": "bad"}]))
    factory = Mock(side_effect=AssertionError("model must not be loaded"))
    persist = Mock(side_effect=AssertionError("index must not be written"))
    monkeypatch.setattr(ingest, "Embedder", factory)
    monkeypatch.setattr(ingest, "save_dense_index", persist)
    with pytest.raises(RuntimeError, match="incomplete index"):
        ingest.main()
    factory.assert_not_called()
    persist.assert_not_called()


def test_audit_detects_missing_row_even_when_other_rows_repeat_the_same_numbers(tmp_path):
    document = source(tmp_path, "| Name | Score |\n| --- | --- |\n| A | 20 |\n| B | 20 |")
    parents = []
    chunks = combined_chunk(document.pages, metadata=document.metadata, parent_chunks=parents)
    table = next(chunk for chunk in chunks if chunk.metadata["chunk_type"] == "table")
    table.content = table.content.replace("| B | 20 |", "")
    assert any(error["code"] == "table_row_missing" for error in audit_documents([document], chunks, parents)["errors"])


def test_expanded_context_location_and_deadline_do_not_answer_an_unrelated_topic():
    from src.generation.rag_chain import RAGChain
    chain = RAGChain(None, None, min_lexical_coverage=.5)
    evidence = [(Chunk("admission", "Ngày mai nhận hồ sơ tuyển sinh tại Hà Nội."), 1.)]
    assert not chain._has_lexical_evidence("Ngày mai ở Hà Nội có mưa không?", evidence)
    assert chain._has_lexical_evidence("Ngày mai nhận hồ sơ ở Hà Nội?", evidence)
