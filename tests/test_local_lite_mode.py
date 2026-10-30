from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from api.routers import chat
from api.schemas import ChatResponse
from src.config import Settings, settings
from src.generation.rag_chain import FALLBACK_ANSWER, RAGChain
from src.ingestion.chunker import Chunk
from src.retrieval.lite_selector import LiteEvidenceSelector
from src.retrieval.domain_signals import QueryScope, assess, expansion_allowed, years, source_campuses


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE_A_CASE_IDS = (
    "admissions_sat_act_threshold",
    "cutoff_infosec_compare_campuses",
    "tuition_compare_2026",
    "negative_veterinary",
)


class StaticRetriever:
    def __init__(self, candidates):
        self.candidates = candidates

    def search(self, query, top_k):
        return self.candidates[:top_k]


class StaticLLM:
    def __init__(self, answer="Thông tin được hỗ trợ [1]."):
        self.answer_text = answer
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.answer_text

    def generate_stream(self, **kwargs):
        self.calls.append(kwargs)
        midpoint = max(1, len(self.answer_text) // 2)
        yield self.answer_text[:midpoint]
        yield self.answer_text[midpoint:]


@pytest.fixture
def isolated_chat_state():
    names = ("rag_chain", "rag_manifest", "rag_initialization_error", "rag_public_error")
    original = {name: getattr(chat, name) for name in names}
    try:
        for name in names:
            setattr(chat, name, None)
        yield
    finally:
        for name, value in original.items():
            setattr(chat, name, value)


def _manifest():
    return {
        "embedding_model": settings.EMBEDDING_MODEL,
        "document_count": 1,
        "chunk_count": 1,
        "backend": "test",
    }


def _fake_reranker_module(factory) -> ModuleType:
    module = ModuleType("src.retrieval.reranker")
    module.Reranker = factory
    return module


def test_runtime_defaults_select_single_lite_profile_without_query_rewrite():
    assert Settings.model_fields["LOCAL_DEV_PROFILE"].default is True
    assert Settings.model_fields["MODELS_LOCAL_ONLY"].default is True
    assert Settings.model_fields["RERANK_BACKEND"].default == "lite"
    assert Settings.model_fields["QUERY_REWRITE_ENABLED"].default is False
    with pytest.raises(ValidationError, match="RERANK_BACKEND"):
        Settings(_env_file=None, RERANK_BACKEND="automatic")


def test_lite_initialization_never_constructs_neural_reranker(
    monkeypatch, isolated_chat_state,
):
    neural_factory = MagicMock(side_effect=AssertionError("neural reranker loaded"))
    monkeypatch.setitem(
        sys.modules,
        "src.retrieval.reranker",
        _fake_reranker_module(neural_factory),
    )
    monkeypatch.setattr(settings, "RERANK_BACKEND", "lite")
    monkeypatch.setattr(settings, "QUERY_REWRITE_ENABLED", False)

    with patch.object(chat, "LLMClient"):
        chat.initialize_rag(retriever=StaticRetriever([]), manifest=_manifest())

    assert isinstance(chat.rag_chain.reranker, LiteEvidenceSelector)
    assert chat.rag_chain.query_rewriter is None
    neural_factory.assert_not_called()


def test_explicit_neural_backend_opts_out_of_local_profile(monkeypatch, isolated_chat_state):
    sentinel = object()
    neural_factory = MagicMock(return_value=sentinel)
    monkeypatch.setitem(
        sys.modules,
        "src.retrieval.reranker",
        _fake_reranker_module(neural_factory),
    )
    monkeypatch.setattr(settings, "RERANK_BACKEND", "neural")
    monkeypatch.setattr(settings, "LOCAL_DEV_PROFILE", False)
    monkeypatch.setattr(settings, "QUERY_REWRITE_ENABLED", True)

    with patch.object(chat, "LLMClient"):
        chat.initialize_rag(retriever=StaticRetriever([]), manifest=_manifest())

    assert chat.rag_chain.reranker is sentinel
    assert chat.rag_chain.query_rewriter is not None
    neural_factory.assert_called_once_with()


def test_local_profile_cannot_construct_neural_selector_from_stale_backend(monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_DEV_PROFILE", True)
    monkeypatch.setattr(settings, "RERANK_BACKEND", "neural")
    assert isinstance(chat._build_evidence_selector(), LiteEvidenceSelector)


def test_lite_selector_preserves_identity_scores_provenance_and_bound():
    candidates = [
        (
            Chunk(
                "noise",
                "Thông tin liên hệ chung.",
                {"doc_id": "noise-doc", "provenance": {"bm25": 1}},
            ),
            0.95,
        ),
        (
            Chunk(
                "sat",
                "Điểm SAT tối thiểu là 1130/1600.",
                {"doc_id": "admissions", "provenance": {"dense": 2}},
            ),
            0.60,
        ),
        (
            Chunk(
                "act",
                "Điểm ACT tối thiểu là 25/36.",
                {"doc_id": "admissions", "provenance": {"dense": 3}},
            ),
            0.55,
        ),
    ]
    metadata = [chunk.metadata for chunk, _ in candidates]

    selected = LiteEvidenceSelector().rerank(
        "Điểm SAT hoặc ACT tối thiểu là bao nhiêu?", candidates, top_k=2
    )

    assert [chunk.chunk_id for chunk, _, _ in selected] == ["sat", "act"]
    assert [score for _, score, _ in selected] == [0.60, 0.55]
    assert selected[0][0] is candidates[1][0]
    assert selected[1][0] is candidates[2][0]
    assert [chunk.metadata for chunk, _ in candidates] == metadata


def test_method_overview_seeds_cover_definitions_without_injecting_evidence():
    metadata = {"admission_year": "2026"}
    footer = Chunk("footer", "Các phương thức tuyển sinh 2026: xem hướng dẫn tại website.", metadata)
    intro = Chunk("overview", "PTIT sử dụng 05 phương thức tuyển sinh năm 2026. Phương thức 1: Xét tuyển tài năng", metadata)
    definitions = [Chunk(str(i), f"Phương thức {i}: Xét tuyển theo điều kiện {i}.", metadata) for i in range(2, 6)]
    repeated = [Chunk(f"repeat-{i}", "Phương thức 1: Xét tuyển tài năng", metadata) for i in range(8)]
    wrong_year = Chunk("old", "Phương thức 6: Xét tuyển theo kỳ thi khác", {"admission_year": "2025"})
    candidates = [(chunk, .04 - i * .001) for i, chunk in enumerate([footer, *repeated, intro, *definitions, wrong_year])]
    selector = LiteEvidenceSelector()
    query = "PTIT có những phương thức tuyển sinh nào trong năm 2026?"
    ranked = selector.rerank(query, candidates, top_k=len(candidates))
    selected = selector.select_seeds(query, ranked, 5)
    assert [chunk for chunk, _, _ in selected] == [intro, *definitions]
    assert all(any(chunk is original and score == original_score for original, original_score in candidates)
               for chunk, score, _ in selected)
    assert selector.select_seeds("Điểm SAT theo các phương thức là bao nhiêu?", ranked, 5) == ranked[:5]
    assert selector.select_seeds(query, ranked, 0) == []


def test_lite_json_and_sse_citations_are_consistent(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_CANDIDATE_K", 8)
    monkeypatch.setattr(settings, "RERANK_TOP_K", 5)
    chunk = Chunk(
        "sat-evidence",
        "Điểm SAT tối thiểu để xét tuyển là 1130/1600.",
        {
            "doc_id": "admissions-2026",
            "title": "Tuyển sinh 2026",
            "source_url": "https://example.edu/admissions",
            "page_number": 1,
            "provenance": {"dense_rank": 1, "bm25_rank": 2},
        },
    )
    llm = StaticLLM("Điểm SAT tối thiểu là 1130/1600 [1].")
    chain = RAGChain(
        StaticRetriever([(chunk, 0.73)]),
        llm,
        min_lexical_coverage=0.5,
        reranker=LiteEvidenceSelector(),
    )

    json_result = chain.answer("Điểm SAT tối thiểu là bao nhiêu?")
    events = list(chain.answer_stream("Điểm SAT tối thiểu là bao nhiêu?"))
    stream_metadata = next(event for event in events if event["type"] == "metadata")
    serialized_stream = chat._stream_item(stream_metadata)
    serialized_json = ChatResponse(
        answer=json_result.answer,
        citations=json_result.citations,
        route_type=json_result.route_type,
        session_id="test-session",
    ).model_dump()

    assert serialized_stream["citations"] == serialized_json["citations"]
    assert serialized_stream["route_type"] == serialized_json["route_type"]
    citation = serialized_json["citations"][0]
    assert citation["marker"] == 1
    assert citation["chunk_id"] == chunk.chunk_id
    assert citation["doc_id"] == chunk.metadata["doc_id"]
    assert citation["source_url"] == chunk.metadata["source_url"]
    assert citation["score"] == 0.73


def test_lite_sse_uses_one_nonstream_provider_call_then_validated_events():
    chunk = Chunk("sat", "Điểm SAT tối thiểu 1130/1600.", {"doc_id": "d"})
    llm = MagicMock()
    llm.generate.return_value = "Điểm SAT tối thiểu 1130/1600 [1]."
    llm.generate_stream.side_effect = AssertionError("Provider streaming must not be used by Lite")
    chain = RAGChain(StaticRetriever([(chunk, .02)]), llm, min_lexical_coverage=0,
                     reranker=LiteEvidenceSelector())
    events = list(chain.answer_stream("Điểm SAT tối thiểu?"))
    assert [event["type"] for event in events] == ["chunk", "metadata"]
    assert events[0]["text"] == "Điểm SAT tối thiểu 1130/1600."
    assert events[1]["citations"][0]["chunk_id"] == "sat"
    assert events[1]["citations"][0]["marker"] == 1
    llm.generate.assert_called_once()
    llm.generate_stream.assert_not_called()


def test_lite_negative_case_abstains_without_generation():
    llm = StaticLLM("Không được gọi.")
    chain = RAGChain(
        StaticRetriever(
            [(Chunk("tuition", "Học phí ngành Công nghệ thông tin."), 0.90)]
        ),
        llm,
        min_lexical_coverage=0.5,
        reranker=LiteEvidenceSelector(),
    )

    result = chain.answer("PTIT có tuyển ngành Thú y năm 2026 không?")

    assert result.answer == FALLBACK_ANSWER
    assert result.citations == []
    assert result.route_type == "out_of_scope"
    assert llm.calls == []


def test_lite_matches_four_phase_a_golden_case_contracts(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_CANDIDATE_K", 8)
    monkeypatch.setattr(settings, "RERANK_TOP_K", 5)
    dataset = json.loads(
        (PROJECT_ROOT / "eval" / "golden_dataset.json").read_text(encoding="utf-8")
    )
    evidence = dataset["evidence_catalog"]
    cases = {case["id"]: case for case in dataset["cases"]}
    assert set(PHASE_A_CASE_IDS) <= cases.keys()

    for case_id in PHASE_A_CASE_IDS:
        case = cases[case_id]
        candidates = [
            (
                Chunk(
                    evidence_id,
                    evidence[evidence_id]["quote"],
                    {
                        "doc_id": evidence[evidence_id]["doc_id"],
                        "title": evidence[evidence_id]["title"],
                        "source_url": evidence[evidence_id]["source_url"],
                        "page_number": evidence[evidence_id]["page"],
                        "provenance": {"golden": True},
                    },
                ),
                0.60 - (position * 0.01),
            )
            for position, evidence_id in enumerate(case["evidence_ids"])
        ]
        candidates.append(
            (
                Chunk(
                    f"distractor-{case_id}",
                    "Thông tin học phí ngành Công nghệ thông tin.",
                    {"doc_id": "distractor", "provenance": {"golden": False}},
                ),
                0.99,
            )
        )
        if case["expected_route"] == "out_of_scope":
            chain = RAGChain(
                StaticRetriever(candidates),
                StaticLLM(),
                min_lexical_coverage=0.5,
                reranker=LiteEvidenceSelector(),
            )
            prepared = chain._prepare_generation(case["question"], None)
            assert prepared is None, case_id
            continue
        selected = LiteEvidenceSelector().rerank(
            case["question"], candidates[: settings.RERANK_CANDIDATE_K],
            top_k=settings.RERANK_TOP_K,
        )
        selected_ids = {chunk.chunk_id for chunk, _, _ in selected}
        assert set(case["evidence_ids"]) <= selected_ids, case_id
        assert len(selected) <= settings.RERANK_TOP_K


@pytest.mark.parametrize("field,value", [
    ("LITE_DENSE_WEIGHT", -1), ("LITE_SPARSE_WEIGHT", float("nan")),
    ("LITE_RRF_K", float("inf")), ("LITE_DOMAIN_MIN_FACTOR", 0),
])
def test_lite_settings_validate_finite_positive_bounds(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_lite_settings_validate_joint_constraints():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, LITE_DENSE_WEIGHT=0, LITE_SPARSE_WEIGHT=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, LITE_DOMAIN_MIN_FACTOR=5, LITE_DOMAIN_MAX_FACTOR=4)


@pytest.mark.parametrize("code,other", [("CN1", "CN10"), ("A00", "A001"), ("7480202", "17480202")])
def test_entity_boost_uses_exact_tokens(code, other):
    selector = LiteEvidenceSelector()
    exact = Chunk("z", f"Mã {code}")
    partial = Chunk("a", f"Mã {other}")
    assert ("exact_entity_code", 1.8) in selector.explain(f"Mã {code}", exact).factors
    assert not any(name == "exact_entity_code" for name, _ in selector.explain(f"Mã {code}", partial).factors)
    assert selector.rerank(f"Mã {code}", [(partial, .02), (exact, .02)], 1)[0][0] is exact


@pytest.mark.parametrize("query,metadata,eligible,ambiguity", [
    ("Tuyển sinh 2031", {"admission_year": "2030", "academic_year": "2030-2031"}, False, None),
    ("Tuyển sinh 2031", {"admission_year": [2030, 2031]}, True, None),
    ("Học phí 2031-2032", {"academic_year": "2030-2031"}, False, None),
    ("Học phí 2031-2032", {"academic_year": "2031-2032", "admission_year": "2030"}, True, None),
    ("Học phí 2031-2032", {"admission_year": "2032"}, False, None),
    ("Học phí 2031-2032", {"admission_year": "2031"}, True, "academic_range_unconfirmed"),
    ("Tuyển sinh 2031", {"retrieved_at": "2031-01-01"}, True, "year_missing"),
    ("Điểm chuẩn BVH", {"campus_code": "BVS"}, False, None),
    ("Điểm chuẩn Hà Nội", {"campus": "BVH"}, True, None),
    ("Điểm chuẩn TP.HCM", {"campus": "BVH"}, False, None),
    ("Điểm chuẩn BVH", {}, False, "campus_missing"),
])
def test_explicit_year_campus_scopes(query, metadata, eligible, ambiguity):
    decision = assess(QueryScope.parse(query), Chunk("e", "", metadata))
    assert decision.eligible is eligible
    if ambiguity:
        assert ambiguity in decision.ambiguities


def test_year_modes_and_local_campus_heading_precedence():
    assert years("2031").kind == "single"
    assert years("2031 và 2032").kind == "multi"
    assert years("2031–2032").kind == "academic_range"
    assert years("").values == frozenset()
    assert source_campuses(Chunk("x", "", {"title": "BVH và BVS", "heading": "Cơ sở Hà Nội"})) == {"BVH"}


def test_section_demotion_uses_heading_intent_not_incidental_body_word():
    selector = LiteEvidenceSelector()
    query = "Điểm chuẩn CN1"
    incidental = Chunk("i", "Chương trình đào tạo có học phí.", {"heading": "Điểm chuẩn"})
    curriculum = Chunk("c", "CN1", {"heading": "Khung chương trình đào tạo"})
    assert not any(name == "curriculum_section" for name, _ in selector.explain(query, incidental).factors)
    assert ("curriculum_section", .5) in selector.explain(query, curriculum).factors


def test_domain_score_is_bounded_deterministic_and_excludes_wrong_scope():
    selector = LiteEvidenceSelector()
    matching = Chunk("ok", "Điểm chuẩn CN1 BVH 2031", {"admission_year": "2031", "campus": "BVH", "heading": "Điểm chuẩn"})
    wrong = Chunk("wrong", matching.content, {**matching.metadata, "campus": "BVS"})
    candidates = [(wrong, .5), (matching, .02)]
    selected = selector.rerank("Điểm chuẩn CN1 BVH 2031", candidates, 5)
    assert len(selected) == 1
    assert .02 * settings.LITE_DOMAIN_MIN_FACTOR <= selected[0][2] <= .02 * settings.LITE_DOMAIN_MAX_FACTOR
    assert selected == selector.rerank("Điểm chuẩn CN1 BVH 2031", candidates, 5)


@pytest.mark.parametrize("changed", [
    {"doc_id": "other"}, {"admission_year": "2030"}, {"academic_year": "2030-2031"},
    {"campus": "BVS"}, {"campus": None}, {"source_url": "other"}, {"version": "2"},
])
def test_expansion_rejects_cross_scope_or_source_identity(changed):
    metadata = {"doc_id": "d", "admission_year": "2031", "academic_year": "2031-2032", "campus": "BVH", "source_url": "s", "version": "1"}
    seed = Chunk("seed", "", metadata)
    other = Chunk("other", "", {**metadata, **changed})
    scope = QueryScope.parse("Điểm chuẩn BVH 2031")
    assert expansion_allowed(scope, seed, seed)
    assert not expansion_allowed(scope, seed, other)


@pytest.mark.parametrize("program", ["Thú y", "Nông học", "Dược học"])
def test_lite_abstains_on_named_program_despite_generic_admission_overlap(program):
    llm = StaticLLM()
    candidate = Chunk("generic", "PTIT có tuyển ngành Công nghệ thông tin năm 2031. Tuyển sinh đại học.", {"admission_year": "2031"})
    chain = RAGChain(StaticRetriever([(candidate, .02)]), llm, min_lexical_coverage=0, reranker=LiteEvidenceSelector())
    assert chain.answer(f"PTIT có tuyển ngành {program} năm 2031 không?").route_type == "out_of_scope"
    assert llm.calls == []


def test_lite_supported_program_existence_keeps_evidence():
    candidate = Chunk("supported", "PTIT có tuyển ngành Công nghệ thông tin năm 2031.", {"admission_year": "2031"})
    chain = RAGChain(StaticRetriever([(candidate, .02)]), StaticLLM(), min_lexical_coverage=0, reranker=LiteEvidenceSelector())
    assert chain._prepare_generation("PTIT có tuyển ngành Công nghệ thông tin năm 2031 không?", None) is not None


def test_campus_is_part_of_provenance_deduplication():
    from src.generation.assembler import ContextAssembler
    chunks = [(Chunk(campus, "Điểm chuẩn 27.5", {"doc_id": "d", "section": "Điểm chuẩn", "campus": campus}), .02) for campus in ("BVH", "BVS")]
    assert len(ContextAssembler().assemble(chunks)) == 2
    assert len(RAGChain._select_citation_sources("Điểm chuẩn", chunks)) == 2


def test_ambiguous_scope_cannot_authorize_expansion():
    seed = Chunk("seed", "", {"doc_id": "d"})
    neighbor = Chunk("neighbor", "", {"doc_id": "d"})
    assert not expansion_allowed(QueryScope.parse("Học phí 2031"), seed, neighbor)
    assert not expansion_allowed(QueryScope.parse("Học phí BVH"), seed, neighbor)


def test_unified_profile_overrides_stale_pair_without_changing_embedding_identity():
    configured = Settings(_env_file=None, LOCAL_DEV_PROFILE=True, INDEX_DIR="obsolete-index", QDRANT_COLLECTION="obsolete-collection", RERANK_BACKEND="neural", QUERY_REWRITE_ENABLED=True)
    assert configured.INDEX_DIR == "data/index/module3-phase-a-20261019-223230"
    assert configured.QDRANT_COLLECTION == "admitai_chunks_phase_a_20261019_223230"
    assert configured.RERANK_BACKEND == "lite"
    assert configured.QUERY_REWRITE_ENABLED is False
    assert configured.MODELS_LOCAL_ONLY is True
    assert configured.EMBEDDING_MODEL == "BAAI/bge-m3"


def test_missing_cached_model_never_falls_back_to_download_in_local_profile(monkeypatch):
    from src.retrieval.embedder import Embedder
    from src.runtime_resources import StartupCheckError
    monkeypatch.setattr(settings, "MODELS_LOCAL_ONLY", True)
    with (
        patch("src.retrieval.embedder.snapshot_download", return_value="cached-test-model"),
        patch("src.retrieval.embedder.SentenceTransformer", side_effect=OSError("no cached model")) as factory,
    ):
        with pytest.raises(StartupCheckError) as caught:
            Embedder("local-only-test-model")
    assert caught.value.code == "MODEL_UNAVAILABLE"
    assert factory.call_count == 1
    assert factory.call_args.kwargs["local_files_only"] is True


def test_live_availability_abstains_before_retrieval_or_generation():
    retriever, llm = MagicMock(), StaticLLM()
    chain = RAGChain(retriever, llm, reranker=LiteEvidenceSelector())
    question = "Hiện giờ ký túc xá PTIT còn chính xác bao nhiêu phòng trống?"
    assert chain.answer(question).route_type == "out_of_scope"
    assert list(chain.answer_stream(question))[0]["route_type"] == "out_of_scope"
    retriever.search.assert_not_called()
    assert llm.calls == []


def test_required_golden_quotes_keep_json_sse_source_page_snippet_mapping():
    dataset = json.loads((PROJECT_ROOT / "eval/golden_dataset.json").read_text(encoding="utf-8"))
    for case in dataset["cases"]:
        if case["id"] not in PHASE_A_CASE_IDS or not case["evidence_ids"]:
            continue
        records = [dataset["evidence_catalog"][eid] for eid in case["evidence_ids"]]
        candidates = [(Chunk(record["chunk_id"], record["quote"], {"doc_id": record["doc_id"], "title": record["title"], "source_url": record["source_url"], "page_number": record["page"]}), .02) for record in records]
        selected = LiteEvidenceSelector().rerank(case["question"], candidates, top_k=5)
        answer = "\n".join(f"{chunk.content} [{index}]." for index, (chunk, _, _) in enumerate(selected, 1))
        chain = RAGChain(StaticRetriever(candidates), StaticLLM(answer), min_lexical_coverage=0, reranker=LiteEvidenceSelector())
        sync = chain.answer(case["question"])
        events = list(chain.answer_stream(case["question"]))
        metadata = next(event for event in events if event["type"] == "metadata")
        assert sync.citations == metadata["citations"]
        assert "".join(event["text"] for event in events if event["type"] == "chunk") == sync.answer
        for marker, citation in enumerate(sync.citations, 1):
            expected = selected[marker - 1][0]
            assert citation["marker"] == marker
            assert citation["chunk_id"] == expected.chunk_id
            assert citation["page"] == expected.metadata["page_number"]
            assert citation["source_url"] == expected.metadata["source_url"]
            assert citation["snippet"]


def test_lite_context_budget_counts_source_metadata_and_wrappers(monkeypatch):
    monkeypatch.setattr(settings, "LITE_CONTEXT_MAX_CHARS", 1000)
    candidate = Chunk("sat", "SAT 1130", {"title": "Very long title " * 200})
    chain = RAGChain(StaticRetriever([(candidate, .02)]), StaticLLM(), min_lexical_coverage=0, reranker=LiteEvidenceSelector())
    assert chain._prepare_generation("SAT", None) is None
