import logging
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from src.generation.llm_client import LLMClient, LLMUpstreamError
from src.generation.prompts import build_rag_prompt, format_citations, SYSTEM_PROMPT
from src.generation.session_memory import SessionMemory
from src.generation.rag_chain import FALLBACK_ANSWER, RAGChain

class MockChunk:
    def __init__(self, chunk_id, content, metadata=None):
        self.chunk_id = chunk_id
        self.content = content
        self.metadata = metadata or {}

class MockRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, question, top_k=5):
        # Return a list of tuples (Chunk, score)
        return [(chunk, 0.9) for chunk in self.chunks[:top_k]]

def test_llm_client_generation():
    # Mock LLMClient
    with patch("src.generation.llm_client.OpenAI") as MockOpenAI:
        mock_client_instance = MagicMock()
        MockOpenAI.return_value = mock_client_instance
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is a mock answer."
        mock_client_instance.chat.completions.create.return_value = mock_response

        client = LLMClient(
            api_key="fake_key",
            model="fake-model",
            base_url="fake-url",
            reasoning_effort="minimal",
        )
        answer = client.generate("system prompt", [{"role": "user", "content": "hello"}])
        
        assert answer == "This is a mock answer."
        mock_client_instance.chat.completions.create.assert_called_once()
        generation_kwargs = mock_client_instance.chat.completions.create.call_args.kwargs
        assert generation_kwargs["max_tokens"] == 1024
        assert generation_kwargs["reasoning_effort"] == "minimal"

def test_llm_client_error():
    with patch("src.generation.llm_client.OpenAI") as MockOpenAI:
        mock_client_instance = MagicMock()
        MockOpenAI.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")

        client = LLMClient(api_key="fake", model="fake", base_url="fake-url")
        with pytest.raises(LLMUpstreamError, match="unexpected error"):
            client.generate("sys", [])


def test_llm_client_legacy_sdk_forwards_reasoning_effort_through_extra_body():
    captured = {}

    class LegacyCompletions:
        def create(
            self,
            *,
            messages,
            model,
            temperature,
            max_tokens,
            stream=False,
            extra_body=None,
        ):
            captured.update(
                {
                    "messages": messages,
                    "model": model,
                    "stream": stream,
                    "extra_body": extra_body,
                }
            )
            return [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="Xin chào"))]
                )
            ]

    legacy_client = SimpleNamespace(
        chat=SimpleNamespace(completions=LegacyCompletions())
    )
    with patch("src.generation.llm_client.OpenAI", return_value=legacy_client):
        client = LLMClient(
            api_key="fake",
            model="gemma-4-26b-a4b-it",
            base_url="fake-url",
            reasoning_effort="minimal",
        )
        chunks = list(client.generate_stream("system", [{"role": "user", "content": "hello"}]))

    assert chunks == ["Xin chào"]
    assert captured["stream"] is True
    assert captured["extra_body"] == {"reasoning_effort": "minimal"}

def test_build_rag_prompt_and_format_citations():
    chunk1 = MockChunk(1, "Text A", {"source": "doc1.pdf"})
    chunk2 = MockChunk(2, "Text B") # No source metadata, uses chunk_id
    
    citations = format_citations([chunk1, chunk2])
    assert "doc1.pdf" in citations
    assert "Text A" in citations
    assert "Chunk 2" in citations
    assert "Text B" in citations

    messages = build_rag_prompt([chunk1, chunk2], "What is A?")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "What is A?" in messages[0]["content"]
    assert "doc1.pdf" not in messages[0]["content"]
    assert "retrieved_context" not in messages[0]["content"]


def test_system_prompt_uses_sources_and_natural_markdown():
    assert "Chỉ dùng thông tin có trong <retrieved_context>" in SYSTEM_PROMPT
    assert "không suy đoán" in SYSTEM_PROMPT
    assert "Backend tạo marker [n]" in SYSTEM_PROMPT
    assert "Không tự tạo URL" in SYSTEM_PROMPT
    assert "Không nhắc hoặc giải thích các cấu trúc nội bộ" in SYSTEM_PROMPT
    for internal_term in (
        "retrieved_context",
        "document",
        "retrieval",
        "chunk",
        "pipeline",
        "system prompt",
        "dữ liệu truy xuất",
        "ngữ cảnh được cung cấp",
    ):
        assert internal_term in SYSTEM_PROMPT
    assert "Trả lời trực tiếp vào điều người dùng hỏi ngay ở câu đầu tiên" in SYSTEM_PROMPT
    assert "không lặp lại nguyên câu hỏi" in SYSTEM_PROMPT
    assert "Trả lời bằng tiếng Việt tự nhiên" in SYSTEM_PROMPT
    assert "Với câu hỏi đơn giản, ưu tiên 1–3 đoạn ngắn" in SYSTEM_PROMPT
    assert "Chỉ dùng heading khi câu trả lời thật sự có từ hai phần nội dung khác nhau" in SYSTEM_PROMPT
    assert "Không tạo heading chỉ để trang trí" in SYSTEM_PROMPT
    assert "Dùng bullet hoặc numbered list khi có nhiều mục cần liệt kê" in SYSTEM_PROMPT
    assert "bảng Markdown" in SYSTEM_PROMPT
    assert "**bold**" in SYSTEM_PROMPT
    assert "có chọn lọc" in SYSTEM_PROMPT
    assert "Chỉ dùng code block cho code hoặc dữ liệu kỹ thuật" in SYSTEM_PROMPT
    assert "không đặt nội dung hội thoại thông thường trong code block" in SYSTEM_PROMPT
    assert "Không dùng emoji mặc định" in SYSTEM_PROMPT
    assert "Chỉ dùng tối đa một emoji" in SYSTEM_PROMPT
    assert "dữ liệu tham khảo nội bộ" in SYSTEM_PROMPT
    assert "Không được sao chép cấu trúc, tên trường, nhãn document hoặc metadata" in SYSTEM_PROMPT
    assert "Chỉ xuất một JSON object hợp lệ" in SYSTEM_PROMPT
    assert "Không xuất suy luận nội bộ" in SYSTEM_PROMPT
    assert "<thought>" in SYSTEM_PROMPT
    assert "Không lặp lại hoặc tóm tắt câu hỏi" in SYSTEM_PROMPT
    assert "Question:" in SYSTEM_PROMPT
    assert "Role:" in SYSTEM_PROMPT
    assert "Constraint:" in SYSTEM_PROMPT
    assert "Document:" in SYSTEM_PROMPT
    assert "source_ids" in SYSTEM_PROMPT
    assert "Không tự viết marker trong text/cells" in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.index("</response_style>") < SYSTEM_PROMPT.index("<output_contract>")
    assert SYSTEM_PROMPT.index("</output_contract>") < SYSTEM_PROMPT.rindex("<retrieved_context>")


@pytest.mark.parametrize(
    "question",
    [
        "PTIT có những ngành đào tạo nào?",
        "Học phí PTIT năm 2026 là bao nhiêu?",
        "Điểm chuẩn ngành Công nghệ thông tin năm 2025 là bao nhiêu?",
        "PTIT có những phương thức tuyển sinh nào trong năm 2026?",
        "So sánh học phí chương trình chuẩn và chất lượng cao.",
    ],
)
def test_generation_contract_preserves_representative_questions(question):
    message = build_rag_prompt([], question)[0]

    assert message["role"] == "user"
    assert question in message["content"]
    assert message["content"].startswith("<user_question>")
    assert message["content"].endswith("</user_question>")
    assert "retrieved_context" not in message["content"]


def test_fallback_is_natural_and_non_technical():
    assert FALLBACK_ANSWER == "Mình chưa tìm thấy thông tin này trong dữ liệu tuyển sinh hiện có."
    for internal_term in ("retrieved_context", "retrieval", "chunk", "pipeline", "system prompt"):
        assert internal_term not in FALLBACK_ANSWER.lower()


@pytest.mark.parametrize(
    ("wrapped", "expected"),
    [
        ("<assistant_answer>\n### Ngành đào tạo\n- CNTT [1]\n</assistant_answer>", "### Ngành đào tạo\n- CNTT [1]"),
        ("<response>Thông tin tuyển sinh [1]</response>", "Thông tin tuyển sinh [1]"),
        ("<thought>Phân tích nội bộ</thought>\n<final>Thông tin tuyển sinh [1]</final>", "Thông tin tuyển sinh [1]"),
        ("<analysis>Phân tích chưa hoàn tất", ""),
        ("Nội dung <document> hợp lệ trong câu trả lời", "Nội dung <document> hợp lệ trong câu trả lời"),
    ],
)
def test_clean_answer_only_removes_known_outer_wrappers(wrapped, expected):
    assert RAGChain._clean_answer(wrapped) == expected

def test_session_memory_basic():
    mem = SessionMemory()
    session_id = mem.create_session()
    
    mem.add_message(session_id, "user", "Hello")
    history = mem.get_history(session_id)
    assert len(history) == 1
    assert history[0]["role"] == "user"
    
def test_session_memory_isolation():
    mem = SessionMemory()
    sid1 = mem.create_session()
    sid2 = mem.create_session()
    
    mem.add_message(sid1, "user", "Msg 1")
    mem.add_message(sid2, "user", "Msg 2")
    
    h1 = mem.get_history(sid1)
    h2 = mem.get_history(sid2)
    assert h1[0]["content"] == "Msg 1"
    assert h2[0]["content"] == "Msg 2"

def test_session_memory_max_turns():
    mem = SessionMemory()
    sid = mem.create_session()
    
    # 6 turns (12 messages)
    for i in range(6):
        mem.add_message(sid, "user", f"U{i}")
        mem.add_message(sid, "assistant", f"A{i}")
        
    history = mem.get_history(sid, max_turns=5)
    # 5 turns = 10 messages
    assert len(history) == 10
    assert history[0]["content"] == "U1"  # First message should be U1, dropping U0 and A0
    assert history[-1]["content"] == "A5"

def test_lite_ranks_beyond_neural_cap_before_expanding_and_preserves_markers(monkeypatch):
    from src.config import settings
    from src.ingestion.chunker import Chunk
    from src.retrieval.fusion import HybridRetriever
    from src.retrieval.lite_selector import LiteEvidenceSelector
    from unittest.mock import MagicMock

    monkeypatch.setattr(settings, "RERANK_CANDIDATE_K", 8)
    metadata = {"doc_id": "d", "admission_year": "2031", "campus": "BVH", "heading": "Điểm chuẩn"}
    distractors = [(Chunk(f"noise-{i}", "Điểm chuẩn CN10", {**metadata, "campus": "BVS"}), 1.0) for i in range(12)]
    child = Chunk("seed", "Điểm chuẩn CN1 27.5", {**metadata, "parent_chunk_id": "parent", "chunk_type": "child"})
    parent = Chunk("parent", "Điểm chuẩn CN1 BVH 2031: 27.5", {**metadata, "chunk_type": "parent"})

    class Branch:
        def __init__(self, results):
            self.results = results
        def search(self, query, top_k):
            return self.results[:top_k]

    hybrid = HybridRetriever(Branch(distractors + [(child, .5)]), Branch([]), [parent.__dict__])
    original_expand = hybrid.expand
    hybrid.expand = MagicMock(wraps=original_expand)
    chain = RAGChain(hybrid, MagicMock(), top_k=5, min_lexical_coverage=0, reranker=LiteEvidenceSelector())
    prepared = chain._prepare_generation("Điểm chuẩn CN1 BVH 2031", None)
    assert [item[0].chunk_id for item in hybrid.expand.call_args.args[1]] == ["seed"]
    assert [source.prompt_chunk.chunk_id for source in prepared.sources] == ["parent"]
    assert 'id="1"' in prepared.system_prompt
    assert 'doc_id="d"' in prepared.system_prompt


def test_lite_scope_uses_original_question_when_rewriter_drops_year():
    from src.ingestion.chunker import Chunk
    from src.retrieval.lite_selector import LiteEvidenceSelector
    from unittest.mock import MagicMock

    retriever = MockRetriever([Chunk("old", "Học phí năm 2030", {"admission_year": "2030"})])
    rewriter = MagicMock()
    rewriter.rewrite.return_value = "Học phí"
    chain = RAGChain(retriever, MagicMock(), query_rewriter=rewriter, reranker=LiteEvidenceSelector())
    assert chain._prepare_generation("Học phí năm 2031", None) is None


def test_rag_chain_answer_order():
    chunks = [MockChunk(1, "Info", {"source": "file.pdf"})]
    retriever = MockRetriever(chunks)
    
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Info [1]"
    
    chain = RAGChain(retriever, mock_llm)
    
    response = chain.answer("Info?")
    
    assert response.answer == "Info"
    assert response.citations[0]["source"] == "file.pdf"
    assert response.route_type == "general"
    
    mock_llm.generate.assert_called_once()
    
    call_kwargs = mock_llm.generate.call_args[1]
    assert "file.pdf" in call_kwargs["system_prompt"]
    assert "Info" in call_kwargs["system_prompt"]
    assert len(call_kwargs["messages"]) == 1
    assert "Info?" in call_kwargs["messages"][0]["content"]
    assert "file.pdf" not in call_kwargs["messages"][0]["content"]


@pytest.mark.parametrize(
    "answer,expected_markers",
    [
        ("Only second [2]", [2]),
        ("Duplicate [1] and again [1]", [1]),
        ("Unknown [99]", []),
        ("No marker", []),
        ("Valid [2] but invalid [99]", [2]),
    ],
)
def test_citations_return_only_valid_markers(answer, expected_markers):
    chunks = [
        MockChunk("a", "admission alpha", {"doc_id": "a", "page_number": 1}),
        MockChunk("b", "admission beta", {"doc_id": "b", "page_number": 2}),
    ]
    sources = RAGChain._select_citation_sources(
        "admission", [(chunks[0], 0.9), (chunks[1], 0.8)], preserve_order=True
    )

    citations = RAGChain._citations_for_answer(sources, answer, "admission")

    assert [citation["marker"] for citation in citations] == expected_markers
    assert [citation["chunk_id"] for citation in citations] == [
        chunks[marker - 1].chunk_id for marker in expected_markers
    ]


def test_citation_collision_keeps_each_prompt_evidence_self_consistent(caplog):
    chunks = [
        MockChunk(
            "collision",
            "admission evidence from A",
            {"doc_id": "doc-a", "page_number": 1, "source_url": "https://a.test"},
        ),
        MockChunk(
            "collision",
            "admission evidence from B",
            {"doc_id": "doc-b", "page_number": 7, "source_url": "https://b.test"},
        ),
    ]

    with caplog.at_level(logging.WARNING, logger="src.generation.rag_chain"):
        sources = RAGChain._select_citation_sources(
            "admission evidence", [(chunks[0], 0.9), (chunks[1], 0.8)], preserve_order=True
        )
    citations = RAGChain._citations_for_answer(
        sources, "Evidence A [1]. Evidence B [2].", "admission evidence"
    )

    assert len(sources) == 2
    assert [(source.prompt_chunk.content, source.prompt_chunk.metadata["doc_id"]) for source in sources] == [
        (chunks[0].content, "doc-a"),
        (chunks[1].content, "doc-b"),
    ]
    assert [(citation["doc_id"], citation["page"], citation["source_url"]) for citation in citations] == [
        ("doc-a", 1, "https://a.test"),
        ("doc-b", 7, "https://b.test"),
    ]
    assert any(message.startswith("citation_identity_collision ") for message in caplog.messages)
    assert "admission evidence" not in "\n".join(caplog.messages)


@pytest.mark.parametrize("preserve_order", [False, True])
def test_citation_representative_fields_and_score_come_from_one_member(preserve_order):
    provenance = {
        "doc_id": "doc",
        "page_number": 4,
        "heading": "Quota",
        "table_index": 1,
        "table_part": 1,
        "row_start": 1,
        "row_end": 1,
    }
    first = MockChunk(
        "first",
        "Admission quota 500",
        {**provenance, "expansion_type": "exact"},
    )
    higher_score = MockChunk(
        "higher-score",
        " admission   QUOTA 500 ",
        {**provenance, "expansion_type": "neighbor"},
    )

    source = RAGChain._select_citation_sources(
        "admission quota",
        [(first, 0.4), (higher_score, 0.9)],
        preserve_order=preserve_order,
    )[0]
    expected = first if preserve_order else higher_score
    expected_score = 0.4 if preserve_order else 0.9

    assert source.prompt_chunk.chunk_id == expected.chunk_id
    assert source.prompt_chunk.content == expected.content
    assert source.prompt_chunk.metadata == expected.metadata
    assert source.retrieval_score == expected_score


@pytest.mark.parametrize("streaming", [False, True])
def test_answer_paths_keep_identical_table_rows_at_distinct_locations(streaming):
    common = {
        "doc_id": "doc",
        "page_number": 4,
        "heading": "Quota",
        "table_index": 2,
        "chunk_type": "table",
        "source_url": "https://example.test/quota",
    }
    chunks = [
        MockChunk(
            "part-1",
            "admission quota | BVS | 300 |",
            {**common, "table_part": 1, "row_start": 1, "row_end": 1},
        ),
        MockChunk(
            "part-2",
            "admission quota | BVS | 300 |",
            {**common, "table_part": 2, "row_start": 5, "row_end": 5},
        ),
    ]
    retriever = MagicMock()
    retriever.search.return_value = [(chunks[0], 0.8), (chunks[1], 0.7)]
    llm = MagicMock()
    answer = "admission quota [1] [2]"
    llm.generate.return_value = answer
    llm.generate_stream.return_value = iter([answer])
    chain = RAGChain(retriever, llm)

    if streaming:
        events = list(chain.answer_stream("admission quota BVS"))
        citations = events[-1]["citations"]
        prompt = llm.generate.call_args.kwargs["system_prompt"]
    else:
        response = chain.answer("admission quota BVS")
        citations = response.citations
        prompt = llm.generate.call_args.kwargs["system_prompt"]

    assert prompt.count('<document id="') == 2
    assert [citation["chunk_id"] for citation in citations] == ["part-1", "part-2"]
    assert [citation["score"] for citation in citations] == [0.8, 0.7]


def test_table_excerpt_prefers_matching_year_campus_and_value():
    content = "\n".join(
        [
            "| Year | Campus | Quota |",
            "| --- | --- | --- |",
            "| 2025 | BVH | 500 |",
            "| 2025 | BVS | 300 |",
            "| 2026 | BVS | 450 |",
        ]
    )
    chunk = MockChunk(
        "table",
        content,
        {"doc_id": "admissions", "page_number": 4, "source_url": "https://example.test/table"},
    )
    sources = RAGChain._select_citation_sources(
        "quota BVS 2026", [(chunk, 0.71)], preserve_order=True
    )

    citation = RAGChain._citations_for_answer(
        sources, "The BVS quota for 2026 is 450 [1].", "quota BVS 2026"
    )[0]

    assert citation["snippet"] == "| 2026 | BVS | 450 |"
    assert citation["doc_id"] == "admissions"
    assert citation["page"] == 4
    assert citation["source_url"] == "https://example.test/table"


def test_rag_chain_stream_does_not_expose_model_thought_blocks():
    chunks = [MockChunk(1, "Thông tin tuyển sinh", {"source": "file.pdf"})]
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "<thought>Phân tích nội bộ</thought><final>Thông tin tuyển sinh [1]</final>"
    mock_llm.generate_stream.return_value = iter(
        [
            "<thought>Phân tích ",
            "nội bộ</thought><final>",
            "Thông tin tuyển sinh [1]",
            "</final>",
        ]
    )

    events = list(RAGChain(MockRetriever(chunks), mock_llm).answer_stream("Thông tin tuyển sinh?"))
    visible_answer = "".join(event["text"] for event in events if event["type"] == "chunk")

    assert visible_answer == "Thông tin tuyển sinh"
    assert "Phân tích nội bộ" not in visible_answer
    assert events[-1]["type"] == "metadata"


def test_rag_chain_keeps_context_marker_and_citation_payload_in_the_same_order():
    chunks = [
        MockChunk(1, "Ngành thứ nhất", {"doc_id": "same-doc", "source": "file.pdf"}),
        MockChunk(2, "Ngành thứ hai", {"doc_id": "same-doc", "source": "file.pdf"}),
    ]
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "- Ngành thứ nhất [1]\n- Ngành thứ hai [2]"

    response = RAGChain(MockRetriever(chunks), mock_llm).answer("Ngành thứ nhất và ngành thứ hai")

    assert [citation["chunk_id"] for citation in response.citations] == ["1", "2"]
    system_prompt = mock_llm.generate.call_args.kwargs["system_prompt"]
    assert system_prompt.index('id="1"') < system_prompt.index('id="2"')

@pytest.mark.parametrize("streaming", [False, True])
def test_rag_chain_cites_each_chunk_from_the_same_document_without_a_two_source_limit(streaming):
    chunks = [
        MockChunk(index, f"Thông tin tuyển sinh ngành {index}", {"doc_id": "same-doc"})
        for index in range(1, 5)
    ]
    llm = MagicMock()
    answer = "Thông tin tuyển sinh [1] [2] [3] [4]"
    llm.generate.return_value = answer
    llm.generate_stream.return_value = iter([answer])
    chain = RAGChain(MockRetriever(chunks), llm)

    if streaming:
        events = list(chain.answer_stream("Thông tin tuyển sinh"))
        citations = events[-1]["citations"]
        system_prompt = llm.generate.call_args.kwargs["system_prompt"]
    else:
        response = chain.answer("Thông tin tuyển sinh")
        citations = response.citations
        system_prompt = llm.generate.call_args.kwargs["system_prompt"]

    assert [citation["marker"] for citation in citations] == [1, 2, 3, 4]
    assert [citation["chunk_id"] for citation in citations] == ["1", "2", "3", "4"]
    for index in range(1, 5):
        assert f'<document id="{index}"' in system_prompt


def test_rag_chain_empty_context():
    retriever = MockRetriever([])
    mock_llm = MagicMock()
    
    chain = RAGChain(retriever, mock_llm)
    response = chain.answer("Question?")
    
    # LLM should not be called
    mock_llm.generate.assert_not_called()
    assert response.answer == FALLBACK_ANSWER
    assert len(response.citations) == 0

def test_rag_chain_empty_question():
    chunks = [MockChunk(1, "Info")]
    retriever = MockRetriever(chunks)
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Answer"
    
    chain = RAGChain(retriever, mock_llm)
    chain.answer("")
    
    # Invalid input must not spend an LLM call or fabricate an answer.
    mock_llm.generate.assert_not_called()


@pytest.mark.parametrize("streaming", [False, True])
def test_rag_timing_logs_all_stages_without_query_or_content(caplog, streaming):
    question = "admission-private-query"
    chunk = MockChunk("winner", "admission private query sensitive-content")
    reranker = SimpleNamespace(
        rerank=lambda _query, candidates, top_k: [
            (candidate, score, 1.0) for candidate, score in candidates[:top_k]
        ]
    )
    llm = MagicMock()
    llm.generate.return_value = "admission private query [1]"
    llm.generate_stream.return_value = iter(["admission private query [1]"])
    chain = RAGChain(MockRetriever([chunk]), llm, reranker=reranker)

    with caplog.at_level(logging.INFO, logger="src.generation.rag_chain"):
        if streaming:
            list(chain.answer_stream(question))
        else:
            chain.answer(question)

    timing_messages = [message for message in caplog.messages if message.startswith("rag_timing")]
    assert {message.split()[1] for message in timing_messages} == {
        "stage=query_transform",
        "stage=retrieval",
        "stage=rerank",
        "stage=generation",
        "stage=total",
    }
    combined = "\n".join(timing_messages)
    assert "stage=retrieval" in combined and "candidates=1" in combined
    assert "stage=rerank" in combined
    assert "input_candidates=1" in combined
    assert "output_candidates=1" in combined
    assert "avg_candidate_chars=" in combined
    assert "max_candidate_chars=" in combined
    assert question not in combined
    assert chunk.content not in combined
