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


def test_system_prompt_enforces_grounding_and_natural_markdown():
    assert "Chỉ dùng thông tin có trong <retrieved_context>" in SYSTEM_PROMPT
    assert "không suy đoán" in SYSTEM_PROMPT
    assert "marker [n] ngay cuối ý hoặc cuối đoạn" in SYSTEM_PROMPT
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
    assert "Chỉ xuất câu trả lời cuối cùng" in SYSTEM_PROMPT
    assert "Không xuất suy luận nội bộ" in SYSTEM_PROMPT
    assert "<thought>" in SYSTEM_PROMPT
    assert "Không lặp lại hoặc tóm tắt câu hỏi" in SYSTEM_PROMPT
    assert "Question:" in SYSTEM_PROMPT
    assert "Role:" in SYSTEM_PROMPT
    assert "Constraint:" in SYSTEM_PROMPT
    assert "Document:" in SYSTEM_PROMPT
    assert "Citation chỉ xuất dưới dạng marker [n]" in SYSTEM_PROMPT
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

def test_rag_chain_answer_order():
    chunks = [MockChunk(1, "Info", {"source": "file.pdf"})]
    retriever = MockRetriever(chunks)
    
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Answer generated"
    
    chain = RAGChain(retriever, mock_llm)
    
    response = chain.answer("Info?")
    
    assert response.answer == "Answer generated"
    assert response.citations[0]["source"] == "file.pdf"
    assert response.route_type == "general"
    
    mock_llm.generate.assert_called_once()
    
    call_kwargs = mock_llm.generate.call_args[1]
    assert "file.pdf" in call_kwargs["system_prompt"]
    assert "Info" in call_kwargs["system_prompt"]
    assert len(call_kwargs["messages"]) == 1
    assert "Info?" in call_kwargs["messages"][0]["content"]
    assert "file.pdf" not in call_kwargs["messages"][0]["content"]


def test_rag_chain_stream_does_not_expose_model_thought_blocks():
    chunks = [MockChunk(1, "Thông tin tuyển sinh", {"source": "file.pdf"})]
    mock_llm = MagicMock()
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

    assert visible_answer == "Thông tin tuyển sinh [1]"
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
