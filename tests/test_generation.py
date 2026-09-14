import pytest
from unittest.mock import MagicMock, patch
from src.generation.llm_client import LLMClient
from src.generation.prompts import build_rag_prompt, format_citations, SYSTEM_PROMPT
from src.generation.session_memory import SessionMemory
from src.generation.rag_chain import RAGChain, RAGResponse

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

        client = LLMClient(api_key="fake_key", model="fake-model")
        answer = client.generate("system prompt", [{"role": "user", "content": "hello"}])
        
        assert answer == "This is a mock answer."
        mock_client_instance.chat.completions.create.assert_called_once()

def test_llm_client_error():
    with patch("src.generation.llm_client.OpenAI") as MockOpenAI:
        mock_client_instance = MagicMock()
        MockOpenAI.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")

        client = LLMClient(api_key="fake", model="fake")
        with pytest.raises(RuntimeError, match="LLM Provider Error: API Error"):
            client.generate("sys", [])

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

def test_rag_chain_empty_context():
    retriever = MockRetriever([])
    mock_llm = MagicMock()
    
    chain = RAGChain(retriever, mock_llm)
    response = chain.answer("Question?")
    
    # LLM should not be called
    mock_llm.generate.assert_not_called()
    assert "chưa tìm thấy" in response.answer
    assert "tuyển sinh" in response.answer
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
