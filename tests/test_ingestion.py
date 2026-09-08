import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.chunker import naive_chunk
from src.ingestion.parser import parse_pdf

def test_naive_chunk_empty_text():
    chunks = naive_chunk("")
    assert len(chunks) == 0

def test_naive_chunk_short_text():
    text = "Short text"
    chunks = naive_chunk(text, chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].content == text

def test_naive_chunk_long_text_and_overlap():
    text = "0123456789"
    # chunk_size = 6, overlap = 2 -> step = 4
    # chunks should be:
    # 0: "012345"
    # 1: "456789"
    chunks = naive_chunk(text, chunk_size=6, overlap=2)
    assert len(chunks) == 2
    assert chunks[0].content == "012345"
    assert chunks[1].content == "456789"

def test_naive_chunk_last_chunk():
    text = "0123456789a"
    # chunk_size = 6, overlap = 2 -> step = 4
    # chunks should be:
    # 0: "012345"
    # 1: "456789"
    # 2: "89a"
    chunks = naive_chunk(text, chunk_size=6, overlap=2)
    assert len(chunks) == 3
    assert chunks[0].content == "012345"
    assert chunks[1].content == "456789"
    assert chunks[2].content == "89a"

def test_naive_chunk_validation():
    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        naive_chunk("test", chunk_size=5, overlap=5)
    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        naive_chunk("test", chunk_size=5, overlap=6)

def test_naive_chunk_stable_id():
    text = "hello world"
    chunks1 = naive_chunk(text, chunk_size=5, overlap=1)
    chunks2 = naive_chunk(text, chunk_size=5, overlap=1)
    assert chunks1[0].chunk_id == chunks2[0].chunk_id
    assert chunks1[0].metadata["chunk_index"] == 0

def test_parse_pdf_mock():
    with patch("src.ingestion.parser.pdfplumber.open") as mock_open:
        # Set up the mock
        mock_pdf = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_page3 = MagicMock()
        mock_page3.extract_text.return_value = None # test handling None
        
        mock_pdf.pages = [mock_page1, mock_page2, mock_page3]
        
        # mock_open returns a context manager that yields mock_pdf
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_pdf
        mock_open.return_value = mock_context_manager
        
        pages = parse_pdf("dummy.pdf")
        
        assert len(pages) == 3
        assert pages[0].page_number == 1
        assert pages[0].text == "Page 1 content"
        assert pages[1].page_number == 2
        assert pages[1].text == "Page 2 content"
        assert pages[2].page_number == 3
        assert pages[2].text == ""
