import pdfplumber
from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class ParsedPage:
    page_number: int
    text: str
    tables: List[Any] = field(default_factory=list)

def parse_pdf(file_path: str) -> List[ParsedPage]:
    """
    Parses a PDF file and extracts text per page.
    Tables are intentionally left as empty lists for Phase 1.
    """
    parsed_pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text is None:
                text = ""
            # For Phase 1, we just extract text. Table extraction is left for Phase 2.
            parsed_pages.append(
                ParsedPage(
                    page_number=i + 1,
                    text=text,
                    tables=[]
                )
            )
    return parsed_pages
