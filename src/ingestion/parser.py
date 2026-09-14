import pdfplumber
import re
from dataclasses import dataclass, field
from typing import List, Any, Dict, Tuple
from collections import defaultdict

@dataclass
class Heading:
    text: str
    level: int
    page_number: int

@dataclass
class Table:
    rows: List[List[str]]
    page_number: int

@dataclass
class ParsedPage:
    page_number: int
    text: str
    tables: List[Table] = field(default_factory=list)
    headings: List[Heading] = field(default_factory=list)

def _extract_headings(page: pdfplumber.page.Page) -> List[Heading]:
    """
    Extracts headings from a pdfplumber page using font sizes and heuristics.
    """
    headings = []
    
    try:
        # Extract words with size and position attributes
        words = page.extract_words(extra_attrs=["size"])
        if not words:
            return headings

        # Calculate median font size (body size)
        sizes = sorted([w["size"] for w in words])
        if not sizes:
            return headings
        median_size = sizes[len(sizes) // 2]
        
        # Group words into lines by approximating y-coordinate (top)
        lines = defaultdict(list)
        for word in words:
            # round top to nearest 2 pixels to group same-line words
            rounded_top = round(word["top"] / 2.0) * 2
            lines[rounded_top].append(word)

        # Sort lines by y-coordinate (top to bottom)
        sorted_tops = sorted(lines.keys())
        
        for top in sorted_tops:
            line_words = sorted(lines[top], key=lambda w: w["x0"])
            text = " ".join(w["text"] for w in line_words).strip()
            if not text:
                continue
                
            # calculate average size of the line
            avg_size = sum(w["size"] for w in line_words) / len(line_words)
            
            level = None
            # Heuristic 1: Font size significantly larger than median
            if avg_size > median_size + 4.0:
                level = 1
            elif avg_size > median_size + 1.5:
                level = 2
                
            # Heuristic 2: Matches common heading patterns in Vietnamese docs
            # e.g., "Điều 1.", "Chương I.", "Phần 1:", "I.", "1."
            pattern = r'^(Điều \d+|Chương [IVXLCDM]+|Phần [IVXLCDM]+|I{1,3}\.|IV\.|V\.|VI{0,3}\.|[A-Z]\.|[0-9]+\.)'
            if re.match(pattern, text, re.IGNORECASE):
                if level is None:
                    level = 3  # Default to level 3 if size doesn't indicate larger
                    
            if level is not None:
                headings.append(Heading(text=text, level=level, page_number=page.page_number))
    except Exception as e:
        print(f"Warning: Failed to extract headings on page {page.page_number} - {e}")

    return headings

def _extract_tables(page: pdfplumber.page.Page) -> List[Table]:
    """
    Extracts tables from a pdfplumber page with graceful error handling.
    """
    tables = []
    try:
        extracted = page.extract_tables()
        for t in extracted:
            if not t:
                continue
            # Remove empty rows and None values, clean up text
            cleaned_table = []
            for row in t:
                # filter out fully empty rows
                if all(cell is None or str(cell).strip() == "" for cell in row):
                    continue
                cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                cleaned_table.append(cleaned_row)
            
            # Skip empty tables after cleaning
            if cleaned_table:
                tables.append(Table(rows=cleaned_table, page_number=page.page_number))
    except Exception as e:
        print(f"Warning: Failed to extract table on page {page.page_number} - {e}")
    
    return tables

def parse_pdf(file_path: str) -> List[ParsedPage]:
    """
    Parses a PDF file and extracts text, headings, and tables per page.
    """
    parsed_pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text is None:
                text = ""
                
            tables = _extract_tables(page)
            headings = _extract_headings(page)
            
            parsed_pages.append(
                ParsedPage(
                    page_number=i + 1,
                    text=text,
                    tables=tables,
                    headings=headings
                )
            )
    return parsed_pages
