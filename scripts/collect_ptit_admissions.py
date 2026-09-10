"""Collect and normalize official PTIT undergraduate-admission sources.

The collector stores byte-for-byte HTTP response bodies in data/raw/ptit and
creates citation-ready Markdown in data/processed/ptit. Re-running never
overwrites a changed response: a UTC timestamp is appended to the new version.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import pdfplumber
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as to_markdown


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "ptit"
PROCESSED_ROOT = ROOT / "data" / "processed" / "ptit"
RETRIEVED_AT = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
USER_AGENT = "AdmitAI-PTIT-Official-Admissions-Collector/1.0 (+research; citation corpus)"
TIMEOUT = 45

REQUIRED_FRONTMATTER = {
    "doc_id", "title", "source_url", "source_domain", "source_type",
    "published_at", "updated_at", "retrieved_at", "academic_year",
    "admission_year", "department", "audience", "language", "content_hash",
    "raw_file",
}


@dataclass(frozen=True)
class Seed:
    url: str
    category: str
    source_type: str = "official_web"
    admission_year: str = "2026"
    academic_year: str = "2026-2027"
    department: str = "Học viện Công nghệ Bưu chính Viễn thông"
    audience: str = "candidate"
    supersedes: str = ""
    note: str = ""


SEEDS = [
    Seed("https://tuyensinh.ptit.edu.vn/", "admissions", "official_web"),
    Seed("https://tuyensinh.ptit.edu.vn/thong-tin-tuyen-sinh-dai-hoc-chinh-quy/", "admissions", "official_web"),
    Seed("https://tuyensinh.ptit.edu.vn/de-an-tuyen-sinh/thong-tin-tuyen-sinh-dai-hoc-chinh-quy-nam-2026/", "admissions", "official_web", note="Trang chuyên mục có nội dung trùng/bao quát cùng thông tin tuyển sinh 2026; giữ để phát hiện duplicate."),
    Seed("https://tuyensinh.ptit.edu.vn/sua-doi-bo-sung-mot-so-noi-dung-cua-thong-tin-tuyen-sinh-dai-hoc-he-chinh-quy-nam-2026/", "announcements", "announcement", supersedes="ptit-2026-admissions-thong-tin-tuyen-sinh-dai-hoc-chinh-quy-v1"),
    Seed("https://tuyensinh.ptit.edu.vn/thong-baophuong-thuc-tuyen-sinh-dai-hoc-he-chinh-quy-nam-2026/", "admissions", "announcement"),
    Seed("https://tuyensinh.ptit.edu.vn/thong-bao-tuyen-sinh-dai-hoc-chinh-quy-nam-2026/", "admissions", "announcement"),
    Seed("https://tuyensinh.ptit.edu.vn/thong-bao-xet-tuyen-thang-va-uu-tien-xet-tuyen-vao-dai-hoc-he-chinh-quy-nam-2026/", "admissions", "announcement"),
    Seed("https://tuyensinh.ptit.edu.vn/thong-bao-ve-viec-mo-he-thong-dang-ky-thong-tin-tuyen-sinh-truc-tuyen-vao-dai-hoc-chinh-quy-nam-2026/", "admissions", "announcement"),
    Seed("https://tuyensinh.ptit.edu.vn/thong-bao-dieu-chinh-thoi-gian-dang-ky-thong-tin-xet-tuyen-truc-tuyen-cho-thi-sinh-dang-ky-xet-tuyen-vao-dai-hoc-chinh-quy-nam-2026/", "announcements", "announcement"),
    Seed("https://tuyensinh.ptit.edu.vn/thong-bao-nguong-dam-bao-chat-luong-dau-vao-trinh-do-dai-hoc-chinh-quy-dot-1-nam-2026/", "admissions", "announcement"),
    Seed("https://tuyensinh.ptit.edu.vn/thong-bao-bang-quy-doi-tuong-duong-giua-cac-phuong-thuc-xet-tuyen-dai-hoc-he-chinh-quy-nam-2026/", "admissions", "announcement"),
    Seed("https://tuyensinh.ptit.edu.vn/thong-bao-chuan-dau-vao-chuong-trinh-dao-tao-thac-si-tai-nang-va-chuong-trinh-dao-tao-vi-mach-ban-dan-trinh-do-dai-hoc-doi-voi-phuong-thuc-xet-tuyen-dua-vao-ket-qua-thi-tot-nghiep-thpt-nam-2026/", "admissions", "announcement"),
    Seed("https://tuyensinh.ptit.edu.vn/thong-bao-diem-chuan-trung-tuyen-vao-dai-hoc-he-chinh-quy-nam-2026/", "admissions", "announcement"),
    Seed("https://tuyensinh.ptit.edu.vn/thong-bao-ve-viec-nhap-hoc-dai-hoc-chinh-quy-nam-2026-co-so-dao-tao-phia-bac-bvh/", "admissions", "announcement"),
    Seed("https://tuyensinh.ptit.edu.vn/gioi-thieu/chinh-sach-hoc-bong/", "scholarships", "official_web"),
    Seed("https://tuyensinh.ptit.edu.vn/gioi-thieu/cau-hoi-thuong-gap/", "admissions", "official_web", note="FAQ công khai; một số câu trả lời có thể thuộc mùa tuyển sinh cũ và được giữ nguyên ngày nguồn."),
    Seed("https://tuyensinh.ptit.edu.vn/quy-che-6/", "regulations", "regulation"),
    Seed("https://tuyensinh.ptit.edu.vn/2026-sua-doi-bo-sung-mot-so-dieu-cua-quy-che-tuyen-sinh-dai-hoc-cua-hoc-vien-cong-nghe-buu-chinh-vien-thong/", "regulations", "regulation", supersedes="ptit-2026-regulations-quy-che-6-v1"),
    Seed("https://tuyensinh.ptit.edu.vn/quy-che-5/", "regulations", "regulation"),
    Seed("https://tuyensinh.ptit.edu.vn/de-an-tuyen-sinh/thong-tin-tuyen-sinh-dai-hoc-chinh-quy-nam-2025/", "admissions", "official_web", "2025", "2025-2026"),
    Seed("https://tuyensinh.ptit.edu.vn/thong-bao-tuyen-sinh-dai-hoc-he-chinh-quy-nam-2025/", "admissions", "announcement", "2025", "2025-2026"),
    Seed("https://tuyensinh.ptit.edu.vn/thong-bao-ve-viec-sua-doi-bo-sung-mot-so-noi-dung-cua-de-an-tuyen-sinh-va-thong-bao-tuyen-sinh-dai-hoc-he-chinh-quy-nam-2025/", "announcements", "announcement", "2025", "2025-2026"),
    Seed("https://tuyensinh.ptit.edu.vn/gioi-thieu/xem-diem-cac-nam-truoc/diem-trung-tuyen-nam-2025/", "admissions", "official_web", "2025", "2025-2026"),
    Seed("https://tuyensinh.ptit.edu.vn/gioi-thieu/xem-diem-cac-nam-truoc/diem-trung-tuyen-2024/", "admissions", "official_web", "2024", "2024-2025"),
    Seed("https://tuyensinh.ptit.edu.vn/gioi-thieu/xem-diem-cac-nam-truoc/diem-trung-tuyen-2023/", "admissions", "official_web", "2023", "2023-2024"),
    Seed("https://tuyensinh.ptit.edu.vn/de-an-tuyen-sinh-nam-2024/", "admissions", "official_web", "2024", "2024-2025"),
    Seed("https://tuyensinh.ptit.edu.vn/de-an-tuyen-sinh-nam-2023/", "admissions", "official_web", "2023", "2023-2024"),
]

# Current program pages published by the official admissions portal. Slug/title
# matching is intentionally narrow so unrelated news is never silently included.
PROGRAM_TITLE_TERMS = {
    "công nghệ thông tin", "an toàn thông tin", "trí tuệ nhân tạo",
    "khoa học máy tính", "kỹ thuật dữ liệu", "kỹ thuật điện tử viễn thông",
    "kỹ thuật điều khiển", "công nghệ kỹ thuật điện", "báo chí",
    "truyền thông đa phương tiện", "công nghệ đa phương tiện",
    "quản trị kinh doanh", "thương mại điện tử", "marketing", "kế toán",
    "công nghệ tài chính", "logistics", "quan hệ công chúng",
    "thiết kế và phát triển game", "vi mạch bán dẫn", "internet vạn vật",
    "trí tuệ nhân tạo vạn vật", "phân tích dữ liệu", "đổi mới sáng tạo",
    "uav", "hàng không vũ trụ", "công nghệ công nghiệp văn hóa",
    "thiết kế đồ họa game",
}
PROGRAM_EXCLUDE_TERMS = {
    "điểm chuẩn", "tuyển sinh", "thông báo", "học bổng", "hội thảo",
    "talkshow", "cuộc thi", "ngày hội", "chúc mừng", "khai giảng",
    "kết quả", "kiểm tra", "kết thúc", "có gì khác biệt", "góc nhìn",
    "thông tin báo chí",
}

PROGRAM_URLS = [
    "https://tuyensinh.ptit.edu.vn/nhom-cac-nganh-chuong-trinh-dao-tao-linh-vuc-ky-thuat-cong-nghe/",
    "https://tuyensinh.ptit.edu.vn/nhom-cac-nganh-chuong-trinh-dao-tao-linh-vuc-kinh-te-kinh-doanh-va-quan-ly/",
    "https://tuyensinh.ptit.edu.vn/nhom-cac-nganh-chuong-trinh-dao-tao-linh-vuc-bao-chi-truyen-thong/",
    "https://tuyensinh.ptit.edu.vn/nhom-cac-chuong-trinh-chat-luong-cao-tien-tien-dac-thu/",
    "https://tuyensinh.ptit.edu.vn/nhom-cac-nganh-chuong-trinh-dao-tao-linh-vuc-ky-thuat-cong-nghe-mien-nam/",
    "https://tuyensinh.ptit.edu.vn/nhom-cac-nganh-chuong-trinh-dao-tao-linh-vuc-kinh-te-kinh-doanh-va-quan-ly-mien-nam/",
    "https://tuyensinh.ptit.edu.vn/nhom-cac-nganh-chuong-trinh-dao-tao-linh-vuc-bao-chi-truyen-thong-mien-nam/",
    "https://tuyensinh.ptit.edu.vn/nhom-cac-chuong-trinh-chat-luong-cao-tien-tien-dac-thu-mien-nam/",
    "https://tuyensinh.ptit.edu.vn/truyen-thong-da-phuong-tien-chat-luong-cao/",
    "https://tuyensinh.ptit.edu.vn/phan-tich-du-lieu-trong-tai-chinh-kinh-doanh/",
    "https://tuyensinh.ptit.edu.vn/logistics-va-quan-tri-chuoi-cung-ung/",
    "https://tuyensinh.ptit.edu.vn/tri-tue-nhan-tao-van-vat-aiot/",
    "https://tuyensinh.ptit.edu.vn/thiet-ke-va-phat-trien-game/",
    "https://tuyensinh.ptit.edu.vn/bao-chi/",
    "https://tuyensinh.ptit.edu.vn/truyen-thong-da-phuong-tien/",
    "https://tuyensinh.ptit.edu.vn/marketing-chat-luong-cao/",
    "https://tuyensinh.ptit.edu.vn/ke-toan-chat-luong-cao/",
    "https://tuyensinh.ptit.edu.vn/quan-he-cong-chung/",
    "https://tuyensinh.ptit.edu.vn/an-toan-thong-tin-chat-luong-cao/",
    "https://tuyensinh.ptit.edu.vn/cong-nghe-thong-tin-chat-luong-cao/",
    "https://tuyensinh.ptit.edu.vn/cong-nghe-thong-tin-cu-nhan-dinh-huong-ung-dung/",
    "https://tuyensinh.ptit.edu.vn/cong-nghe-thong-tin-viet-nhat/",
    "https://tuyensinh.ptit.edu.vn/ky-thuat-du-lieu/",
    "https://tuyensinh.ptit.edu.vn/khoa-hoc-may-tinh/",
    "https://tuyensinh.ptit.edu.vn/tri-tue-nhan-tao/",
    "https://tuyensinh.ptit.edu.vn/cong-nghe-vi-mach-ban-dan/",
    "https://tuyensinh.ptit.edu.vn/ky-thuat-dieu-khien-va-tu-dong-hoa/",
    "https://tuyensinh.ptit.edu.vn/cong-nghe-internet-van-vat-iot/",
    "https://tuyensinh.ptit.edu.vn/tuyen-sinh-ptit-tong-quan-ve-nganh-hoc-cong-nghe-tai-chinh-fintech/",
    "https://tuyensinh.ptit.edu.vn/tuyen-sinh-ptit-tong-quan-ve-nganh-hoc-dien-tu-vien-thong/",
    "https://tuyensinh.ptit.edu.vn/tuyen-sinh-ptit-tong-quan-ve-nganh-hoc-marketing/",
    "https://tuyensinh.ptit.edu.vn/tuyen-sinh-ptit-tong-quan-ve-nganh-hoc-thuong-mai-dien-tu/",
    "https://tuyensinh.ptit.edu.vn/tuyen-sinh-ptit-tong-quan-ve-nganh-hoc-cong-nghe-da-phuong-tien/",
    "https://tuyensinh.ptit.edu.vn/tuyen-sinh-ptit-tong-quan-ve-nganh-hoc-quan-tri-kinh-doanh-tai-ptit/",
    "https://tuyensinh.ptit.edu.vn/tuyen-sinh-ptit-tong-quan-ve-nganh-hoc-cong-nghe-thong-tin/",
    "https://tuyensinh.ptit.edu.vn/tuyen-sinh-ptit-tong-quan-ve-nganh-hoc-tai-chinh-ke-toan-tai-ptit/",
    "https://tuyensinh.ptit.edu.vn/tuyen-sinh-ptit-tong-quan-ve-nganh-hoc-dien-dien-tu/",
    "https://tuyensinh.ptit.edu.vn/tuyen-sinh-ptit-tong-quan-ve-nganh-hoc-an-toan-thong-tin/",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def slugify(value: str, max_len: int = 90) -> str:
    import unicodedata

    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("đ", "d")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:max_len].rstrip("-") or "document"


def yaml_string(value: Any) -> str:
    return json.dumps("" if value is None else str(value), ensure_ascii=False)


def frontmatter(fields: dict[str, Any]) -> str:
    ordered = [
        "doc_id", "title", "source_url", "source_domain", "source_type",
        "published_at", "updated_at", "retrieved_at", "academic_year",
        "admission_year", "department", "audience", "language",
        "content_hash", "raw_file", "page_number", "parent_doc_id",
        "section", "supersedes", "duplicate_of", "version", "extraction_status",
    ]
    lines = ["---"]
    for key in ordered:
        if key in fields:
            if key == "page_number" and isinstance(fields[key], int):
                lines.append(f"{key}: {fields[key]}")
            else:
                lines.append(f"{key}: {yaml_string(fields[key])}")
    for key in sorted(set(fields) - set(ordered)):
        lines.append(f"{key}: {yaml_string(fields[key])}")
    return "\n".join(lines + ["---", ""])


def versioned_path(base: Path, data: bytes) -> tuple[Path, int, bool]:
    """Return path/version/created; preserve old bytes when the source changes."""
    base.parent.mkdir(parents=True, exist_ok=True)
    candidates = [base] + sorted(base.parent.glob(f"{base.stem}_v*{base.suffix}"))
    existing = [p for p in candidates if p.exists()]
    digest = sha256_bytes(data)
    for idx, path in enumerate(existing, start=1):
        if sha256_bytes(path.read_bytes()) == digest:
            return path, idx, False
    if not existing:
        return base, 1, True
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base.with_name(f"{base.stem}_v{stamp}{base.suffix}"), len(existing) + 1, True


def request(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, timeout=TIMEOUT, allow_redirects=True)
    response.raise_for_status()
    return response


def find_wp_api(soup: BeautifulSoup, page_url: str) -> str:
    node = soup.find("link", attrs={"rel": lambda x: x and "alternate" in x, "type": "application/json"})
    if node and node.get("href"):
        return urljoin(page_url, node["href"])
    return ""


def select_content_html(soup: BeautifulSoup) -> str:
    selectors = [
        "article .entry-content", ".entry-content", ".post-content",
        ".td-post-content", ".single-content", "article", "main",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if node and len(node.get_text(" ", strip=True)) >= 80:
            return str(node)
    return str(soup.body or soup)


def clean_content_html(content_html: str, base_url: str) -> tuple[str, list[str], dict[str, int]]:
    soup = BeautifulSoup(content_html, "html.parser")
    original_table_count = len(soup.find_all("table"))
    for tag in soup.find_all(["script", "style", "nav", "footer", "form", "iframe", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(True):
        for attr in ["class", "id", "style", "onclick", "data-elementor-type"]:
            tag.attrs.pop(attr, None)
    links: list[str] = []
    for tag in soup.find_all(["a", "img"]):
        attr = "href" if tag.name == "a" else "src"
        if tag.get(attr):
            tag[attr] = urljoin(base_url, tag[attr])
        if tag.name == "a" and tag.get("href") and tag["href"].startswith("http"):
            links.append(tag["href"])
    for img in soup.find_all("img"):
        if not img.get("alt"):
            img.decompose()
    text = soup.get_text(" ", strip=True)
    stats = {
        "source_table_count": original_table_count,
        "source_numeric_token_count": len(re.findall(r"(?<!\w)\d+(?:[.,:/-]\d+)*(?!\w)", text)),
    }
    return str(soup), sorted(set(links)), stats


def html_table_to_markdown(table: Any) -> str:
    """Expand rowspan/colspan into a rectangular Markdown table."""
    grid: dict[tuple[int, int], str] = {}
    rows = table.find_all("tr")
    max_col = 0
    for row_index, row in enumerate(rows):
        col_index = 0
        cells = row.find_all(["th", "td"], recursive=False)
        for cell in cells:
            while (row_index, col_index) in grid:
                col_index += 1
            try:
                rowspan = max(1, int(cell.get("rowspan", 1)))
            except (TypeError, ValueError):
                rowspan = 1
            try:
                colspan = max(1, int(cell.get("colspan", 1)))
            except (TypeError, ValueError):
                colspan = 1
            value = re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).replace("|", "\\|")
            for row_offset in range(rowspan):
                for col_offset in range(colspan):
                    grid[(row_index + row_offset, col_index + col_offset)] = value
            col_index += colspan
            max_col = max(max_col, col_index)
    total_rows = max((row for row, _ in grid), default=-1) + 1
    if not total_rows or not max_col:
        return ""
    matrix = [[grid.get((row, col), "") for col in range(max_col)] for row in range(total_rows)]
    lines = [
        "| " + " | ".join(matrix[0]) + " |",
        "| " + " | ".join(["---"] * max_col) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in matrix[1:])
    return "\n".join(lines)


def html_to_md(content_html: str) -> str:
    soup = BeautifulSoup(content_html, "html.parser")
    table_replacements: dict[str, str] = {}
    for index, table in enumerate(soup.find_all("table")):
        token = f"PTITTABLETOKEN{index:04d}ZZZ"
        table_replacements[token] = html_table_to_markdown(table)
        table.replace_with(soup.new_string(token))
    md = to_markdown(
        str(soup),
        heading_style="ATX",
        bullets="-",
        strip=["span"],
        newline_style="backslash",
    )
    md = html.unescape(md).replace("\u00a0", " ")
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r"^\s*Menu\s*$", "", md, flags=re.MULTILINE)
    for token, rendered_table in table_replacements.items():
        md = md.replace(token, f"\n\n{rendered_table}\n\n")
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n"


def extract_page_metadata(soup: BeautifulSoup, api: dict[str, Any] | None, seed: Seed) -> tuple[str, str, str, str]:
    if api:
        title = BeautifulSoup(api.get("title", {}).get("rendered", ""), "html.parser").get_text(" ", strip=True)
        published = api.get("date", "")
        updated = api.get("modified", "")
        content_html = api.get("content", {}).get("rendered", "")
        if title and content_html:
            return title, published, updated, content_html
    og_title = soup.find("meta", property="og:title")
    h1 = soup.find("h1")
    title = (og_title.get("content", "") if og_title else "") or (h1.get_text(" ", strip=True) if h1 else "")
    pub = soup.find("meta", property="article:published_time")
    mod = soup.find("meta", property="article:modified_time")
    return title or urlparse(seed.url).path.strip("/").split("/")[-1], pub.get("content", "") if pub else "", mod.get("content", "") if mod else "", select_content_html(soup)


def discover_program_seeds(session: requests.Session) -> list[Seed]:
    found: dict[str, Seed] = {url: Seed(url, "programs", "official_web") for url in PROGRAM_URLS}
    # Program landing pages are WordPress Pages (often created before 2026 but
    # revised for this intake). Posts are news/announcements and are excluded.
    for rest_type in ("pages",):
        base = f"https://tuyensinh.ptit.edu.vn/wp-json/wp/v2/{rest_type}"
        page = 1
        while True:
            response = request(session, f"{base}?per_page=100&page={page}&_fields=link,date,modified,title,slug")
            items = response.json()
            for item in items:
                title = BeautifulSoup(item.get("title", {}).get("rendered", ""), "html.parser").get_text(" ", strip=True)
                normalized = title.casefold()
                is_program = any(term in normalized for term in PROGRAM_TITLE_TERMS)
                excluded = any(term in normalized for term in PROGRAM_EXCLUDE_TERMS)
                if is_program and not excluded:
                    found[item["link"]] = Seed(item["link"], "programs", "official_web")
            total_pages = int(response.headers.get("X-WP-TotalPages", "1"))
            if page >= total_pages:
                break
            page += 1
    return sorted(found.values(), key=lambda s: s.url)


def make_doc_id(seed: Seed, slug: str, version: int, page: int | None = None) -> str:
    doc = f"ptit-{seed.admission_year}-{seed.category}-{slugify(slug)}-v{version}"
    return f"{doc}-p{page:03d}" if page is not None else doc


def markdown_table(table: list[list[Any]]) -> str:
    rows = []
    width = max((len(row) for row in table), default=0)
    if width == 0:
        return ""
    for row in table:
        cells = []
        for cell in list(row) + [""] * (width - len(row)):
            value = re.sub(r"\s+", " ", str(cell or "")).strip().replace("|", "\\|")
            cells.append(value)
        rows.append(cells)
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def extract_pdf_pages(
    raw_path: Path,
    url: str,
    title: str,
    seed: Seed,
    published_at: str,
    updated_at: str,
    parent_doc_id: str,
    version: int,
) -> list[dict[str, Any]]:
    records = []
    relative_raw = raw_path.relative_to(ROOT).as_posix()
    try:
        with pdfplumber.open(raw_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text_body = (page.extract_text(x_tolerance=2, y_tolerance=3) or "").strip()
                tables = page.extract_tables() or []
                sections = [f"# {title}", f"## Trang {page_number}"]
                if text_body:
                    sections.append(text_body)
                for idx, table in enumerate(tables, start=1):
                    rendered = markdown_table(table)
                    if rendered:
                        sections.extend([f"### Bảng {idx}", rendered])
                body = "\n\n".join(sections).strip() + "\n"
                extraction_status = "ok" if text_body or tables else "no_text_detected"
                digest = sha256_bytes(body.encode("utf-8"))
                doc_id = make_doc_id(seed, slugify(title), version, page_number)
                if extraction_status == "no_text_detected":
                    records.append({
                        "doc_id": doc_id,
                        "page_number": page_number,
                        "extraction_status": extraction_status,
                        "content_hash": digest,
                        "raw_file": relative_raw,
                    })
                    continue
                fields = {
                    "doc_id": doc_id,
                    "title": f"{title} — trang {page_number}",
                    "source_url": url,
                    "source_domain": urlparse(url).netloc,
                    "source_type": "official_pdf",
                    "published_at": published_at,
                    "updated_at": updated_at,
                    "retrieved_at": RETRIEVED_AT,
                    "academic_year": seed.academic_year,
                    "admission_year": seed.admission_year,
                    "department": seed.department,
                    "audience": seed.audience,
                    "language": "vi",
                    "content_hash": digest,
                    "raw_file": relative_raw,
                    "page_number": page_number,
                    "parent_doc_id": parent_doc_id,
                    "section": f"Trang {page_number}",
                    "version": version,
                    "extraction_status": extraction_status,
                }
                target = PROCESSED_ROOT / seed.category / f"{doc_id}.md"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(frontmatter(fields) + body, encoding="utf-8", newline="\n")
                records.append({"doc_id": doc_id, "processed_file": target.relative_to(ROOT).as_posix(), "page_number": page_number, "extraction_status": extraction_status, "content_hash": digest})
    except Exception as exc:  # Corrupt/scanned PDFs remain in raw and are reported.
        records.append({"error": f"PDF extraction failed: {type(exc).__name__}: {exc}"})
    return records


def collect_one(session: requests.Session, seed: Seed, skip_attachments: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    record: dict[str, Any] = {"source_url": seed.url, "retrieved_at": started, "category": seed.category}
    processed_records: list[dict[str, Any]] = []
    try:
        response = request(session, seed.url)
        record.update({
            "final_url": response.url,
            "http_status": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "etag": response.headers.get("ETag", ""),
            "last_modified": response.headers.get("Last-Modified", ""),
            "response_sha256": sha256_bytes(response.content),
        })
        soup = BeautifulSoup(response.content, "html.parser")
        api_url = find_wp_api(soup, response.url)
        api_data = None
        if api_url:
            try:
                api_response = request(session, api_url)
                api_data = api_response.json()
                record["wp_api_url"] = api_url
            except Exception as exc:
                record["wp_api_error"] = f"{type(exc).__name__}: {exc}"
        title, published_at, updated_at, content_html = extract_page_metadata(soup, api_data, seed)
        slug = urlparse(response.url).path.strip("/").split("/")[-1]
        raw_name = f"{seed.admission_year}_{slugify(slug)}.html"
        raw_path, version, created = versioned_path(RAW_ROOT / "html" / raw_name, response.content)
        if created:
            raw_path.write_bytes(response.content)
        cleaned_html, links, source_stats = clean_content_html(content_html, response.url)
        body = html_to_md(cleaned_html)
        pdf_links = [u for u in links if re.search(r"\.pdf(?:$|[?#])", u, re.I)]
        if pdf_links:
            body += "\n## Tệp PDF đính kèm\n\n" + "\n".join(f"- [{Path(urlparse(u).path).name}]({u})" for u in pdf_links) + "\n"
        digest = sha256_bytes(body.encode("utf-8"))
        doc_id = make_doc_id(seed, slug, version)
        fields = {
            "doc_id": doc_id,
            "title": title,
            "source_url": response.url,
            "source_domain": urlparse(response.url).netloc,
            "source_type": seed.source_type,
            "published_at": published_at,
            "updated_at": updated_at,
            "retrieved_at": RETRIEVED_AT,
            "academic_year": seed.academic_year,
            "admission_year": seed.admission_year,
            "department": seed.department,
            "audience": seed.audience,
            "language": "vi",
            "content_hash": digest,
            "raw_file": raw_path.relative_to(ROOT).as_posix(),
            "section": "Toàn văn",
            "supersedes": seed.supersedes,
            "duplicate_of": "",
            "version": version,
            "extraction_status": "ok" if len(body) >= 80 else "insufficient_content",
        }
        target = PROCESSED_ROOT / seed.category / f"{doc_id}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(frontmatter(fields) + body, encoding="utf-8", newline="\n")
        md_table_count = sum(1 for line in body.splitlines() if re.fullmatch(r"\|(?:\s*:?-+:?\s*\|)+", line))
        md_numeric_count = len(re.findall(r"(?<!\w)\d+(?:[.,:/-]\d+)*(?!\w)", body))
        record.update({
            "title": title,
            "published_at": published_at,
            "updated_at": updated_at,
            "raw_file": raw_path.relative_to(ROOT).as_posix(),
            "raw_version": version,
            "raw_created": created,
            "processed_file": target.relative_to(ROOT).as_posix(),
            "doc_id": doc_id,
            "content_hash": digest,
            "links_count": len(links),
            "pdf_links": pdf_links,
            **source_stats,
            "markdown_table_count": md_table_count,
            "markdown_numeric_token_count": md_numeric_count,
            "note": seed.note,
            "supersedes": seed.supersedes,
        })
        processed_records.append({"doc_id": doc_id, "processed_file": target.relative_to(ROOT).as_posix(), "content_hash": digest, "source_url": response.url})

        # The official admissions information embeds tuition in a larger page.
        # Preserve an exact section-only derivative so tuition retrieval does not
        # depend on a very large admission-plan chunk.
        if seed.admission_year == "2026" and seed.category == "admissions" and "thong-tin-tuyen-sinh-dai-hoc-chinh-quy" in slug:
            tuition_match = re.search(
                r"(?ms)^(#{1,6}\s+[^\n]*Học phí dự kiến với sinh viên chính quy:[^\n]*\n.*?)(?=^#{1,6}\s+[^\n]*10\\?\.2|\Z)",
                body,
            )
            if tuition_match:
                tuition_body = tuition_match.group(1).strip() + "\n"
                tuition_hash = sha256_bytes(tuition_body.encode("utf-8"))
                tuition_doc_id = make_doc_id(Seed(response.url, "tuition"), f"{slug}-hoc-phi", version)
                tuition_fields = dict(fields)
                tuition_fields.update({
                    "doc_id": tuition_doc_id,
                    "title": f"{title} — Học phí dự kiến",
                    "source_type": "official_web",
                    "content_hash": tuition_hash,
                    "section": "10.1. Học phí dự kiến với sinh viên chính quy",
                    "parent_doc_id": doc_id,
                    "supersedes": "",
                })
                tuition_target = PROCESSED_ROOT / "tuition" / f"{tuition_doc_id}.md"
                tuition_target.parent.mkdir(parents=True, exist_ok=True)
                tuition_target.write_text(frontmatter(tuition_fields) + tuition_body, encoding="utf-8", newline="\n")
                processed_records.append({"doc_id": tuition_doc_id, "processed_file": tuition_target.relative_to(ROOT).as_posix(), "content_hash": tuition_hash, "source_url": response.url})

        for pdf_index, pdf_url in enumerate([] if skip_attachments else pdf_links, start=1):
            try:
                pdf_response = request(session, pdf_url)
                pdf_name = Path(urlparse(pdf_response.url).path).name or f"attachment-{pdf_index}.pdf"
                pdf_base = RAW_ROOT / "pdf" / f"{seed.admission_year}_{slugify(slug)}__{slugify(Path(pdf_name).stem)}.pdf"
                pdf_path, pdf_version, pdf_created = versioned_path(pdf_base, pdf_response.content)
                if pdf_created:
                    pdf_path.write_bytes(pdf_response.content)
                pdf_meta = {
                    "source_url": pdf_url,
                    "final_url": pdf_response.url,
                    "parent_source_url": response.url,
                    "retrieved_at": RETRIEVED_AT,
                    "http_status": pdf_response.status_code,
                    "content_type": pdf_response.headers.get("Content-Type", ""),
                    "title": f"{title} — {pdf_name}",
                    "published_at": published_at,
                    "updated_at": updated_at,
                    "raw_file": pdf_path.relative_to(ROOT).as_posix(),
                    "response_sha256": sha256_bytes(pdf_response.content),
                    "raw_version": pdf_version,
                    "raw_created": pdf_created,
                    "category": seed.category,
                }
                record.setdefault("attachments", []).append(pdf_meta)
                page_records = extract_pdf_pages(pdf_path, pdf_response.url, pdf_meta["title"], seed, published_at, updated_at, doc_id, pdf_version)
                pdf_meta["page_extraction"] = page_records
                processed_records.extend(item for item in page_records if item.get("processed_file"))
            except Exception as exc:
                record.setdefault("attachment_errors", []).append({"url": pdf_url, "error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:
        record.update({"http_status": getattr(getattr(exc, "response", None), "status_code", None), "error": f"{type(exc).__name__}: {exc}"})
    return record, processed_records


def parse_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    block = text.split("---", 2)[1]
    result = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        try:
            result[key] = json.loads(value)
        except Exception:
            result[key] = value
    return result


def run_quality_checks(source_records: list[dict[str, Any]], excluded_no_text: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    md_files = sorted(PROCESSED_ROOT.glob("**/*.md"))
    metadata_errors = []
    missing_raw = []
    content_hash_mismatches = []
    raw_hash_mismatches = []
    malformed_markdown_tables = []
    hashes: dict[str, list[str]] = {}
    no_text_pages: list[Any] = []
    processed_pdf_pages: set[tuple[str, int]] = set()
    for path in md_files:
        complete_text = path.read_text(encoding="utf-8")
        fm = parse_frontmatter(path)
        missing = sorted(REQUIRED_FRONTMATTER - set(fm))
        if missing:
            metadata_errors.append({"file": path.relative_to(ROOT).as_posix(), "missing_keys": missing})
        raw_file = fm.get("raw_file", "")
        if not raw_file or not (ROOT / raw_file).exists():
            missing_raw.append(path.relative_to(ROOT).as_posix())
        body = complete_text.split("---", 2)[2].lstrip("\n") if complete_text.startswith("---\n") else complete_text
        if fm.get("content_hash") and sha256_bytes(body.encode("utf-8")) != fm.get("content_hash"):
            content_hash_mismatches.append(path.relative_to(ROOT).as_posix())
        table_width = None
        for line_number, line in enumerate(body.splitlines(), start=1):
            if not line.startswith("|"):
                table_width = None
                continue
            width = len(re.findall(r"(?<!\\)\|", line))
            if table_width is None:
                table_width = width
            elif width != table_width:
                malformed_markdown_tables.append({
                    "file": path.relative_to(ROOT).as_posix(),
                    "line": line_number,
                    "expected_pipe_count": table_width,
                    "actual_pipe_count": width,
                })
        hashes.setdefault(str(fm.get("content_hash", "")), []).append(str(fm.get("doc_id", path.stem)))
        if fm.get("extraction_status") == "no_text_detected":
            no_text_pages.append(str(fm.get("doc_id", path.stem)))
        if fm.get("source_type") == "official_pdf" and fm.get("raw_file") and str(fm.get("page_number", "")).isdigit():
            processed_pdf_pages.add((str(fm["raw_file"]), int(fm["page_number"])))
    pdf_audit_errors = []
    audited_raw_files = set()
    for record in source_records:
        for attachment in record.get("attachments", []):
            raw_file = str(attachment.get("raw_file", ""))
            if not raw_file or raw_file in audited_raw_files:
                continue
            audited_raw_files.add(raw_file)
            try:
                with pdfplumber.open(ROOT / raw_file) as pdf:
                    for page_number in range(1, len(pdf.pages) + 1):
                        if (raw_file, page_number) not in processed_pdf_pages:
                            no_text_pages.append({
                                "source_url": attachment.get("final_url", attachment.get("source_url", "")),
                                "raw_file": raw_file,
                                "page_number": page_number,
                            })
            except Exception as exc:
                pdf_audit_errors.append({"raw_file": raw_file, "error": f"{type(exc).__name__}: {exc}"})
    for record in source_records:
        raw_file = record.get("raw_file")
        expected_hash = record.get("response_sha256")
        if raw_file and expected_hash and (ROOT / raw_file).exists() and sha256_bytes((ROOT / raw_file).read_bytes()) != expected_hash:
            raw_hash_mismatches.append(raw_file)
        for attachment in record.get("attachments", []):
            raw_file = attachment.get("raw_file")
            expected_hash = attachment.get("response_sha256")
            if raw_file and expected_hash and (ROOT / raw_file).exists() and sha256_bytes((ROOT / raw_file).read_bytes()) != expected_hash:
                raw_hash_mismatches.append(raw_file)
    duplicates = [docs for digest, docs in hashes.items() if digest and len(docs) > 1]
    urls_ok = sum(1 for r in source_records if r.get("http_status") == 200)
    urls_failed = [{"url": r["source_url"], "status": r.get("http_status"), "error": r.get("error", "")} for r in source_records if r.get("http_status") != 200]
    table_warnings = []
    numeric_warnings = []
    for record in source_records:
        if record.get("source_table_count", 0) > record.get("markdown_table_count", 0):
            table_warnings.append({"doc_id": record.get("doc_id"), "source_tables": record.get("source_table_count"), "markdown_tables": record.get("markdown_table_count")})
        source_nums = record.get("source_numeric_token_count", 0)
        md_nums = record.get("markdown_numeric_token_count", 0)
        if source_nums and md_nums < source_nums * 0.9:
            numeric_warnings.append({"doc_id": record.get("doc_id"), "source_numeric_tokens": source_nums, "markdown_numeric_tokens": md_nums})
    return {
        "checked_at": RETRIEVED_AT,
        "url_checks": {"ok": urls_ok, "failed": urls_failed},
        "raw_processed_mapping": {"checked_markdown_files": len(md_files), "missing_raw": missing_raw},
        "frontmatter": {"required_keys": sorted(REQUIRED_FRONTMATTER), "errors": metadata_errors},
        "hash_integrity": {"content_hash_mismatches": content_hash_mismatches, "raw_hash_mismatches": sorted(set(raw_hash_mismatches))},
        "content_preservation": {
            "table_warnings": table_warnings,
            "numeric_warnings": numeric_warnings,
            "pdf_pages_without_processed_text": no_text_pages,
            "pdf_audit_errors": pdf_audit_errors,
            "malformed_markdown_tables": malformed_markdown_tables,
        },
        "duplicates_by_exact_processed_hash": duplicates,
        "superseded_records": [{"doc_id": r.get("doc_id"), "supersedes": r.get("supersedes")} for r in source_records if r.get("supersedes")],
    }


def remove_unusable_processed_pages() -> list[dict[str, Any]]:
    """Remove collector-created placeholder Markdown for image-only PDF pages."""
    removed: list[dict[str, Any]] = []
    processed_resolved = PROCESSED_ROOT.resolve()
    for path in PROCESSED_ROOT.glob("**/*.md"):
        fm = parse_frontmatter(path)
        if fm.get("extraction_status") != "no_text_detected":
            continue
        resolved = path.resolve()
        if processed_resolved not in resolved.parents:
            raise RuntimeError(f"Refusing to remove file outside processed root: {resolved}")
        path.unlink()
        removed.append({
            "source_url": fm.get("source_url", ""),
            "raw_file": fm.get("raw_file", ""),
            "page_number": fm.get("page_number", ""),
        })
    return removed


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--skip-programs", action="store_true")
    parser.add_argument("--only-url", action="append", default=[], help="Crawl only this manifest URL and merge with existing metadata")
    parser.add_argument("--only-tables", action="store_true", help="Reprocess only sources whose HTML contains tables")
    parser.add_argument("--skip-attachments", action="store_true", help="Do not redownload/re-extract linked PDFs")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "vi,en;q=0.7"})
    program_seeds: list[Seed] = []
    if not args.skip_programs:
        try:
            program_seeds = discover_program_seeds(session)
        except Exception as exc:
            print(f"Program discovery failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    if args.discover_only:
        print(json.dumps([asdict(s) for s in program_seeds], ensure_ascii=False, indent=2))
        return 0

    for folder in [RAW_ROOT / "html", RAW_ROOT / "pdf"]:
        folder.mkdir(parents=True, exist_ok=True)
    for folder in ["admissions", "programs", "tuition", "scholarships", "regulations", "announcements"]:
        (PROCESSED_ROOT / folder).mkdir(parents=True, exist_ok=True)

    all_seeds = SEEDS + program_seeds
    # Preserve manifest order while removing duplicate URLs.
    unique_seeds = list({seed.url: seed for seed in all_seeds}.values())
    requested = set(args.only_url)
    if args.only_tables:
        if not (RAW_ROOT / "metadata.json").exists():
            raise SystemExit("--only-tables requires an existing metadata.json")
        existing_metadata = json.loads((RAW_ROOT / "metadata.json").read_text(encoding="utf-8"))
        requested.update(
            record.get("source_url") for record in existing_metadata.get("sources", [])
            if record.get("source_table_count", 0) > 0
        )
    partial_mode = bool(requested)
    if partial_mode:
        unique_seeds = [seed for seed in unique_seeds if seed.url in requested]
        missing_requested = sorted(requested - {seed.url for seed in unique_seeds})
        if missing_requested:
            raise SystemExit(f"Requested URL is not in manifest: {missing_requested}")
    source_records: list[dict[str, Any]] = []
    processed_records: list[dict[str, Any]] = []
    for index, seed in enumerate(unique_seeds, start=1):
        print(f"[{index}/{len(unique_seeds)}] {seed.url}", flush=True)
        record, processed = collect_one(session, seed, args.skip_attachments)
        source_records.append(record)
        processed_records.extend(processed)

    if partial_mode and (RAW_ROOT / "metadata.json").exists():
        previous = json.loads((RAW_ROOT / "metadata.json").read_text(encoding="utf-8"))
        previous_by_url = {record.get("source_url"): record for record in previous.get("sources", [])}
        for record in source_records:
            old = previous_by_url.get(record.get("source_url"), {})
            for key in ("attachments", "attachment_errors"):
                if key not in record and key in old:
                    record[key] = old[key]
        retained = [
            record for record in previous.get("sources", [])
            if record.get("http_status") == 200 and record.get("source_url") not in requested
        ]
        source_records = retained + source_records

    # Mark exact duplicate top-level HTML documents without modifying historical files.
    first_by_hash: dict[str, str] = {}
    for record in source_records:
        digest = record.get("content_hash")
        if not digest or not record.get("processed_file"):
            continue
        if digest in first_by_hash:
            record["duplicate_of"] = first_by_hash[digest]
            path = ROOT / record["processed_file"]
            if path.exists():
                text_value = path.read_text(encoding="utf-8")
                text_value = re.sub(
                    r"(?m)^duplicate_of:.*$",
                    f"duplicate_of: {yaml_string(record['duplicate_of'])}",
                    text_value,
                    count=1,
                )
                path.write_text(text_value, encoding="utf-8", newline="\n")
        else:
            first_by_hash[digest] = record.get("doc_id", "")

    removed_no_text = remove_unusable_processed_pages()
    actual_processed_count = len(list(PROCESSED_ROOT.glob("**/*.md")))
    qa = run_quality_checks(source_records, removed_no_text)
    excluded_page_count = len(qa["content_preservation"]["pdf_pages_without_processed_text"])
    metadata = {
        "dataset": "PTIT official undergraduate admissions",
        "scope": {
            "current_admission_year": 2026,
            "campuses": ["BVH - Hà Nội", "BVS - TP.HCM"],
            "program_types": ["đại trà", "chất lượng cao", "đặc thù", "liên kết quốc tế"],
            "historical_cutoff_years": [2023, 2024, 2025],
            "source_policy": "Official PTIT domains only; no secondary sources in this collection.",
        },
        "retrieved_at": RETRIEVED_AT,
        "collector": "scripts/collect_ptit_admissions.py",
        "source_count": len(source_records),
        "processed_document_count": actual_processed_count,
        "excluded_image_only_pdf_pages": excluded_page_count,
        "sources": source_records,
    }
    metadata_path = RAW_ROOT / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    qa_path = PROCESSED_ROOT / "quality_report.json"
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({
        "sources": len(source_records),
        "program_sources": len(program_seeds),
        "processed_documents": actual_processed_count,
        "excluded_image_only_pdf_pages": excluded_page_count,
        "failed_urls": len(qa["url_checks"]["failed"]),
        "metadata_errors": len(qa["frontmatter"]["errors"]),
        "missing_raw": len(qa["raw_processed_mapping"]["missing_raw"]),
        "duplicates": len(qa["duplicates_by_exact_processed_hash"]),
        "metadata": metadata_path.relative_to(ROOT).as_posix(),
        "quality_report": qa_path.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
