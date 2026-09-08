# AdmitAI — Data & Database

> Nguồn dữ liệu RAG, schema database, và chiến lược lưu trữ.  
> Xem kiến trúc: [02-system-architecture.md](./02-system-architecture.md)

---

## 1. Nguồn dữ liệu RAG

### 1.1 Loại nguồn

| Nguồn | Định dạng | Nội dung | Ví dụ |
|---|---|---|---|
| Đề án tuyển sinh | PDF | Chỉ tiêu, phương thức, tổ hợp xét tuyển, điều kiện | Đề án tuyển sinh PTIT 2024-2025 |
| Thông tin học phí | PDF / Web | Bảng học phí theo ngành, theo năm | Quyết định học phí PTIT |
| Chương trình đào tạo | PDF / Web | Danh sách ngành, mô tả ngành, chương trình | CTDT các khoa |
| Điểm chuẩn | PDF / Web | Điểm chuẩn qua các năm, theo phương thức | Điểm chuẩn 2020-2024 |
| FAQ tuyển sinh | Web / Text | Câu hỏi thường gặp và câu trả lời | FAQ từ website PTIT |
| Thông báo tuyển sinh | Web | Lịch trình, deadline, thay đổi chính sách | Tin tức tuyển sinh |

### 1.2 Đặc điểm dữ liệu

- **Ngôn ngữ**: Tiếng Việt
- **Kích thước corpus**: Nhỏ (ước tính vài chục đến trăm trang)
- **Cập nhật**: Theo năm tuyển sinh (không real-time)
- **Chứa bảng biểu**: Có (điểm chuẩn, học phí, chỉ tiêu) → cần table-aware processing
- **Chứa số liệu**: Có (điểm số, học phí, chỉ tiêu) → cần structured extraction

---

## 2. Data Processing Pipeline

### 2.1 Parsing

```python
# Ý tưởng xử lý (không phải code cuối cùng)

# 1. Text extraction
import pdfplumber
with pdfplumber.open("deanTS.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()       # Văn bản
        tables = page.extract_tables()   # Bảng biểu (riêng biệt)

# 2. Table extraction (nếu cần chính xác hơn)
import camelot
tables = camelot.read_pdf("deanTS.pdf", pages="all")
```

### 2.2 Chunking Strategy

Chunking là phần tự code quan trọng nhất. Chiến lược qua các module:

| Module | Strategy | Mô tả |
|---|---|---|
| Tuần 1 (Naive) | Fixed-size | Split text theo character count + overlap |
| Tuần 2 (Nâng cao) | Structure-aware | Giữ heading, paragraph boundaries |
| Tuần 2 | Table-aware | Bảng biểu là chunk riêng, kèm metadata (tên bảng, context) |
| Tuần 2 | Parent-child | Chunk nhỏ để retrieval, chunk lớn (parent) để generation |

### 2.3 Chunk Schema (JSON)

```json
{
  "chunk_id": "doc001_chunk_015",
  "content": "Ngành Công nghệ thông tin: Chỉ tiêu 500...",
  "metadata": {
    "source_file": "dean_tuyen_sinh_2024.pdf",
    "page_number": 5,
    "chunk_type": "text",
    "heading": "Chỉ tiêu tuyển sinh theo ngành",
    "parent_chunk_id": "doc001_parent_003",
    "created_at": "2024-01-15T10:00:00Z"
  }
}
```

Chunk bảng biểu:

```json
{
  "chunk_id": "doc001_table_003",
  "content": "| Ngành | Điểm chuẩn 2023 | Điểm chuẩn 2024 |\n|---|---|---|\n| CNTT | 26.5 | 27.0 |...",
  "metadata": {
    "source_file": "diem_chuan_2024.pdf",
    "page_number": 2,
    "chunk_type": "table",
    "table_title": "Bảng điểm chuẩn theo ngành 2023-2024",
    "parent_chunk_id": "doc001_parent_001",
    "created_at": "2024-01-15T10:00:00Z"
  }
}
```

---

## 3. Vector Store — Qdrant

### 3.1 Collection schema

```json
{
  "collection_name": "admitai_chunks",
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "payload_schema": {
    "source_file": "keyword",
    "chunk_type": "keyword",
    "heading": "text",
    "page_number": "integer",
    "parent_chunk_id": "keyword",
    "created_at": "datetime"
  }
}
```

> **Note**: Vector size phụ thuộc embedding model. `bge-m3` output 1024 dims, `bge-small-en-v1.5` output 384 dims.

### 3.2 Metadata Filtering

Qdrant hỗ trợ metadata filtering native — cần cho:
- Filter theo `source_file` (chỉ tìm trong tài liệu cụ thể)
- Filter theo `chunk_type` (chỉ tìm bảng / chỉ tìm text)
- Filter theo `page_number` range

---

## 4. BM25 Index

`rank_bm25` chạy in-memory (Python), không cần DB riêng.

```python
from rank_bm25 import BM25Okapi

# Tokenize corpus (tiếng Việt cần word segmentation)
tokenized_corpus = [doc.split() for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

# Search
tokenized_query = query.split()
scores = bm25.get_scores(tokenized_query)
```

> **Lưu ý tiếng Việt**: Cân nhắc dùng thư viện word segmentation (VD: `underthesea`, `pyvi`) để tokenize chính xác hơn cho BM25. Tuy nhiên, split đơn giản cũng hoạt động ở mức cơ bản.

---

## 5. Relational Database — Application Data

### 5.1 Lựa chọn

| Giai đoạn | Lựa chọn | Lý do |
|---|---|---|
| Phát triển & demo | **SQLite** | Zero-config, đủ cho quy mô nhỏ |
| Production (nếu cần) | PostgreSQL | Scale tốt hơn, chạy trong Docker Compose |

### 5.2 Schema

#### Bảng `users` (Nhân viên tuyển sinh)

| Column | Type | Constraint | Mô tả |
|---|---|---|---|
| id | INTEGER | PK, AUTO | ID nhân viên |
| username | VARCHAR(50) | UNIQUE, NOT NULL | Tên đăng nhập |
| password_hash | VARCHAR(255) | NOT NULL | Mật khẩu (bcrypt hash) |
| full_name | VARCHAR(100) | | Họ tên |
| role | VARCHAR(20) | DEFAULT 'staff' | Role (staff / admin) |
| created_at | DATETIME | DEFAULT NOW | Ngày tạo |

#### Bảng `chat_sessions`

| Column | Type | Constraint | Mô tả |
|---|---|---|---|
| id | VARCHAR(36) | PK | UUID session |
| started_at | DATETIME | DEFAULT NOW | Thời điểm bắt đầu |
| ended_at | DATETIME | | Thời điểm kết thúc |

#### Bảng `chat_messages`

| Column | Type | Constraint | Mô tả |
|---|---|---|---|
| id | INTEGER | PK, AUTO | ID message |
| session_id | VARCHAR(36) | FK → chat_sessions.id | Session |
| role | VARCHAR(10) | NOT NULL | 'user' hoặc 'assistant' |
| content | TEXT | NOT NULL | Nội dung tin nhắn |
| citations | JSON | | Danh sách source trích dẫn |
| route_type | VARCHAR(20) | | 'rag' / 'calculator' / 'out_of_scope' |
| created_at | DATETIME | DEFAULT NOW | Thời điểm |

#### Bảng `escalations`

| Column | Type | Constraint | Mô tả |
|---|---|---|---|
| id | INTEGER | PK, AUTO | ID |
| message_id | INTEGER | FK → chat_messages.id | Message gốc |
| status | VARCHAR(20) | DEFAULT 'pending' | 'pending' / 'resolved' |
| staff_reply | TEXT | | Câu trả lời nhân viên |
| resolved_by | INTEGER | FK → users.id | Nhân viên xử lý |
| resolved_at | DATETIME | | Thời điểm xử lý |
| created_at | DATETIME | DEFAULT NOW | |

#### Bảng `documents`

| Column | Type | Constraint | Mô tả |
|---|---|---|---|
| id | INTEGER | PK, AUTO | ID |
| filename | VARCHAR(255) | NOT NULL | Tên file gốc |
| file_path | VARCHAR(500) | NOT NULL | Đường dẫn lưu trữ |
| file_size | INTEGER | | Kích thước (bytes) |
| num_chunks | INTEGER | | Số chunks sau processing |
| status | VARCHAR(20) | DEFAULT 'processing' | 'processing' / 'indexed' / 'error' |
| uploaded_by | INTEGER | FK → users.id | Người upload |
| uploaded_at | DATETIME | DEFAULT NOW | |

### 5.3 ER Diagram

```
┌──────────┐     ┌────────────────┐     ┌──────────────┐
│  users   │     │ chat_sessions  │     │ chat_messages │
├──────────┤     ├────────────────┤     ├──────────────┤
│ id (PK)  │     │ id (PK)        │◄───┤│ session_id   │
│ username │     │ started_at     │     │ id (PK)      │
│ pass_hash│     │ ended_at       │     │ role         │
│ full_name│     └────────────────┘     │ content      │
│ role     │                            │ citations    │
│ created  │◄──────────────────────┐    │ route_type   │
└──────────┘                       │    │ created_at   │
      ▲                            │    └──────┬───────┘
      │                            │           │
      │    ┌──────────────┐        │    ┌──────▼───────┐
      │    │  documents   │        │    │ escalations  │
      │    ├──────────────┤        │    ├──────────────┤
      └────┤ uploaded_by  │        └────┤ resolved_by  │
           │ id (PK)      │             │ id (PK)      │
           │ filename     │             │ message_id   │
           │ file_path    │             │ status       │
           │ num_chunks   │             │ staff_reply  │
           │ status       │             │ resolved_at  │
           └──────────────┘             └──────────────┘
```

---

## 6. Lưu trữ File

| Loại file | Vị trí | Mô tả |
|---|---|---|
| PDF gốc | `data/raw/` | File nguồn chưa xử lý |
| Chunks (JSON) | `data/processed/` | Chunks đã parse xong, 1 file JSON per source document |
| Uploaded PDFs | `data/raw/` | PDF nhân viên upload qua dashboard |
| Eval results | `eval_results/` | Metric RAGAS mỗi module (JSON/CSV) |
