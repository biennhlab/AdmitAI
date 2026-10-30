# AdmitAI — System Architecture

> Kiến trúc hệ thống và tech stack chi tiết.  
> Xem yêu cầu: [01-requirements.md](./01-requirements.md)

---

## 1. Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENTS                                    │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐   │
│  │  Chatbot Web (HTML/JS)│   │  Admin Dashboard (HTML/JS)       │   │
│  │  (Thí sinh/Phụ huynh) │   │  (Nhân viên tuyển sinh)          │   │
│  └──────────┬───────────┘    └──────────────┬───────────────────┘   │
└─────────────┼───────────────────────────────┼───────────────────────┘
              │ HTTP/REST                     │ HTTP/REST
              ▼                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                               │
│  ┌───────────────┐ ┌──────────────┐ ┌────────────────────────────┐ │
│  │  Chat API     │ │  Admin API   │ │  Auth Middleware           │ │
│  │  /api/chat    │ │  /api/admin  │ │  (JWT / API Key)           │ │
│  └───────┬───────┘ └──────┬───────┘ └────────────────────────────┘ │
│          │                │                                         │
│  ┌───────▼────────────────▼─────────────────────────────────────┐  │
│  │                    Core RAG Pipeline                          │  │
│  │  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌───────────────────┐ │  │
│  │  │ Router  │→│ Query    │→│Retrieval│→│ Reranker          │ │  │
│  │  │(Claude  │ │Transform │ │(Hybrid) │ │(bge-reranker-v2)  │ │  │
│  │  │tool use)│ │(HyDE/    │ │         │ │                   │ │  │
│  │  │         │ │ rewrite) │ │         │ │                   │ │  │
│  │  └────┬────┘ └──────────┘ └────┬────┘ └────────┬──────────┘ │  │
│  │       │                        │                │            │  │
│  │       ▼                   ┌────▼────┐    ┌──────▼──────┐     │  │
│  │  ┌─────────┐              │Dense    │    │ Generation  │     │  │
│  │  │Calculat.│              │(Qdrant) │    │ (Claude API)│     │  │
│  │  │Tool     │              ├─────────┤    │ + Self-RAG  │     │  │
│  │  └─────────┘              │Sparse   │    │ + Citation  │     │  │
│  │                           │(BM25)   │    └─────────────┘     │  │
│  │                           ├─────────┤                        │  │
│  │                           │RRF      │                        │  │
│  │                           │Fusion   │                        │  │
│  │                           └─────────┘                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Data Layer                                │  │
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐ │  │
│  │  │ Qdrant   │  │ SQLite/      │  │ BM25 Index             │ │  │
│  │  │(vectors) │  │ PostgreSQL   │  │ (rank_bm25 in-memory)  │ │  │
│  │  │          │  │(chat logs,   │  │                        │ │  │
│  │  │          │  │ users, docs) │  │                        │ │  │
│  │  └──────────┘  └──────────────┘  └────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tech Stack chi tiết

### 2.1 Backend & API

| Thành phần | Lựa chọn | Lý do (từ đặc tả) |
|---|---|---|
| **Backend framework** | FastAPI | Chuẩn công nghiệp cho AI service, async, tự động sinh OpenAPI docs |
| **Ngôn ngữ** | Python 3.10+ | Bắt buộc — ecosystem RAG/eval (RAGAS, sentence-transformers) |
| **Phương án nhẹ hơn** | Flask | Nếu muốn đơn giản hơn |

### 2.2 Document Processing

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| **PDF text extraction** | `pdfplumber` | Tách riêng logic đọc bảng vs văn bản |
| **Table extraction** | `camelot` hoặc `pdfplumber.extract_tables()` | Table-aware chunking (Module 1) |
| **Chunking** | Tự code | Phần học quan trọng nhất — structure-aware/table-aware/parent-child |
| **Phương án nhẹ hơn** | `unstructured` library | Nếu PDF quá phức tạp |

### 2.3 Embedding & Retrieval

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| **Embedding model** | `BAAI/bge-m3` qua `sentence-transformers` | Đa ngôn ngữ, hỗ trợ tiếng Việt tốt, chạy local miễn phí |
| **Phương án nhẹ hơn** | `BAAI/bge-small-en-v1.5` | Nhẹ hơn nhưng kém tiếng Việt |
| **Vector store** | Qdrant (Docker) | Vector DB công nghiệp, metadata filtering native |
| **Phương án nhẹ hơn** | ChromaDB (embedded) | Không cần Docker |
| **Sparse search** | `rank_bm25` | Nhẹ, đủ học nguyên lý fusion |
| **Phương án nhẹ hơn** | Elasticsearch/OpenSearch | Nếu muốn hạ tầng search thật |
| **Fusion (RRF)** | Tự code (~10 dòng) | Thuật toán đơn giản, hiểu bản chất |

### 2.4 Reranking & Generation

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| **Reranker** | `BAAI/bge-reranker-v2-m3` qua CrossEncoder | Miễn phí, chạy local, hiểu cross-encoder vs bi-encoder |
| **Phương án nhẹ hơn** | Cohere Rerank API | Phiên bản managed |
| **LLM generation** | Claude API (Haiku test / Sonnet demo) | Tool use tốt cho agentic RAG |
| **Phương án nhẹ hơn** | OpenAI GPT-4o-mini | Nếu đã quen |

### 2.5 Agentic & Routing

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| **Query transformation** | Gọi LLM API trực tiếp, tự code prompt | Không cần thư viện riêng |
| **Router / Agentic** | Claude tool use / function calling native | Chuẩn công nghiệp, không cần LangGraph |
| **Phương án nhẹ hơn** | LangGraph | Nếu muốn học agent orchestration graph-based |

### 2.6 Evaluation & Tracking

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| **Evaluation** | RAGAS (`ragas` package) | Chuẩn công nghiệp: Context Precision/Recall, Faithfulness, Answer Relevancy |
| **Experiment tracking** | JSON/CSV + markdown, hoặc Weights & Biases (free tier) | W&B là MLOps thật, điểm cộng CV |

### 2.7 Infrastructure

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| **Containerization** | Docker Compose (backend + Qdrant) | Chuẩn deploy thật, dễ reproduce |
| **Frontend** | HTML/JS thuần | Tận dụng kỹ năng có sẵn, trọng tâm ở backend |
| **Version control** | Git (mỗi module = 1 branch/tag) | Thể hiện tiến trình cải thiện trên GitHub |

---

## 3. Cấu trúc thư mục

```
AdmitAI/
├── data/
│   ├── raw/                 # PDF gốc, file nguồn tuyển sinh
│   └── processed/           # JSON đã chunk theo từng module
├── src/
│   ├── ingestion/           # parsing, chunking (module 0-1)
│   ├── retrieval/           # dense, bm25, fusion, rerank (module 2, 4)
│   ├── query_transform/     # rewrite, HyDE, multi-query (module 3)
│   ├── generation/          # prompt, citation, faithfulness check (module 5)
│   ├── routing/             # agentic router + calculator tool (module 6)
│   └── eval/                # golden dataset, RAGAS runner (module 7)
├── eval_results/            # kết quả metric từng module (json/csv)
├── api/                     # FastAPI app
├── frontend/
│   ├── chatbot/             # HTML/JS chatbot cho thí sinh
│   └── dashboard/           # HTML/JS admin dashboard
├── docs/                    # tài liệu project (bạn đang đọc)
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 4. Luồng dữ liệu chính

### 4.1 Ingestion Pipeline (offline)

```
PDF/Web sources
    │
    ▼
Document Parsing (pdfplumber/camelot)
    │
    ▼
Structure-aware Chunking (tự code)
    │
    ├──► Embedding (bge-m3) ──► Qdrant (dense index)
    │
    └──► Tokenize ──► BM25 Index (rank_bm25, in-memory)
```

### 4.2 Query Pipeline (online)

```
User Question (tiếng Việt)
    │
    ▼
Router (Claude tool use)
    ├── RAG path ──► Query Transform ──► Hybrid Retrieval ──► Rerank ──► Generate + Self-RAG ──► Response
    ├── Calculator path ──► Calculator Tool ──► Response
    └── Out-of-scope ──► Fallback message + Log escalation
```

> Chi tiết pipeline: [06-ai-rag-pipeline.md](./06-ai-rag-pipeline.md) & [07-agentic-rag-tools.md](./07-agentic-rag-tools.md)

---

## 5. Giao tiếp giữa các thành phần

| From | To | Protocol | Mô tả |
|---|---|---|---|
| Chatbot frontend | FastAPI | HTTP REST (JSON) | Gửi câu hỏi, nhận câu trả lời |
| Dashboard frontend | FastAPI | HTTP REST (JSON) | CRUD lịch sử, escalation, quản lý docs |
| FastAPI | Qdrant | gRPC / HTTP | Vector search |
| FastAPI | Claude API | HTTPS | LLM generation, tool use |
| FastAPI | SQLite/PostgreSQL | Direct (SQLAlchemy) | Chat logs, users, document metadata |
| FastAPI | BM25 index | In-memory (Python) | Sparse search |
| FastAPI | sentence-transformers | In-process (Python) | Embedding, reranking |
