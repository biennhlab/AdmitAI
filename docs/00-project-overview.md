# AdmitAI — Project Overview

> **Tên project**: AdmitAI  
> **Mục tiêu**: Xây dựng hệ thống chatbot RAG (Retrieval-Augmented Generation) tư vấn tuyển sinh cho Học viện Công nghệ Bưu chính Viễn thông (PTIT).

---

## 1. Bối cảnh

Thí sinh và phụ huynh cần tra cứu thông tin tuyển sinh PTIT từ nhiều nguồn phân tán (PDF đề án, website, FAQ). AdmitAI tổng hợp các nguồn này vào một hệ thống RAG thông minh, cho phép hỏi-đáp bằng tiếng Việt với câu trả lời chính xác, có trích dẫn nguồn.

## 2. Mục tiêu kép

### 2.1 Sản phẩm thực tế
- Chatbot tư vấn tuyển sinh cho **thí sinh/phụ huynh**
- Dashboard quản trị cho **nhân viên tuyển sinh** (xem lịch sử, xử lý fallback, quản lý dữ liệu RAG)

### 2.2 Learning project
- Học sâu cơ chế RAG qua **8 module/tuần** (Naive RAG → Agentic RAG → Evaluation)
- Tự code phần cốt lõi (chunking, retrieval, fusion, rerank), chỉ dùng thư viện cho hạ tầng
- Xem chi tiết: [12-learning-roadmap.md](./12-learning-roadmap.md)

## 3. Scope

### Trong scope
| Hạng mục | Mô tả |
|---|---|
| Chatbot RAG | Hỏi-đáp tuyển sinh PTIT bằng tiếng Việt, có citation |
| Agentic RAG | Router phân loại câu hỏi + calculator tính điểm tổ hợp/xét tuyển |
| Admin Dashboard | Xem lịch sử câu hỏi, xử lý escalation, quản lý dữ liệu RAG |
| Hybrid Search | Dense (embedding) + Sparse (BM25) + RRF fusion |
| Reranking | Cross-encoder reranker |
| Self-RAG/Corrective RAG | Tự đánh giá chất lượng retrieval và generation |
| Evaluation | RAGAS metrics qua từng module |
| Containerization | Docker Compose (backend + Qdrant) |

### Ngoài scope
- Mobile app native
- Hệ thống đăng ký xét tuyển trực tuyến
- Tích hợp trực tiếp vào hệ thống IT nội bộ PTIT
- Multi-tenant (nhiều trường khác nhau)

## 4. Actors

| Actor | Mô tả | Giao diện |
|---|---|---|
| **Thí sinh / Phụ huynh** | Hỏi thông tin tuyển sinh, điểm chuẩn, học phí, tổ hợp xét tuyển... | Chatbot web |
| **Nhân viên tuyển sinh** | Xem lịch sử, trả lời câu hỏi bot không xử lý được, quản lý dữ liệu RAG | Admin dashboard |

## 5. Nguyên tắc kỹ thuật

Từ đặc tả gốc, project tuân thủ các nguyên tắc sau:

1. **Tự code phần cốt lõi** — Không dùng LangChain/LlamaIndex như hộp đen; tự viết chunking, retrieval logic, fusion, rerank để hiểu bản chất
2. **Python là ngôn ngữ chính** — Bắt buộc vì ecosystem RAG/eval (RAGAS, sentence-transformers, rank-bm25)
3. **Giới thiệu tech tăng dần** — Không setup hết stack ngay tuần 1; thêm công cụ khi đã hiểu tại sao cần
4. **Ưu tiên model local miễn phí** — Embedding & reranker chạy local (bge-small, bge-reranker); LLM dùng model rẻ khi test
5. **Frontend đơn giản** — HTML/JS thuần gọi FastAPI, trọng tâm học nằm ở backend

## 6. Tech Stack tổng quan

> Chi tiết đầy đủ: [02-system-architecture.md](./02-system-architecture.md)

| Layer | Công nghệ |
|---|---|
| Backend API | FastAPI (Python) |
| Document Parsing | pdfplumber, camelot |
| Embedding | BAAI/bge-m3 (multilingual, local) |
| Vector Store | Qdrant (Docker) |
| Sparse Search | rank_bm25 |
| Reranker | BAAI/bge-reranker-v2-m3 (local) |
| LLM | Claude API (Haiku test / Sonnet demo) |
| Evaluation | RAGAS |
| Frontend | HTML/JS thuần |
| Containerization | Docker Compose |

## 7. Tài liệu liên quan

| # | Tài liệu | Mục đích |
|---|---|---|
| 01 | [Requirements](./01-requirements.md) | Yêu cầu chức năng & phi chức năng |
| 02 | [System Architecture](./02-system-architecture.md) | Kiến trúc, tech stack chi tiết |
| 03 | [User Flows & Use Cases](./03-user-flows-use-cases.md) | Luồng người dùng |
| 04 | [Data & Database](./04-data-and-database.md) | Nguồn dữ liệu, schema DB |
| 05 | [API Specification](./05-api-specification.md) | Endpoints FastAPI |
| 06 | [AI/RAG Pipeline](./06-ai-rag-pipeline.md) | RAG pipeline chi tiết |
| 07 | [Agentic RAG & Tools](./07-agentic-rag-tools.md) | Router, calculator, self-RAG |
| 08 | [Frontend & UX](./08-frontend-ux.md) | Chatbot + Dashboard |
| 09 | [Security & Guardrails](./09-security-guardrails.md) | Auth, guardrails |
| 10 | [Testing & Evaluation](./10-testing-evaluation.md) | RAGAS, test plan |
| 11 | [Deployment & DevOps](./11-deployment-devops.md) | Docker, deployment |
| 12 | [Learning Roadmap](./12-learning-roadmap.md) | 8 tuần module progression |
| 13 | [Acceptance Criteria](./13-acceptance-criteria.md) | Tiêu chí nghiệm thu |
