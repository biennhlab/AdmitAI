# AdmitAI — Requirements

> Tài liệu yêu cầu chức năng và phi chức năng.  
> Xem tổng quan project: [00-project-overview.md](./00-project-overview.md)

---

## 1. Yêu cầu chức năng (Functional Requirements)

### FR-1: Chatbot tư vấn tuyển sinh (Thí sinh / Phụ huynh)

| ID | Yêu cầu | Mô tả |
|---|---|---|
| FR-1.1 | Hỏi-đáp tuyển sinh | Người dùng gửi câu hỏi bằng tiếng Việt, chatbot trả lời dựa trên dữ liệu RAG |
| FR-1.2 | Citation/trích nguồn | Mỗi câu trả lời phải kèm trích dẫn nguồn (tên tài liệu, trang/đoạn) |
| FR-1.3 | Multi-turn trong session | Chatbot nhớ ngữ cảnh trong phạm vi 1 session (bộ nhớ tạm) |
| FR-1.4 | Tính điểm tổ hợp | Calculator tool tính điểm tổ hợp xét tuyển khi thí sinh cung cấp điểm các môn |
| FR-1.5 | Tính điểm xét tuyển | Calculator tool tính tổng điểm xét tuyển (điểm thi + ưu tiên + khu vực) |
| FR-1.6 | Fallback khi không biết | Khi chatbot không đủ thông tin → thông báo rõ ràng, ghi nhận câu hỏi để nhân viên xem |
| FR-1.7 | Router phân loại | Tự động phân loại câu hỏi (RAG lookup / calculator / out-of-scope) trước khi xử lý |

### FR-2: Admin Dashboard (Nhân viên tuyển sinh)

| ID | Yêu cầu | Mô tả |
|---|---|---|
| FR-2.1 | Xem lịch sử câu hỏi | Xem danh sách tất cả câu hỏi thí sinh đã hỏi, kèm câu trả lời của bot |
| FR-2.2 | Xem escalation | Danh sách câu hỏi bot không trả lời được (fallback/escalation), đánh dấu ưu tiên |
| FR-2.3 | Trả lời trực tiếp | Nhân viên gõ câu trả lời cho câu hỏi bot không xử lý được (live agent mode) |
| FR-2.4 | Quản lý dữ liệu RAG | Upload PDF mới, xóa tài liệu cũ → re-index tự động |
| FR-2.5 | Đăng nhập | Nhân viên phải đăng nhập để truy cập dashboard |

### FR-3: RAG Pipeline

| ID | Yêu cầu | Mô tả |
|---|---|---|
| FR-3.1 | Document ingestion | Parse PDF (text + bảng riêng biệt), hỗ trợ nhiều nguồn |
| FR-3.2 | Structure-aware chunking | Tự code chunking logic: table-aware, parent-child, structure-aware |
| FR-3.3 | Hybrid retrieval | Dense search (embedding) + Sparse search (BM25) + RRF fusion |
| FR-3.4 | Query transformation | Rewrite, HyDE, multi-query trước khi retrieval |
| FR-3.5 | Reranking | Cross-encoder reranker sau retrieval |
| FR-3.6 | Self-RAG / Corrective RAG | Tự đánh giá context relevance và faithfulness, retry nếu cần |

> Chi tiết pipeline: [06-ai-rag-pipeline.md](./06-ai-rag-pipeline.md) & [07-agentic-rag-tools.md](./07-agentic-rag-tools.md)

---

## 2. Yêu cầu phi chức năng (Non-Functional Requirements)

### NFR-1: Performance

| ID | Yêu cầu | Target |
|---|---|---|
| NFR-1.1 | Thời gian phản hồi chatbot | < 10s cho câu hỏi thông thường (bao gồm retrieval + generation) |
| NFR-1.2 | Embedding/Reranker local | Chạy trên CPU, không yêu cầu GPU mạnh (corpus PTIT nhỏ) |
| NFR-1.3 | Concurrent users | Hỗ trợ tối thiểu 5-10 users đồng thời (demo/test) |

### NFR-2: Chất lượng AI

| ID | Yêu cầu | Target |
|---|---|---|
| NFR-2.1 | Context Precision | Đo bằng RAGAS, cải thiện qua từng module |
| NFR-2.2 | Context Recall | Đo bằng RAGAS, cải thiện qua từng module |
| NFR-2.3 | Faithfulness | Câu trả lời phải trung thành với context, không hallucinate |
| NFR-2.4 | Answer Relevancy | Câu trả lời phải đúng trọng tâm câu hỏi |

> Target cụ thể: [10-testing-evaluation.md](./10-testing-evaluation.md)

### NFR-3: Bảo mật

| ID | Yêu cầu |
|---|---|
| NFR-3.1 | API key LLM không hardcode, dùng environment variable |
| NFR-3.2 | Admin dashboard yêu cầu đăng nhập |
| NFR-3.3 | Chatbot có guardrails chống prompt injection cơ bản |
| NFR-3.4 | Rate limiting cho public API |

> Chi tiết: [09-security-guardrails.md](./09-security-guardrails.md)

### NFR-4: Khả năng vận hành

| ID | Yêu cầu |
|---|---|
| NFR-4.1 | Toàn bộ hệ thống chạy được bằng `docker compose up` |
| NFR-4.2 | Có README hướng dẫn setup từ đầu |
| NFR-4.3 | Cấu trúc thư mục rõ ràng theo module |
| NFR-4.4 | Mỗi module là 1 branch/tag trên Git |

### NFR-5: Ngôn ngữ & UX

| ID | Yêu cầu |
|---|---|
| NFR-5.1 | Chatbot giao tiếp bằng tiếng Việt |
| NFR-5.2 | Dữ liệu RAG bằng tiếng Việt |
| NFR-5.3 | Embedding model hỗ trợ tiếng Việt (bge-m3 multilingual) |
| NFR-5.4 | Giao diện chatbot thân thiện, dễ dùng cho thí sinh |

---

## 3. Ràng buộc (Constraints)

| Ràng buộc | Mô tả |
|---|---|
| **Ngôn ngữ lập trình** | Python (bắt buộc — ecosystem RAG/eval) |
| **Không dùng framework RAG cao cấp** | Không LangChain/LlamaIndex cho phần cốt lõi; tự code chunking, retrieval, fusion |
| **Chi phí** | Ngân sách sinh viên — ưu tiên model local miễn phí, LLM dùng model rẻ khi test |
| **Máy chạy** | Máy cá nhân, không cần GPU mạnh (corpus nhỏ, bge-small/reranker chạy tốt trên CPU) |
| **Thời gian** | 8 tuần learning path |
| **Frontend** | HTML/JS thuần — trọng tâm ở backend, không đầu tư framework frontend mới |

---

## 4. Giả định (Assumptions)

1. Corpus tuyển sinh PTIT có kích thước nhỏ (vài chục trang PDF + FAQ)
2. Số lượng user đồng thời thấp (demo/test, không phải production scale)
3. Dữ liệu tuyển sinh cập nhật theo năm (không real-time)
4. Thí sinh hỏi bằng tiếng Việt tự nhiên, có thể viết tắt hoặc sai chính tả nhẹ
