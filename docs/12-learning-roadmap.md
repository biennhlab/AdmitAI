# AdmitAI — Learning Roadmap (8 Tuần)

> Tiến trình học 8 tuần, kết hợp milestone kỹ thuật.  
> Đây là tài liệu kết hợp learning path + implementation milestones.

---

## Nguyên tắc cốt lõi

> **"Fail trước, rồi mới cải thiện"** — Không setup hết stack ngay tuần 1. Mỗi module thêm 1 technique mới SAU KHI đã thấy hạn chế của module trước. Đó là cách hiểu tại sao mỗi kỹ thuật cần thiết.

---

## Tuần 1 — Module 1: Naive RAG

### Mục tiêu
Xây dựng pipeline RAG đơn giản nhất có thể hoạt động, làm baseline để so sánh.

### Tech cần setup
| Công nghệ | Mục đích |
|---|---|
| FastAPI | API server |
| pdfplumber | Parse PDF |
| sentence-transformers (`bge-small-en-v1.5`) | Embedding |
| numpy (cosine similarity thủ công) | Vector search (chưa dùng Qdrant) |

### Deliverables
- [ ] Parse PDF đề án tuyển sinh → extract text
- [ ] Fixed-size chunking (character split + overlap)
- [ ] Embed chunks bằng bge-small
- [ ] Cosine similarity search (numpy, in-memory)
- [ ] Simple prompt → Claude Haiku generate answer
- [ ] FastAPI endpoint `POST /api/chat`
- [ ] Frontend chatbot cơ bản (HTML/JS)
- [ ] Chạy thử 5-10 câu hỏi, ghi nhận điểm yếu

### Kỳ vọng điểm yếu (để cải thiện ở module sau)
- Chunking cắt giữa câu/bảng → mất context
- Cosine similarity bỏ sót documents (từ khóa khác nhau nhưng cùng ý)
- Không có citation

### Eval checkpoint
Chạy RAGAS trên golden dataset (30 câu), ghi vào `eval_results/module1_naive_rag.json`

> **Ref**: [06-ai-rag-pipeline.md §2.2 Strategy 1](./06-ai-rag-pipeline.md)

---

## Tuần 2 — Module 2: Advanced Chunking

### Mục tiêu
Cải thiện chất lượng chunks — đây là phần tự code quan trọng nhất.

### Tech mới
| Công nghệ | Mục đích |
|---|---|
| camelot / `pdfplumber.extract_tables()` | Table extraction |
| Logic chunking tự code | Structure-aware, table-aware, parent-child |

### Deliverables
- [ ] Structure-aware chunking (giữ heading boundaries)
- [ ] Table-aware chunking (bảng = chunk riêng, kèm context)
- [ ] Parent-child chunking (child retrieval → parent generation)
- [ ] Thêm metadata vào mỗi chunk (source, page, heading, type)
- [ ] So sánh eval Module 1 vs Module 2

### Câu hỏi cần trả lời
- Chunk size tối ưu cho corpus PTIT là bao nhiêu?
- Parent-child tỷ lệ kích thước nên thế nào?
- Bảng điểm chuẩn nên chunk theo năm hay theo ngành?

> **Ref**: [06-ai-rag-pipeline.md §2.2](./06-ai-rag-pipeline.md)

---

## Tuần 3 — Module 3: Hybrid Search

### Mục tiêu
Chuyển từ cosine similarity thủ công sang hybrid retrieval (dense + sparse).

### Tech mới
| Công nghệ | Mục đích |
|---|---|
| Qdrant (Docker) | Vector database thật |
| rank_bm25 | Sparse search |
| RRF fusion (tự code) | Kết hợp 2 kết quả |
| Nâng cấp sang `bge-m3` | Hỗ trợ tiếng Việt tốt hơn |

### Deliverables
- [ ] Setup Qdrant qua Docker Compose
- [ ] Index chunks vào Qdrant (dense vectors)
- [ ] Build BM25 index (in-memory)
- [ ] Tự code RRF fusion (~10 dòng)
- [ ] Metadata filtering trong Qdrant
- [ ] So sánh: dense-only vs BM25-only vs hybrid
- [ ] So sánh eval Module 2 vs Module 3

### Câu hỏi cần trả lời
- Dense search mạnh ở đâu, yếu ở đâu so với BM25?
- RRF k=60 hay giá trị khác tốt hơn?
- Metadata filtering giúp ích trong trường hợp nào?

> **Ref**: [06-ai-rag-pipeline.md §3.2](./06-ai-rag-pipeline.md), [04-data-and-database.md §3](./04-data-and-database.md)

---

## Tuần 4 — Module 4: Query Transformation

### Mục tiêu
Cải thiện retrieval bằng cách biến đổi câu hỏi trước khi search.

### Tech mới
| Công nghệ | Mục đích |
|---|---|
| Claude API (Haiku) | Rewrite, HyDE, multi-query |

### Deliverables
- [ ] Query rewrite (rephrase câu hỏi rõ ràng hơn)
- [ ] HyDE — Hypothetical Document Embedding
- [ ] Multi-query (split câu hỏi phức tạp)
- [ ] Tự code prompt templates, gọi LLM trực tiếp
- [ ] A/B test: có query transform vs không
- [ ] So sánh eval Module 3 vs Module 4

### Câu hỏi cần trả lời
- HyDE cải thiện bao nhiêu % so với query gốc?
- Khi nào nên dùng multi-query vs rewrite?
- Latency tăng bao nhiêu khi thêm query transform?

> **Ref**: [06-ai-rag-pipeline.md §3.1](./06-ai-rag-pipeline.md)

---

## Tuần 5 — Module 5: Reranking

### Mục tiêu
Thêm cross-encoder reranker sau retrieval để re-score chính xác hơn.

### Tech mới
| Công nghệ | Mục đích |
|---|---|
| `BAAI/bge-reranker-v2-m3` qua CrossEncoder | Reranking |

### Deliverables
- [ ] Integrate cross-encoder reranker
- [ ] Pipeline: retrieve top-20 → rerank → top-5
- [ ] Hiểu sự khác biệt bi-encoder (embedding) vs cross-encoder (reranker)
- [ ] Thêm citation vào câu trả lời
- [ ] So sánh eval Module 4 vs Module 5

### Câu hỏi cần trả lời
- Retrieve bao nhiêu (top-K) trước khi rerank?
- Reranker tốn bao nhiêu thời gian thêm?
- Cross-encoder có cải thiện rõ rệt không?

> **Ref**: [06-ai-rag-pipeline.md §3.3](./06-ai-rag-pipeline.md)

---

## Tuần 6 — Module 6: Corrective / Self-RAG

### Mục tiêu
Chatbot tự đánh giá chất lượng retrieval và generation, retry nếu kém.

### Tech mới
Không cần lib mới — dùng **prompt tự đánh giá** qua Claude API.

### Deliverables
- [ ] Context relevance check (LLM judge: relevant hay không?)
- [ ] Faithfulness check (câu trả lời có trung thành với context?)
- [ ] Retry logic: nếu context không relevant → rewrite query → retry retrieval
- [ ] Fallback: nếu retry vẫn fail → log escalation
- [ ] So sánh eval Module 5 vs Module 6 (đặc biệt Faithfulness metric)

### Câu hỏi cần trả lời
- Self-RAG tốn bao nhiêu LLM call thêm?
- Threshold nào để quyết định "relevant enough"?
- Bao nhiêu retry là đủ trước khi fallback?

> **Ref**: [06-ai-rag-pipeline.md §3.5](./06-ai-rag-pipeline.md)

---

## Tuần 7 — Module 7: Agentic RAG

### Mục tiêu
Thêm router phân loại câu hỏi + calculator tool. Tích hợp admin dashboard.

### Tech mới
| Công nghệ | Mục đích |
|---|---|
| Claude tool use / function calling | Router, agentic flow |

### Deliverables
- [ ] Router: phân loại câu hỏi → rag_lookup / calculate_score / out_of_scope
- [ ] Calculator tool: tính điểm tổ hợp, điểm xét tuyển
- [ ] Multi-tool: kết hợp calculator + RAG trong 1 turn
- [ ] Escalation flow: log câu hỏi out-of-scope
- [ ] Admin dashboard: login, lịch sử, escalation, quản lý dữ liệu
- [ ] Admin API endpoints
- [ ] Auth (JWT)
- [ ] So sánh eval Module 6 vs Module 7

> **Ref**: [07-agentic-rag-tools.md](./07-agentic-rag-tools.md), [08-frontend-ux.md §3](./08-frontend-ux.md)

---

## Tuần 8 — Module 8: Evaluation & Polish

### Mục tiêu
Đánh giá toàn diện, tối ưu, viết case study cho CV/phỏng vấn.

### Tech mới
| Công nghệ | Mục đích |
|---|---|
| RAGAS (`ragas` package) | Evaluation chuẩn công nghiệp |
| W&B (optional) | Experiment tracking |

### Deliverables
- [ ] Golden dataset đầy đủ (30-50 câu hỏi + ground truth)
- [ ] Chạy RAGAS eval cho tất cả modules
- [ ] Bảng so sánh metric qua từng module (comparison_table.md)
- [ ] Fix bugs, tối ưu performance
- [ ] Docker Compose chạy full stack 1 lệnh
- [ ] README.md hoàn chỉnh (case study, setup guide, demo screenshots)
- [ ] (Optional) Deploy lên Railway/Render
- [ ] (Optional) Log metrics vào W&B

> **Ref**: [10-testing-evaluation.md](./10-testing-evaluation.md), [11-deployment-devops.md](./11-deployment-devops.md)

---

## Tổng kết — Timeline

```
Tuần 1  ████████  Naive RAG (baseline)
Tuần 2  ████████  Advanced Chunking
Tuần 3  ████████  Hybrid Search (Qdrant + BM25)
Tuần 4  ████████  Query Transformation
Tuần 5  ████████  Reranking
Tuần 6  ████████  Self-RAG / Corrective
Tuần 7  ████████  Agentic RAG + Admin Dashboard
Tuần 8  ████████  Evaluation + Polish + Deploy
```

### Mỗi tuần nên:
1. **Đọc lý thuyết** (1-2 ngày): Hiểu kỹ thuật mới
2. **Code** (3-4 ngày): Implement module
3. **Eval** (1 ngày): Chạy RAGAS, ghi metric, so sánh với module trước
4. **Commit + Tag**: `git tag v0.X-module-name`

### Chi phí LLM ước tính (8 tuần)
- **Test/debug**: Claude Haiku — rẻ, chạy lặp nhiều lần
- **Eval cuối mỗi module**: Claude Haiku (30-50 câu × 8 modules = ~400 calls)
- **Demo cuối cùng**: Claude Sonnet — chất lượng cao nhất
- **Ước tính**: < $10-20 cho toàn bộ 8 tuần (với Haiku)
