# AdmitAI — AI/RAG Pipeline Design

> Chi tiết kỹ thuật RAG pipeline: ingestion, retrieval, generation.  
> Xem kiến trúc tổng quan: [02-system-architecture.md](./02-system-architecture.md)  
> Xem agentic features: [07-agentic-rag-tools.md](./07-agentic-rag-tools.md)

---

## 1. Pipeline Overview

```
                        INGESTION (Offline)
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Raw PDF/ │───►│ Document │───►│ Chunking │───►│ Indexing │
│ Web data │    │ Parsing  │    │ (tự code)│    │(Qdrant + │
└──────────┘    └──────────┘    └──────────┘    │  BM25)   │
                                                └──────────┘

                         QUERY (Online)
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ User     │───►│ Query    │───►│ Hybrid   │───►│ Reranker │───►│ Generate │
│ Question │    │Transform │    │Retrieval │    │          │    │+ Self-RAG│
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

---

## 2. Ingestion Pipeline (Offline)

### 2.1 Document Parsing

**Mục tiêu**: Tách riêng text vs bảng biểu — đây là điểm quan trọng nhất, không dùng loader tự động.

**Tools**:
- `pdfplumber` — text extraction + basic table extraction
- `camelot` — table extraction chính xác hơn (nếu cần)

**Logic cốt lõi cần tự code**:

```
Cho mỗi trang PDF:
  1. Extract text blocks (paragraphs, headings)
  2. Detect và extract tables riêng biệt
  3. Gán metadata: page number, heading hierarchy, table title
  4. Tách text content và table content thành 2 luồng xử lý khác nhau
```

**Lưu ý**:
- Phải detect heading hierarchy để biết context (VD: heading "Ngành CNTT" → các paragraph/table bên dưới thuộc về CNTT)
- Table phải giữ nguyên structure (rows, columns), không flatten thành text đơn thuần

### 2.2 Chunking (Tự code — Core learning)

Đây là phần **không dùng thư viện** (không `RecursiveCharacterTextSplitter`). Tự viết logic để hiểu trade-off.

#### Strategy 1: Fixed-size chunking (Tuần 1 — Naive RAG)

```
Input text → Split theo character count
  - chunk_size: 500-1000 chars
  - overlap: 100-200 chars
  - Vấn đề: cắt giữa câu, mất context
```

#### Strategy 2: Structure-aware chunking (Tuần 2)

```
Input text → Split theo cấu trúc tài liệu
  - Ưu tiên: heading boundary > paragraph boundary > sentence boundary
  - Không cắt giữa 1 section (nếu section < max_size)
  - Nếu section quá dài → split thêm theo paragraph
```

#### Strategy 3: Table-aware chunking (Tuần 2)

```
Table → 1 chunk riêng biệt
  - Kèm metadata: table title, context heading phía trên
  - Nếu bảng quá lớn → split theo nhóm rows (giữ header)
  - Đầu chunk thêm mô tả: "Bảng X thuộc mục Y"
```

#### Strategy 4: Parent-child chunking (Tuần 2)

```
Document → Parent chunks (lớn, 1000-2000 chars)
         → Child chunks (nhỏ, 200-500 chars, thuộc 1 parent)

Retrieval: search trên child chunks (chính xác hơn)
Generation: dùng parent chunk (context đầy đủ hơn)
```

### 2.3 Embedding

| Thuộc tính | Giá trị |
|---|---|
| Model | `BAAI/bge-m3` (đa ngôn ngữ, tiếng Việt) |
| Library | `sentence-transformers` |
| Chạy | Local (CPU đủ cho corpus nhỏ) |
| Dimension | 1024 |
| Phương án nhẹ | `BAAI/bge-small-en-v1.5` (384 dims) |

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
embeddings = model.encode(chunks, normalize_embeddings=True)
```

### 2.4 Indexing

```
Chunks + Embeddings
    │
    ├──► Qdrant: upsert vectors + payload metadata
    │
    └──► BM25: tokenize text → build BM25Okapi index (in-memory)
```

> Chi tiết Qdrant schema & BM25: [04-data-and-database.md](./04-data-and-database.md)

---

## 3. Query Pipeline (Online)

### 3.1 Query Transformation (Tuần 4 — Module 3)

Cải thiện chất lượng retrieval bằng cách biến đổi câu hỏi gốc.

#### a. Query Rewrite

```
Input:  "điểm chuẩn CNTT?"
Prompt: "Hãy viết lại câu hỏi sau đầy đủ và rõ ràng hơn: {query}"
Output: "Điểm chuẩn ngành Công nghệ thông tin của Học viện PTIT năm gần nhất là bao nhiêu?"
```

#### b. HyDE (Hypothetical Document Embedding)

```
Input:  "điểm chuẩn CNTT?"
Prompt: "Hãy viết một đoạn văn ngắn trả lời câu hỏi sau (dù không chắc chắn): {query}"
Output: "Điểm chuẩn ngành Công nghệ thông tin năm 2024 là 27.0 điểm..."
→ Embed đoạn hypothetical này thay vì query gốc
→ Vector gần với actual document hơn
```

#### c. Multi-query

```
Input:  "So sánh CNTT và ATTT"
Output: ["Điểm chuẩn ngành CNTT?", "Điểm chuẩn ngành ATTT?", "Khác biệt CNTT và ATTT?"]
→ Chạy retrieval cho mỗi sub-query, merge results
```

**Tự code**: Gọi thẳng Claude API với prompt, không cần thư viện riêng.

### 3.2 Hybrid Retrieval (Tuần 3 — Module 2)

Kết hợp dense search và sparse search qua **Reciprocal Rank Fusion (RRF)**.

#### Dense Search (Embedding)

```python
# Embed query
query_vector = embed_model.encode(query, normalize_embeddings=True)

# Search Qdrant
results_dense = qdrant_client.search(
    collection_name="admitai_chunks",
    query_vector=query_vector,
    limit=20
)
```

#### Sparse Search (BM25)

```python
tokenized_query = query.split()  # hoặc word segmentation
scores = bm25.get_scores(tokenized_query)
top_indices = scores.argsort()[-20:][::-1]
results_sparse = [(idx, scores[idx]) for idx in top_indices]
```

#### RRF Fusion (Tự code ~10 dòng)

```python
def reciprocal_rank_fusion(result_lists, k=60):
    """
    Combine multiple ranked lists using RRF.
    
    Args:
        result_lists: List of lists, each containing (doc_id, score) tuples
        k: Constant (default 60) — controls how much lower ranks are penalized
    
    Returns:
        Sorted list of (doc_id, fused_score)
    """
    fused_scores = {}
    for result_list in result_lists:
        for rank, (doc_id, _) in enumerate(result_list):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank + 1)
    
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

# Sử dụng
fused = reciprocal_rank_fusion([results_dense, results_sparse])
top_k = fused[:10]  # Lấy top 10
```

### 3.3 Reranking (Tuần 5 — Module 4)

Sau retrieval, dùng cross-encoder reranker để re-score chính xác hơn.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

# Score mỗi cặp (query, chunk)
pairs = [(query, chunk.text) for chunk in retrieved_chunks]
scores = reranker.predict(pairs)

# Sort theo score giảm dần
reranked = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
top_chunks = [chunk for chunk, score in reranked[:5]]
```

**Tại sao reranker cần thiết**:
- Bi-encoder (embedding search): encode query và document riêng biệt → nhanh nhưng less precise
- Cross-encoder (reranker): encode cặp (query, document) cùng lúc → chậm hơn nhưng chính xác hơn
- Kết hợp: retrieval nhanh (top 20) → rerank chính xác (top 5)

### 3.4 Generation (Tuần 1+ — Module 0-5)

#### Prompt template cơ bản

```
Bạn là trợ lý tư vấn tuyển sinh của Học viện Công nghệ Bưu chính Viễn thông (PTIT).
Hãy trả lời câu hỏi dựa HOÀN TOÀN trên các thông tin được cung cấp bên dưới.
Nếu thông tin không đủ để trả lời, hãy nói rõ là không có đủ thông tin.
Luôn trích dẫn nguồn tài liệu khi trả lời.

--- Thông tin tham khảo ---
{context_chunks}

--- Câu hỏi ---
{user_question}

--- Trả lời ---
```

#### Citation format

Mỗi câu trả lời kèm citation dạng:

```
Điểm chuẩn ngành CNTT năm 2024 là 27.0 điểm [1].

---
Nguồn:
[1] dean_tuyen_sinh_2024.pdf, trang 5
```

### 3.5 Self-RAG / Corrective RAG (Tuần 6 — Module 5)

Tự đánh giá chất lượng retrieval và generation bằng prompt, **không cần thư viện mới**.

#### Context Relevance Check

```
Prompt: "Đánh giá xem các đoạn context sau có liên quan đến câu hỏi không.
Trả lời: RELEVANT hoặc NOT_RELEVANT, kèm lý do.

Context: {retrieved_chunks}
Câu hỏi: {query}"

→ Nếu NOT_RELEVANT → retry với query rewrite hoặc fallback
```

#### Faithfulness Check

```
Prompt: "Kiểm tra xem câu trả lời sau có trung thành với context được cung cấp không.
Trả lời có chứa thông tin nào KHÔNG có trong context không?

Context: {chunks}
Câu trả lời: {generated_answer}"

→ Nếu có hallucination → regenerate với prompt chặt hơn
```

#### Self-RAG Decision Flow

```
Query → Retrieval → Context Relevance Check
                         │
                    ┌─────┴─────┐
                    ▼           ▼
                RELEVANT    NOT_RELEVANT
                    │           │
                    ▼           ▼
               Generate    Retry (rewrite query)
                    │           │
                    ▼           ▼
            Faithfulness    Still not relevant?
            Check                │
              │                  ▼
         ┌────┴────┐        Fallback +
         ▼         ▼        Escalation
       FAITHFUL  UNFAITHFUL
         │         │
         ▼         ▼
      Return    Regenerate
      answer    (stricter prompt)
```

---

## 4. Session Memory (In-session context)

Chatbot nhớ ngữ cảnh trong 1 session (bộ nhớ tạm, không persist).

```python
# In-memory session store
sessions = {}  # session_id → list of messages

def get_context_messages(session_id, max_turns=5):
    """Lấy max N turns gần nhất của session."""
    if session_id not in sessions:
        return []
    return sessions[session_id][-max_turns * 2:]  # user + assistant

# Khi gọi LLM, prepend conversation history
messages = get_context_messages(session_id) + [
    {"role": "user", "content": current_question}
]
```

---

## 5. Module Progression — Pipeline Evolution

| Tuần | Module | Thay đổi pipeline |
|---|---|---|
| 1 | Naive RAG | Fixed chunking → cosine similarity thủ công → simple prompt |
| 2 | Chunking nâng cao | → Structure/table/parent-child chunking |
| 3 | Hybrid search | → Qdrant dense + BM25 sparse + RRF fusion |
| 4 | Query transformation | → Rewrite/HyDE/multi-query trước retrieval |
| 5 | Reranking | → Cross-encoder reranker sau retrieval |
| 6 | Self-RAG / Corrective | → Context relevance + faithfulness checks |
| 7 | Agentic RAG | → Router + calculator tool (xem [07](./07-agentic-rag-tools.md)) |
| 8 | Evaluation | → RAGAS metrics (xem [10](./10-testing-evaluation.md)) |

> Roadmap chi tiết: [12-learning-roadmap.md](./12-learning-roadmap.md)
