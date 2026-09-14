# AdmitAI — Agentic RAG & Tools

> Router, calculator tool, function calling, agentic flow.  
> Xem RAG pipeline cơ bản: [06-ai-rag-pipeline.md](./06-ai-rag-pipeline.md)  
> Tương ứng Tuần 7 — Module 6 trong [12-learning-roadmap.md](./12-learning-roadmap.md)

---

## 1. Tổng quan Agentic RAG

Agentic RAG mở rộng pipeline RAG cơ bản bằng cách thêm **Router** — một "agent" quyết định cách xử lý câu hỏi trước khi chạy pipeline.

```
User Question
     │
     ▼
┌─────────────────────────────┐
│         ROUTER              │
│  (Claude tool use /         │
│   function calling)         │
│                             │
│  Phân loại câu hỏi:        │
│  ┌───────────────────────┐  │
│  │ 1. rag_lookup         │──│──► RAG Pipeline (06)
│  │ 2. calculate_score    │──│──► Calculator Tool
│  │ 3. out_of_scope       │──│──► Fallback + Escalation
│  └───────────────────────┘  │
└─────────────────────────────┘
```

---

## 2. Router — Claude Tool Use / Function Calling

### 2.1 Cơ chế

Dùng **Claude tool use native** (không cần LangGraph). Khai báo tools cho Claude, để model tự quyết định gọi tool nào.

### 2.2 Tool Definitions

```python
tools = [
    {
        "name": "rag_lookup",
        "description": "Tìm kiếm thông tin tuyển sinh PTIT từ cơ sở dữ liệu. "
                       "Dùng khi câu hỏi cần tra cứu thông tin như: điểm chuẩn, "
                       "chỉ tiêu, ngành học, phương thức xét tuyển, học phí, "
                       "chương trình đào tạo, lịch trình tuyển sinh, v.v.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Câu hỏi đã được clarify/rewrite để tìm kiếm"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_score",
        "description": "Tính điểm tổ hợp xét tuyển hoặc điểm xét tuyển THPT. "
                       "Dùng khi thí sinh cung cấp điểm các môn và muốn biết "
                       "điểm tổ hợp hoặc tổng điểm xét tuyển.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject_group": {
                    "type": "string",
                    "description": "Tổ hợp xét tuyển (VD: A00, A01, D01, D07...)"
                },
                "scores": {
                    "type": "object",
                    "description": "Điểm từng môn. Key là tên môn, value là điểm.",
                    "additionalProperties": {
                        "type": "number"
                    }
                },
                "priority_score": {
                    "type": "number",
                    "description": "Điểm ưu tiên (nếu có). Mặc định 0."
                },
                "region_priority": {
                    "type": "number",
                    "description": "Điểm ưu tiên khu vực (nếu có). Mặc định 0."
                }
            },
            "required": ["subject_group", "scores"]
        }
    }
]
```

### 2.3 Router Prompt

```
Bạn là trợ lý tư vấn tuyển sinh PTIT. Phân tích câu hỏi của thí sinh và
quyết định dùng công cụ phù hợp:

1. rag_lookup — Khi cần tra cứu thông tin tuyển sinh (điểm chuẩn, ngành học, 
   học phí, phương thức xét tuyển, v.v.)
2. calculate_score — Khi thí sinh cung cấp điểm và muốn tính điểm tổ hợp
   hoặc điểm xét tuyển

Nếu câu hỏi KHÔNG liên quan đến tuyển sinh PTIT, hãy trả lời lịch sự rằng
bạn chỉ hỗ trợ tư vấn tuyển sinh.

Lịch sử hội thoại:
{conversation_history}

Câu hỏi hiện tại: {user_question}
```

### 2.4 Xử lý Tool Calls

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-...",
    tools=tools,
    messages=messages,
    system=router_system_prompt
)

# Xử lý response
for block in response.content:
    if block.type == "tool_use":
        if block.name == "rag_lookup":
            # Chạy RAG pipeline với query đã extract
            result = run_rag_pipeline(block.input["query"])
        elif block.name == "calculate_score":
            # Chạy calculator
            result = calculate_admission_score(block.input)
        
        # Gửi tool result về Claude để format câu trả lời
        final_response = client.messages.create(
            model="claude-sonnet-...",
            messages=[
                *messages,
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
                ]}
            ]
        )
    elif block.type == "text":
        # Out-of-scope — Claude trả lời trực tiếp (fallback)
        # Log vào escalation queue
        log_escalation(user_question, block.text)
```

---

## 3. Calculator Tool — Chi tiết

### 3.1 Tổ hợp xét tuyển PTIT

| Tổ hợp | Môn 1 | Môn 2 | Môn 3 |
|---|---|---|---|
| A00 | Toán | Vật lý | Hóa học |
| A01 | Toán | Vật lý | Tiếng Anh |
| D01 | Toán | Ngữ văn | Tiếng Anh |
| D07 | Toán | Hóa học | Tiếng Anh |
| ... | ... | ... | ... |

> **Note**: Danh sách tổ hợp đầy đủ cần lấy từ đề án tuyển sinh thực tế.

### 3.2 Công thức tính điểm

#### Điểm tổ hợp
```
Điểm tổ hợp = Môn 1 + Môn 2 + Môn 3
```

#### Điểm xét tuyển
```
Điểm xét tuyển = Điểm tổ hợp + Điểm ưu tiên đối tượng + Điểm ưu tiên khu vực
```

### 3.3 Implementation

```python
# Mapping tổ hợp → các môn
SUBJECT_GROUPS = {
    "A00": ["Toán", "Vật lý", "Hóa học"],
    "A01": ["Toán", "Vật lý", "Tiếng Anh"],
    "D01": ["Toán", "Ngữ văn", "Tiếng Anh"],
    "D07": ["Toán", "Hóa học", "Tiếng Anh"],
    # ... thêm tổ hợp từ đề án
}

def calculate_admission_score(input_data):
    """
    Tính điểm xét tuyển.
    
    Args:
        input_data: {
            "subject_group": "A00",
            "scores": {"Toán": 8.0, "Vật lý": 7.5, "Hóa học": 9.0},
            "priority_score": 0.5,    # Ưu tiên đối tượng (optional)
            "region_priority": 0.25   # Ưu tiên khu vực (optional)
        }
    
    Returns:
        dict với kết quả tính toán chi tiết
    """
    group = input_data["subject_group"].upper()
    scores = input_data["scores"]
    priority = input_data.get("priority_score", 0)
    region = input_data.get("region_priority", 0)
    
    if group not in SUBJECT_GROUPS:
        return {"error": f"Tổ hợp {group} không hợp lệ"}
    
    required_subjects = SUBJECT_GROUPS[group]
    
    # Validate đủ điểm
    missing = [s for s in required_subjects if s not in scores]
    if missing:
        return {"error": f"Thiếu điểm môn: {', '.join(missing)}"}
    
    # Tính toán
    subject_total = sum(scores[s] for s in required_subjects)
    total = subject_total + priority + region
    
    return {
        "subject_group": group,
        "subjects": {s: scores[s] for s in required_subjects},
        "subject_total": subject_total,
        "priority_score": priority,
        "region_priority": region,
        "total_score": total,
        "breakdown": f"{' + '.join(f'{s}: {scores[s]}' for s in required_subjects)}"
                     f"{f' + Ưu tiên: {priority}' if priority else ''}"
                     f"{f' + Khu vực: {region}' if region else ''}"
                     f" = {total}"
    }
```

### 3.4 Ví dụ tương tác

**User**: "Em được 8 Toán, 7.5 Lý, 9 Hóa. Xét tổ hợp A00 thì điểm tổ hợp bao nhiêu ạ?"

**Router**: Gọi `calculate_score` với:
```json
{
  "subject_group": "A00",
  "scores": {"Toán": 8.0, "Vật lý": 7.5, "Hóa học": 9.0}
}
```

**Calculator result**:
```json
{
  "subject_group": "A00",
  "subject_total": 24.5,
  "total_score": 24.5,
  "breakdown": "Toán: 8.0 + Vật lý: 7.5 + Hóa học: 9.0 = 24.5"
}
```

**Claude format response**:
> Với tổ hợp A00, điểm tổ hợp của em là **24.5 điểm** (Toán: 8.0 + Vật lý: 7.5 + Hóa học: 9.0). Nếu em có điểm ưu tiên đối tượng hoặc khu vực, tổng điểm xét tuyển sẽ cao hơn. Em muốn so sánh với điểm chuẩn ngành nào không?

---

## 4. Fallback & Escalation Flow

```
Câu hỏi out-of-scope
     │
     ▼
Router → Không gọi tool nào → Claude trả lời trực tiếp
     │
     ▼
"Xin lỗi, tôi chỉ hỗ trợ tư vấn tuyển sinh PTIT.
 Bạn có thể hỏi về điểm chuẩn, ngành học, phương thức xét tuyển..."
     │
     ▼
Log vào bảng `escalations` (status: pending)
     │
     ▼
Nhân viên thấy trên Dashboard → Xử lý nếu cần
```

Các trường hợp escalation:
1. **Out-of-scope**: Câu hỏi không liên quan tuyển sinh
2. **Insufficient context**: RAG không tìm đủ thông tin (Self-RAG reject)
3. **Low confidence**: Model không tự tin về câu trả lời
4. **User feedback negative**: Thí sinh đánh giá câu trả lời không tốt

---

## 5. Kết hợp RAG + Calculator (Multi-step)

Một số câu hỏi cần cả hai tool:

**Ví dụ**: "Điểm em 8 Toán, 7.5 Lý, 9 Hóa. Có đủ điểm vào ngành CNTT không?"

**Flow**:
1. Router gọi `calculate_score` → kết quả 24.5 (A00)
2. Router gọi `rag_lookup` → tìm điểm chuẩn CNTT
3. Claude so sánh 2 kết quả → "Điểm tổ hợp của em là 24.5. Điểm chuẩn CNTT năm 2024 là 27.0. Em cần thêm 2.5 điểm nữa..."

> **Lưu ý**: Claude tool use hỗ trợ multi-tool calling trong 1 turn. Router có thể gọi cả 2 tools rồi tổng hợp kết quả.
