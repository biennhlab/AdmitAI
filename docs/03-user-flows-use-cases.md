# AdmitAI — User Flows & Use Cases

> Luồng người dùng và use cases cho cả 2 actor.  
> Xem yêu cầu: [01-requirements.md](./01-requirements.md)

---

## 1. Actors

| Actor | Mô tả | Xác thực |
|---|---|---|
| **Thí sinh / Phụ huynh** | Người dùng cuối, hỏi thông tin tuyển sinh | Không cần đăng nhập |
| **Nhân viên tuyển sinh** | Quản trị, xem lịch sử, trả lời escalation, quản lý dữ liệu | Cần đăng nhập |

---

## 2. Use Cases — Thí sinh / Phụ huynh

### UC-1: Hỏi thông tin tuyển sinh

**Actor**: Thí sinh / Phụ huynh  
**Precondition**: Truy cập trang chatbot  
**Main flow**:
1. Người dùng mở trang chatbot
2. Nhập câu hỏi bằng tiếng Việt (VD: "Điểm chuẩn ngành CNTT năm 2024 là bao nhiêu?")
3. Hệ thống route câu hỏi → RAG path
4. Query transformation (rewrite/HyDE nếu cần)
5. Hybrid retrieval (dense + BM25 + RRF fusion)
6. Reranking
7. Self-RAG kiểm tra context relevance
8. LLM generate câu trả lời kèm citation
9. Hiển thị câu trả lời + nguồn trích dẫn

**Alternative flow — Context không đủ (Self-RAG reject)**:
- 7a. Self-RAG đánh giá context không đủ relevant
- 7b. Retry retrieval với query được rewrite
- 7c. Nếu vẫn không đủ → fallback (UC-3)

**Postcondition**: Người dùng nhận câu trả lời có citation

---

### UC-2: Tính điểm xét tuyển

**Actor**: Thí sinh / Phụ huynh  
**Precondition**: Truy cập trang chatbot  
**Main flow**:
1. Người dùng hỏi về tính điểm (VD: "Tôi được 8 Toán, 7 Lý, 9 Hóa. Điểm tổ hợp A00 là bao nhiêu?")
2. Router phân loại → calculator path
3. Claude tool use gọi calculator tool với tham số trích xuất
4. Calculator tính toán kết quả
5. LLM format câu trả lời dễ hiểu
6. Hiển thị kết quả

**Alternative flow — Thiếu thông tin**:
- 2a. Người dùng không cung cấp đủ điểm các môn
- 2b. Chatbot hỏi lại thông tin còn thiếu

**Postcondition**: Người dùng biết điểm tổ hợp / điểm xét tuyển

---

### UC-3: Câu hỏi ngoài scope / Fallback

**Actor**: Thí sinh / Phụ huynh  
**Precondition**: Câu hỏi không thuộc phạm vi dữ liệu RAG  
**Main flow**:
1. Người dùng hỏi câu ngoài scope (VD: "Thời tiết hôm nay thế nào?")
2. Router phân loại → out-of-scope
3. Chatbot trả lời lịch sự: "Tôi chỉ hỗ trợ tư vấn tuyển sinh PTIT..."
4. Câu hỏi được log vào escalation queue

**Postcondition**: Người dùng được thông báo, câu hỏi được ghi nhận

---

### UC-4: Hỏi tiếp (Multi-turn trong session)

**Actor**: Thí sinh / Phụ huynh  
**Precondition**: Đang trong 1 session chat  
**Main flow**:
1. Người dùng đã hỏi câu trước (VD: "Điểm chuẩn CNTT?")
2. Hỏi tiếp liên quan (VD: "Còn ngành An toàn thông tin thì sao?")
3. Hệ thống dùng conversation context trong session để hiểu "ngành An toàn thông tin" liên quan đến "điểm chuẩn"
4. Xử lý RAG pipeline với ngữ cảnh đầy đủ

**Postcondition**: Câu trả lời phù hợp ngữ cảnh hội thoại

---

## 3. Use Cases — Nhân viên tuyển sinh

### UC-5: Đăng nhập Dashboard

**Actor**: Nhân viên tuyển sinh  
**Precondition**: Có tài khoản hệ thống  
**Main flow**:
1. Truy cập trang admin dashboard
2. Nhập username/password
3. Hệ thống xác thực → cấp JWT token
4. Redirect vào dashboard

**Alternative flow — Sai thông tin**:
- 3a. Sai credentials → hiển thị lỗi

---

### UC-6: Xem lịch sử câu hỏi

**Actor**: Nhân viên tuyển sinh  
**Precondition**: Đã đăng nhập  
**Main flow**:
1. Vào tab "Lịch sử câu hỏi"
2. Xem danh sách câu hỏi (mới nhất trước)
3. Mỗi item hiển thị: câu hỏi, câu trả lời bot, thời gian, trạng thái
4. Có thể filter theo thời gian, trạng thái (đã trả lời / escalation)

---

### UC-7: Xử lý Escalation

**Actor**: Nhân viên tuyển sinh  
**Precondition**: Có câu hỏi escalation  
**Main flow**:
1. Vào tab "Cần xử lý" (escalation queue)
2. Xem danh sách câu hỏi bot không trả lời được
3. Click vào câu hỏi → xem chi tiết
4. Gõ câu trả lời trực tiếp
5. Đánh dấu "Đã xử lý"

**Postcondition**: Câu hỏi được giải quyết, đánh dấu hoàn thành

---

### UC-8: Quản lý dữ liệu RAG

**Actor**: Nhân viên tuyển sinh  
**Precondition**: Đã đăng nhập  
**Main flow — Upload**:
1. Vào tab "Quản lý dữ liệu"
2. Xem danh sách tài liệu hiện có (tên, ngày upload, số chunks)
3. Click "Upload tài liệu mới" → chọn file PDF
4. Hệ thống parse, chunk, embed → index vào Qdrant + BM25
5. Hiển thị kết quả: số chunks tạo ra, trạng thái

**Main flow — Xóa**:
1. Chọn tài liệu cần xóa
2. Xác nhận xóa
3. Hệ thống xóa chunks khỏi Qdrant + BM25 index
4. Cập nhật danh sách

---

## 4. User Flow Diagrams

### 4.1 Flow tổng quan — Thí sinh

```
Mở chatbot ──► Nhập câu hỏi ──► Router phân loại
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                  ▼
              RAG Path          Calculator          Out-of-scope
                    │                 │                  │
                    ▼                 ▼                  ▼
            Query Transform    Extract params      Fallback msg
                    │                 │             + Log escalation
                    ▼                 ▼
            Hybrid Retrieval    Calculate score
                    │                 │
                    ▼                 ▼
              Reranker          Format result
                    │                 │
                    ▼                 │
            Self-RAG check           │
              │         │            │
              ▼         ▼            │
            Pass      Retry         │
              │         │            │
              ▼         ▼            │
            Generate (LLM)          │
              │                      │
              ▼                      ▼
        Response + Citation    Response
```

### 4.2 Flow tổng quan — Nhân viên tuyển sinh

```
Đăng nhập ──► Dashboard
                  │
      ┌───────────┼──────────────┐
      ▼           ▼              ▼
  Lịch sử    Escalation    Quản lý data
  câu hỏi    queue              │
      │           │         ┌────┼────┐
      ▼           ▼         ▼         ▼
  Xem list   Xem câu hỏi  Upload   Xóa
  + filter   + trả lời    PDF      tài liệu
                  │              │
                  ▼              ▼
            Đánh dấu       Re-index
            "Đã xử lý"    (chunk + embed)
```
