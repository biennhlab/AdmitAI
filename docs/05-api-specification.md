# AdmitAI — API Specification

> FastAPI endpoints cho chatbot và admin dashboard.  
> Xem kiến trúc: [02-system-architecture.md](./02-system-architecture.md)  
> Xem use cases: [03-user-flows-use-cases.md](./03-user-flows-use-cases.md)

---

## 1. Tổng quan

- **Framework**: FastAPI (tự sinh OpenAPI/Swagger docs tại `/docs`)
- **Base URL**: `http://localhost:8000/api`
- **Auth**: JWT Bearer token cho admin endpoints; public cho chat endpoints
- **Format**: JSON request/response
- **Versioning**: `/api/v1/...` (khuyến nghị cho tương lai)

---

## 2. Public Endpoints — Chat (Thí sinh / Phụ huynh)

### 2.1 `POST /api/chat`

Gửi câu hỏi và nhận câu trả lời từ chatbot.

**Request**:
```json
{
  "message": "Điểm chuẩn ngành CNTT năm 2024 là bao nhiêu?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

| Field | Type | Required | Mô tả |
|---|---|---|---|
| message | string | ✅ | Câu hỏi của thí sinh |
| session_id | string (UUID) | ❌ | ID session, nếu null → tạo session mới |

**Response** `200 OK`:
```json
{
  "answer": "Điểm chuẩn ngành Công nghệ thông tin (CNTT) năm 2024 là 27.0 điểm...",
  "citations": [
    {
      "source": "dean_tuyen_sinh_2024.pdf",
      "page": 5,
      "snippet": "Ngành CNTT: Điểm chuẩn 27.0 (phương thức xét điểm thi THPT)"
    }
  ],
  "route_type": "rag",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

| Field | Type | Mô tả |
|---|---|---|
| answer | string | Câu trả lời |
| citations | array | Danh sách nguồn trích dẫn |
| citations[].source | string | Tên file nguồn |
| citations[].page | integer | Trang trong PDF |
| citations[].snippet | string | Đoạn trích dẫn gốc |
| route_type | string | `rag` / `calculator` / `out_of_scope` |
| session_id | string | ID session (mới hoặc đã có) |

**Response** `400 Bad Request`:
```json
{
  "detail": "Message cannot be empty"
}
```

---

### 2.2 `POST /api/chat/feedback`

Thí sinh gửi feedback về câu trả lời (thumbs up/down).

**Request**:
```json
{
  "message_id": 42,
  "feedback": "positive"
}
```

| Field | Type | Required | Mô tả |
|---|---|---|---|
| message_id | integer | ✅ | ID message cần feedback |
| feedback | string | ✅ | `positive` hoặc `negative` |

**Response** `200 OK`:
```json
{
  "status": "ok"
}
```

---

## 3. Auth Endpoints

### 3.1 `POST /api/auth/login`

Đăng nhập cho nhân viên tuyển sinh.

**Request**:
```json
{
  "username": "staff01",
  "password": "securepassword"
}
```

**Response** `200 OK`:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "username": "staff01",
    "full_name": "Nguyễn Văn A",
    "role": "staff"
  }
}
```

**Response** `401 Unauthorized`:
```json
{
  "detail": "Invalid credentials"
}
```

---

## 4. Admin Endpoints (Nhân viên tuyển sinh)

> Tất cả endpoint dưới đây yêu cầu header: `Authorization: Bearer <token>`

### 4.1 `GET /api/admin/chat-history`

Xem lịch sử tất cả câu hỏi.

**Query params**:

| Param | Type | Default | Mô tả |
|---|---|---|---|
| page | integer | 1 | Trang |
| limit | integer | 20 | Số item/trang |
| status | string | all | `all` / `answered` / `escalated` |
| from_date | string (ISO) | | Lọc từ ngày |
| to_date | string (ISO) | | Lọc đến ngày |

**Response** `200 OK`:
```json
{
  "items": [
    {
      "id": 42,
      "session_id": "550e8400-...",
      "question": "Điểm chuẩn CNTT?",
      "answer": "Điểm chuẩn ngành CNTT năm 2024...",
      "route_type": "rag",
      "has_escalation": false,
      "created_at": "2024-03-15T10:30:00Z"
    }
  ],
  "total": 150,
  "page": 1,
  "limit": 20
}
```

---

### 4.2 `GET /api/admin/escalations`

Danh sách câu hỏi cần xử lý.

**Query params**:

| Param | Type | Default | Mô tả |
|---|---|---|---|
| page | integer | 1 | Trang |
| limit | integer | 20 | Số item/trang |
| status | string | pending | `pending` / `resolved` / `all` |

**Response** `200 OK`:
```json
{
  "items": [
    {
      "id": 5,
      "message_id": 42,
      "question": "Có học bổng cho sinh viên vùng cao không?",
      "bot_answer": null,
      "status": "pending",
      "staff_reply": null,
      "created_at": "2024-03-15T14:00:00Z"
    }
  ],
  "total": 8,
  "page": 1,
  "limit": 20
}
```

---

### 4.3 `POST /api/admin/escalations/{id}/reply`

Nhân viên trả lời câu hỏi escalation.

**Request**:
```json
{
  "reply": "Hiện tại PTIT có chính sách học bổng cho sinh viên vùng cao theo QĐ số..."
}
```

**Response** `200 OK`:
```json
{
  "id": 5,
  "status": "resolved",
  "staff_reply": "Hiện tại PTIT có chính sách...",
  "resolved_by": "staff01",
  "resolved_at": "2024-03-15T15:00:00Z"
}
```

---

### 4.4 `GET /api/admin/documents`

Danh sách tài liệu RAG.

**Response** `200 OK`:
```json
{
  "items": [
    {
      "id": 1,
      "filename": "dean_tuyen_sinh_2024.pdf",
      "file_size": 2048000,
      "num_chunks": 45,
      "status": "indexed",
      "uploaded_by": "staff01",
      "uploaded_at": "2024-01-15T10:00:00Z"
    }
  ],
  "total": 6
}
```

---

### 4.5 `POST /api/admin/documents/upload`

Upload tài liệu mới và trigger re-indexing.

**Request**: `multipart/form-data`

| Field | Type | Required | Mô tả |
|---|---|---|---|
| file | file (PDF) | ✅ | File PDF tài liệu tuyển sinh |

**Response** `202 Accepted`:
```json
{
  "id": 7,
  "filename": "hoc_phi_2025.pdf",
  "status": "processing",
  "message": "Document uploaded. Indexing in progress..."
}
```

> **Note**: Indexing chạy async (parse → chunk → embed → insert Qdrant + BM25). Frontend poll status qua `GET /api/admin/documents/{id}`.

---

### 4.6 `GET /api/admin/documents/{id}`

Chi tiết 1 tài liệu (dùng để poll indexing status).

**Response** `200 OK`:
```json
{
  "id": 7,
  "filename": "hoc_phi_2025.pdf",
  "file_size": 512000,
  "num_chunks": 12,
  "status": "indexed",
  "uploaded_by": "staff01",
  "uploaded_at": "2024-03-20T09:00:00Z"
}
```

---

### 4.7 `DELETE /api/admin/documents/{id}`

Xóa tài liệu và các chunks liên quan khỏi index.

**Response** `200 OK`:
```json
{
  "message": "Document and 12 chunks deleted successfully"
}
```

---

## 5. Error Response Format

Tất cả lỗi tuân theo format FastAPI chuẩn:

```json
{
  "detail": "Error message here"
}
```

| HTTP Code | Ý nghĩa |
|---|---|
| 400 | Bad Request — input không hợp lệ |
| 401 | Unauthorized — chưa đăng nhập hoặc token hết hạn |
| 403 | Forbidden — không có quyền |
| 404 | Not Found — resource không tồn tại |
| 422 | Validation Error — dữ liệu không đúng schema |
| 429 | Too Many Requests — rate limit |
| 500 | Internal Server Error |

---

## 6. Rate Limiting

| Endpoint group | Limit | Mô tả |
|---|---|---|
| `/api/chat` | 30 requests/phút per IP | Chống spam chatbot |
| `/api/admin/*` | 60 requests/phút per user | Admin có limit cao hơn |
| `/api/admin/documents/upload` | 5 uploads/giờ per user | Upload nặng, hạn chế |
