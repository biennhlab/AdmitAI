# AdmitAI — Frontend & UX

> Thiết kế 2 giao diện: Chatbot cho thí sinh + Dashboard cho nhân viên.  
> Xem use cases: [03-user-flows-use-cases.md](./03-user-flows-use-cases.md)  
> Xem API: [05-api-specification.md](./05-api-specification.md)

---

## 1. Nguyên tắc

- **HTML/JS thuần** — không framework frontend mới, trọng tâm học ở backend
- **2 giao diện riêng biệt**: chatbot (public) + dashboard (auth required)
- **Responsive cơ bản** — hoạt động trên desktop và mobile
- **Giao tiếp qua REST API** — fetch / XMLHttpRequest gọi FastAPI

---

## 2. Chatbot — Giao diện thí sinh / phụ huynh

### 2.1 Layout

```
┌─────────────────────────────────────────┐
│  🎓 AdmitAI — Tư vấn tuyển sinh PTIT   │  ← Header
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ 🤖 Xin chào! Tôi là AdmitAI,  │    │  ← Chat area
│  │ trợ lý tư vấn tuyển sinh PTIT. │    │
│  │ Bạn có thể hỏi tôi về:        │    │
│  │ • Điểm chuẩn các ngành         │    │
│  │ • Phương thức xét tuyển        │    │
│  │ • Học phí, tổ hợp môn          │    │
│  │ • Tính điểm tổ hợp             │    │
│  └─────────────────────────────────┘    │
│                                         │
│         ┌───────────────────────┐       │
│         │ 👤 Điểm chuẩn CNTT   │       │  ← User message
│         │ năm 2024?             │       │
│         └───────────────────────┘       │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ 🤖 Điểm chuẩn ngành CNTT...   │    │  ← Bot response
│  │                                 │    │
│  │ 📎 Nguồn: dean_TS_2024.pdf    │    │  ← Citation
│  │ 👍 👎                          │    │  ← Feedback
│  └─────────────────────────────────┘    │
│                                         │
├─────────────────────────────────────────┤
│  [  Nhập câu hỏi...           ] [Gửi]  │  ← Input area
└─────────────────────────────────────────┘
```

### 2.2 Thành phần UI

| Component | Mô tả | Tương tác |
|---|---|---|
| **Header** | Logo, tên project, mô tả ngắn | — |
| **Chat area** | Danh sách tin nhắn (scroll) | Auto-scroll xuống |
| **Welcome message** | Bot chào + gợi ý chủ đề | Hiện khi mở trang |
| **User message** | Tin nhắn thí sinh (canh phải) | — |
| **Bot message** | Câu trả lời bot (canh trái) | — |
| **Citation** | Nguồn trích dẫn (collapsible) | Click mở/đóng chi tiết |
| **Feedback buttons** | 👍 👎 cho mỗi câu trả lời | Click → gửi feedback API |
| **Input box** | Text input + nút Gửi | Enter hoặc click Gửi |
| **Typing indicator** | "Đang suy nghĩ..." khi chờ API | Hiện/ẩn theo trạng thái |
| **Suggested questions** | 3-4 câu hỏi gợi ý | Click → điền vào input |

### 2.3 Xử lý trạng thái

| Trạng thái | Hiển thị |
|---|---|
| Đang gửi | Disable input, hiện typing indicator |
| Nhận response | Hiển thị tin nhắn bot, enable input |
| Lỗi API | "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại." |
| Rate limited | "Bạn đang gửi quá nhanh. Vui lòng đợi..." |

### 2.4 Interaction — Gọi API

```javascript
// Pseudocode
async function sendMessage(message) {
    showTypingIndicator();
    disableInput();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                session_id: currentSessionId
            })
        });
        
        const data = await response.json();
        currentSessionId = data.session_id;
        
        displayBotMessage(data.answer, data.citations);
    } catch (error) {
        displayErrorMessage();
    } finally {
        hideTypingIndicator();
        enableInput();
    }
}
```

---

## 3. Admin Dashboard — Giao diện nhân viên tuyển sinh

### 3.1 Login Page

```
┌─────────────────────────────────────────┐
│                                         │
│       🎓 AdmitAI — Admin Panel         │
│                                         │
│       ┌─────────────────────────┐       │
│       │ Username                │       │
│       │ [                     ] │       │
│       │                         │       │
│       │ Password                │       │
│       │ [                     ] │       │
│       │                         │       │
│       │     [ Đăng nhập ]       │       │
│       └─────────────────────────┘       │
│                                         │
└─────────────────────────────────────────┘
```

### 3.2 Dashboard Layout

```
┌──────────────────────────────────────────────────────────────┐
│  🎓 AdmitAI Admin    │ Xin chào, Nguyễn Văn A   [Đăng xuất]│
├──────────┬───────────────────────────────────────────────────┤
│          │                                                   │
│ Sidebar  │  Main Content Area                                │
│          │                                                   │
│ 📋 Lịch  │  ┌─────────────────────────────────────────────┐  │
│ sử câu   │  │         (Nội dung tab hiện tại)             │  │
│ hỏi      │  │                                             │  │
│          │  │                                             │  │
│ ⚠️ Cần   │  │                                             │  │
│ xử lý    │  │                                             │  │
│ (5)      │  │                                             │  │
│          │  │                                             │  │
│ 📄 Quản  │  │                                             │  │
│ lý dữ    │  │                                             │  │
│ liệu     │  └─────────────────────────────────────────────┘  │
│          │                                                   │
└──────────┴───────────────────────────────────────────────────┘
```

### 3.3 Tab: Lịch sử câu hỏi

```
┌─────────────────────────────────────────────────────────────┐
│ Lịch sử câu hỏi                                            │
│                                                             │
│ Filter: [Tất cả ▼] [Từ ngày: ____] [Đến ngày: ____] [Lọc] │
├─────────────────────────────────────────────────────────────┤
│ # │ Thời gian        │ Câu hỏi           │ Route   │ Status│
│───┼──────────────────┼───────────────────┼─────────┼───────│
│ 1 │ 15/03 10:30      │ Điểm chuẩn CNTT? │ rag     │ ✅    │
│ 2 │ 15/03 10:35      │ Tính điểm A00... │ calc    │ ✅    │
│ 3 │ 15/03 11:00      │ Học bổng vùng... │ —       │ ⚠️    │
│───┼──────────────────┼───────────────────┼─────────┼───────│
│                           [< 1 2 3 ... >]                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Tab: Cần xử lý (Escalation)

```
┌─────────────────────────────────────────────────────────────┐
│ ⚠️ Câu hỏi cần xử lý (5 pending)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 📅 15/03/2024 11:00                                    │ │
│ │ ❓ "Có học bổng cho sinh viên vùng cao không?"          │ │
│ │ 🤖 Bot: (Không đủ thông tin để trả lời)                │ │
│ │                                                         │ │
│ │ Trả lời:                                                │ │
│ │ [                                                     ] │ │
│ │ [                                                     ] │ │
│ │                          [Gửi trả lời] [Bỏ qua]        │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 📅 15/03/2024 14:20                                    │ │
│ │ ❓ "Thủ tục nhập học online như thế nào?"                │ │
│ │ ...                                                     │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 3.5 Tab: Quản lý dữ liệu

```
┌─────────────────────────────────────────────────────────────┐
│ 📄 Quản lý dữ liệu RAG                                    │
│                                      [+ Upload tài liệu]   │
├─────────────────────────────────────────────────────────────┤
│ # │ Tên file                 │ Size  │ Chunks │ Status │ 🗑 │
│───┼──────────────────────────┼───────┼────────┼────────┼───│
│ 1 │ dean_tuyen_sinh_2024.pdf │ 2 MB  │ 45     │ ✅     │ 🗑│
│ 2 │ diem_chuan_2020_2024.pdf │ 500KB │ 12     │ ✅     │ 🗑│
│ 3 │ hoc_phi_2025.pdf         │ 300KB │ —      │ ⏳     │ — │
│ 4 │ faq_tuyen_sinh.pdf       │ 100KB │ 8      │ ✅     │ 🗑│
└─────────────────────────────────────────────────────────────┘

Upload dialog:
┌────────────────────────────────────┐
│ Upload tài liệu mới               │
│                                    │
│ [  Chọn file PDF...          ] 📎  │
│                                    │
│ ⚠️ Chỉ hỗ trợ file PDF            │
│                                    │
│          [Upload] [Hủy]            │
└────────────────────────────────────┘
```

---

## 4. Cấu trúc file Frontend

```
frontend/
├── chatbot/
│   ├── index.html          # Trang chatbot
│   ├── style.css           # CSS chatbot
│   └── app.js              # Logic chat (fetch API, render messages)
├── dashboard/
│   ├── index.html          # Trang dashboard (sau login)
│   ├── login.html          # Trang đăng nhập
│   ├── style.css           # CSS dashboard
│   └── app.js              # Logic dashboard (tabs, API calls, auth)
└── shared/
    └── api.js              # Shared API helper (base URL, error handling)
```

---

## 5. UX Guidelines

### 5.1 Chatbot

| Guideline | Mô tả |
|---|---|
| **Greeting** | Bot chào khi mở trang, kèm 3-4 gợi ý câu hỏi phổ biến |
| **Loading** | Typing indicator ("Đang suy nghĩ...") khi chờ response |
| **Citation** | Nguồn trích dẫn rõ ràng, collapsible để không chiếm chỗ |
| **Error** | Thông báo lỗi thân thiện, khuyên thử lại |
| **Empty state** | Hướng dẫn cách hỏi khi chưa có tin nhắn |
| **Mobile** | Input cố định ở dưới, chat area scroll |

### 5.2 Dashboard

| Guideline | Mô tả |
|---|---|
| **Badge count** | Sidebar hiện số escalation pending |
| **Pagination** | Phân trang cho danh sách dài |
| **Confirm delete** | Xác nhận trước khi xóa tài liệu |
| **Upload feedback** | Progress bar hoặc status khi upload + indexing |
| **Session timeout** | Tự redirect về login khi token hết hạn |
