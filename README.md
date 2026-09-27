# AdmitAI

## Tổng quan dự án

AdmitAI là chatbot tư vấn tuyển sinh PTIT được xây dựng trên kiến trúc sinh nội dung tăng cường truy xuất (Retrieval-Augmented Generation — RAG). Hệ thống truy xuất bằng chứng từ kho dữ liệu tuyển sinh, kiểm tra mức độ bám sát nguồn của câu trả lời và trả về trích dẫn cho mỗi câu trả lời có căn cứ.

```text
Frontend Next.js
  -> Proxy API của Next.js
  -> Backend FastAPI
  -> Truy xuất lai (Dense/Qdrant + BM25)
  -> Mở rộng ngữ cảnh
  -> Xếp hạng lại bằng CrossEncoder
  -> Self-RAG kiểm tra độ liên quan/tính trung thực
  -> LLM
  -> Câu trả lời bám sát nguồn + trích dẫn
```

## Kiến trúc

```text
Trình duyệt
  |
  v
Next.js :3000
  | /api/*
  v
FastAPI :8000
  |
  +-- SQLite
  +-- Qdrant
  +-- Chỉ mục cục bộ
  +-- Embedding BGE-M3
  +-- Bộ xếp hạng lại BGE
  +-- Nhà cung cấp LLM
```

Frontend và backend là hai ứng dụng có thể triển khai độc lập. FastAPI quản lý `/api/*` và không phục vụ ứng dụng Next.js hoặc frontend tĩnh cũ trong môi trường production. Next.js quản lý toàn bộ route web và chuyển tiếp các yêu cầu API phía server đến FastAPI.

`BACKEND_URL` là cấu hình phía server của frontend. Không công khai biến này bằng tiền tố `NEXT_PUBLIC_`. Trong container hoặc môi trường production, hãy đặt biến thành URL dịch vụ mà backend có thể truy cập được, thay vì mặc định dùng `localhost`.

Tệp `frontend/dashboard/style.css` là nền tảng giao diện cho dashboard trong tương lai. HTML và JavaScript của dashboard chưa được triển khai trong repository này.

## Yêu cầu hệ thống

- Python 3.10 trở lên
- Node.js 20.9 trở lên (tương thích với Next.js 16)
- npm
- Docker cùng Docker Compose
- Qdrant (dịch vụ Compose là cách thiết lập cục bộ được khuyến nghị)

## Thiết lập backend

Tạo môi trường ảo từ thư mục gốc của repository:

```bash
python -m venv .venv
```

Kích hoạt trên Windows:

```bash
.venv\Scripts\activate
```

Hoặc trên hệ điều hành tương tự Unix:

```bash
source .venv/bin/activate
```

Cài đặt dependency và tạo tệp môi trường cục bộ:

```bash
pip install -r requirements.txt
copy .env.example .env
```

Trên hệ điều hành tương tự Unix, sử dụng:

```bash
cp .env.example .env
```

Cấu hình `.env` phù hợp với môi trường triển khai. Tệp này chứa các nhóm cấu hình sau:

- **LLM:** API key tương thích với nhà cung cấp, base URL, model, thời gian chờ và số lần thử lại.
- **JWT:** khóa ký và thời hạn token. Hãy thay giá trị dùng cho môi trường phát triển.
- **Cơ sở dữ liệu:** URL cơ sở dữ liệu SQLAlchemy; SQLite là lựa chọn mặc định khi chạy cục bộ.
- **CORS:** danh sách origin trình duyệt được phép gọi trực tiếp FastAPI, phân tách bằng dấu phẩy.
- **Qdrant:** host, port, collection và kích thước batch khi ingest dữ liệu.
- **Embedding/xếp hạng lại:** định danh model embedding BGE và CrossEncoder.
- **Retrieval/RAG:** số lượng ứng viên, ngưỡng điểm, kích thước chunk và vị trí chỉ mục.

Không commit `.env` hoặc thông tin xác thực thật. Trong môi trường production, đặt `CORS_ORIGINS` thành origin của frontend đã triển khai (hoặc nhiều origin phân tách bằng dấu phẩy), ví dụ:

```text
CORS_ORIGINS=https://admissions.example.edu
```

## Qdrant và ingest dữ liệu

Khởi động Qdrant:

```bash
docker compose up -d qdrant
```

Tạo collection dense trên Qdrant cùng snapshot chỉ mục BM25/cục bộ:

```bash
python scripts/ingest.py --module 3
```

Chạy lại quá trình ingest sau khi thay đổi dữ liệu nguồn, model embedding, cấu hình chunking hoặc cấu hình chỉ mục. Backend sẽ xác thực snapshot cục bộ và collection Qdrant có đồng bộ hay không; chỉ mục cũ hoặc không khớp sẽ khiến RAG hoạt động ở trạng thái suy giảm (`degraded`).

## Chạy backend

Chạy môi trường phát triển với chế độ tự động tải lại:

```bash
uvicorn api.main:app --reload --port 8000
```

Nếu Windows từ chối quyền truy cập pipe đa tiến trình của trình tải lại, hãy chạy không có `--reload`:

```bash
uvicorn api.main:app --port 8000
```

Cách bind tương tự môi trường production:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Kiểm tra trạng thái sẵn sàng của backend tại:

```text
GET http://127.0.0.1:8000/api/health
```

Response báo cáo riêng trạng thái của cơ sở dữ liệu, chỉ mục cục bộ, Qdrant, LLM và RAG. Response có thể ở trạng thái `degraded` trong khi API vẫn hoạt động để phục vụ chẩn đoán. Việc `GET /` trả về `404 Not Found` là đúng thiết kế vì FastAPI chỉ cung cấp API.

## Thiết lập frontend

Tạo tệp môi trường frontend từ `frontend/.env.local.example`, sau đó cài đặt và khởi động Next.js:

```bash
cd frontend
npm install
npm run dev
```

Cấu hình cục bộ phía server của frontend:

```text
BACKEND_URL=http://127.0.0.1:8000
BACKEND_REQUEST_TIMEOUT_MS=65000
```

Các giá trị này chỉ được đọc bởi Route Handler của Next.js. Không đặt API key, khóa bí mật JWT hoặc bí mật backend khác trong biến môi trường frontend hay mã nguồn phía client.

Build cho môi trường production:

```bash
npm run build
npm run start
```

Mở `http://localhost:3000/` để truy cập ứng dụng web. Yêu cầu chat được chuyển qua proxy `/api/chat` của Next.js đến FastAPI.

## Kiểm thử và xác thực

Chạy kiểm thử backend từ thư mục gốc của repository:

```bash
pytest tests -q
```

Chạy các bước kiểm tra frontend từ thư mục `frontend`:

```bash
cd frontend
npm test
npm run lint
npm run build
```

Một số kiểm tra smoke hữu ích cho toàn repository:

```bash
python -m compileall api src scripts
python -c "from api.main import app; print(app.title)"
git diff --check
```

### Đánh giá bằng RAGAS

Bộ dữ liệu được tuyển chọn tại `src/eval/golden_dataset.json` chứa hơn 30 câu hỏi có ground truth dựa trên các nguồn PTIT 2026 đã được ingest. Cần cấu hình API key của LLM cho cả chuỗi RAG production và bộ chấm RAGAS.

Chạy đánh giá smoke với 5 test case, sau đó chạy toàn bộ dataset:

```bash
python scripts/run_eval.py --limit 5
python scripts/run_eval.py
```

Theo mặc định, lệnh này ghi kết quả của triển khai Self-RAG hiện tại vào `eval_results/module6_self_rag.json` và tạo lại `eval_results/comparison_table.md`. Chỉ sử dụng `--module 1` đến `--module 7` khi chạy đúng phiên bản triển khai lịch sử tương ứng; bảng so sánh chỉ liệt kê các tệp kết quả thực sự đã chạy và không tự điền module còn thiếu bằng dữ liệu mẫu.

## Khắc phục sự cố

### `PermissionError: [WinError 5]` khi chạy Uvicorn

Tiến trình tải lại không thể tạo pipe đa tiến trình. Hãy sử dụng terminal có quyền phù hợp hoặc khởi động Uvicorn không có `--reload`. Đây là lỗi khởi động tiến trình, không phải lỗi hiển thị Markdown.

### Không kết nối được Qdrant

Xác nhận container đang chạy bằng `docker compose ps`, đồng thời kiểm tra `QDRANT_HOST`/`QDRANT_PORT` có thể được backend truy cập. Bên trong Compose, host thường là `qdrant`, không phải `localhost`.

### Chỉ mục không đồng bộ

Nếu endpoint health báo thiếu tệp, sai model embedding hoặc số lượng chunk trong Qdrant không khớp, hãy chạy `python scripts/ingest.py --module 3` rồi khởi động lại FastAPI.

### Thiếu API key của LLM

Đặt `LLM_API_KEY` trong `.env` và xác nhận base URL cùng model đã cấu hình khớp với nhà cung cấp được chọn. API chủ động khởi động ở trạng thái `degraded` khi thiếu API key để chức năng chẩn đoán health vẫn hoạt động.

### Frontend trả về `503 BACKEND_UNAVAILABLE`

Proxy Next.js không thể kết nối FastAPI trước khi hết thời gian chờ. Hãy xác nhận FastAPI đang chạy, gọi trực tiếp `/api/health` và kiểm tra `BACKEND_URL` có thể được tiến trình Next.js truy cập. Khi triển khai bằng container, phải sử dụng hostname dịch vụ hoặc URL nội bộ của backend.
