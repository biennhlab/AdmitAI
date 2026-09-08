# AdmitAI — Deployment & DevOps

> Containerization, deployment options, và CI/CD.  
> Xem kiến trúc: [02-system-architecture.md](./02-system-architecture.md)

---

## 1. Local Development

### 1.1 Prerequisites

| Tool | Phiên bản | Mục đích |
|---|---|---|
| Python | 3.10+ | Backend, RAG pipeline |
| Docker + Docker Compose | Latest | Qdrant, containerization |
| Git | Latest | Version control |

### 1.2 Setup từ đầu

```bash
# 1. Clone repo
git clone https://github.com/<user>/AdmitAI.git
cd AdmitAI

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Tạo file .env
cp .env.example .env
# Sửa .env: thêm ANTHROPIC_API_KEY, JWT_SECRET_KEY

# 5. Khởi động Qdrant (Docker)
docker compose up -d qdrant

# 6. Tạo admin user
python scripts/create_admin.py --username admin --password your_password

# 7. Ingest dữ liệu
python scripts/ingest.py --data-dir data/raw/

# 8. Chạy backend
uvicorn api.main:app --reload --port 8000

# 9. Mở frontend
# Chatbot: mở frontend/chatbot/index.html
# Dashboard: mở frontend/dashboard/login.html
```

---

## 2. Docker Compose

### 2.1 docker-compose.yml

```yaml
version: "3.8"

services:
  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"    # REST API
      - "6334:6334"    # gRPC
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

  # FastAPI Backend
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    depends_on:
      - qdrant
    volumes:
      - ./data:/app/data
      - ./eval_results:/app/eval_results
    restart: unless-stopped

volumes:
  qdrant_data:
```

### 2.2 Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.3 Commands

```bash
# Khởi động toàn bộ stack
docker compose up -d

# Xem logs
docker compose logs -f backend

# Dừng
docker compose down

# Rebuild sau khi đổi code
docker compose up -d --build backend

# Reset Qdrant data
docker compose down -v  # WARNING: xóa toàn bộ vectors
```

---

## 3. Deployment Options

Vì bạn chưa quyết định deployment target, đây là phân tích các lựa chọn:

### 3.1 So sánh

| Option | Chi phí | Độ phức tạp | Phù hợp khi |
|---|---|---|---|
| **Local (Docker Compose)** | Free | ⭐ | Demo qua screen share, phỏng vấn trực tiếp |
| **Railway** | Free tier → ~$5/tháng | ⭐⭐ | Muốn link demo online nhanh |
| **Render** | Free tier (spin-down) | ⭐⭐ | Demo không cần always-on |
| **Fly.io** | Free tier (~3 VMs) | ⭐⭐⭐ | Cần nhiều services (backend + Qdrant) |
| **VPS (Hetzner/DigitalOcean)** | ~$5-10/tháng | ⭐⭐⭐ | Full control, ổn định |
| **Hugging Face Spaces** | Free | ⭐⭐ | Demo AI nhanh (Gradio/Streamlit) |

### 3.2 Khuyến nghị

```
Giai đoạn             │  Deployment
──────────────────────┼─────────────
Tuần 1-7 (learning)   │  Local (Docker Compose)
Tuần 8 (demo cuối)    │  Local + HOẶC Railway/Render cho online demo
Phỏng vấn             │  Có link demo online = điểm cộng
```

**Khuyến nghị**: Bắt đầu bằng **Local Docker Compose**, sau khi hoàn thiện thì deploy lên **Railway** hoặc **Render** để có link demo online.

### 3.3 Deploy lên Railway (ví dụ)

```bash
# 1. Cài Railway CLI
npm i -g @railway/cli

# 2. Login
railway login

# 3. Init project
railway init

# 4. Add Qdrant service (từ Docker image)
# → Thêm trong Railway dashboard: qdrant/qdrant

# 5. Add environment variables
railway variables set ANTHROPIC_API_KEY=sk-ant-...
railway variables set JWT_SECRET_KEY=...
railway variables set QDRANT_HOST=qdrant.railway.internal

# 6. Deploy
railway up
```

### 3.4 Deploy lên Render (ví dụ)

```yaml
# render.yaml
services:
  - type: web
    name: admitai-backend
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: ANTHROPIC_API_KEY
        sync: false
      - key: JWT_SECRET_KEY
        generateValue: true

  - type: private
    name: qdrant
    runtime: docker
    dockerCommand: ./entrypoint.sh
    disk:
      name: qdrant-data
      mountPath: /qdrant/storage
      sizeGB: 1
```

---

## 4. Environment Configuration

### 4.1 .env.example

```env
# === LLM ===
ANTHROPIC_API_KEY=your_api_key_here

# === Auth ===
JWT_SECRET_KEY=your_random_secret_here
JWT_EXPIRY_HOURS=24

# === Database ===
DATABASE_URL=sqlite:///./admitai.db

# === Qdrant ===
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=admitai_chunks

# === Embedding ===
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# === RAG Config ===
RETRIEVAL_TOP_K=20
RERANK_TOP_K=5
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

---

## 5. Git Workflow

Từ đặc tả: "mỗi module là 1 branch hoặc tag riêng"

```bash
# Branch strategy
main                    # Production-ready
├── module/01-naive-rag
├── module/02-advanced-chunking
├── module/03-hybrid-search
├── module/04-query-transform
├── module/05-reranking
├── module/06-self-rag
├── module/07-agentic-rag
└── module/08-evaluation

# Workflow mỗi module
git checkout -b module/01-naive-rag
# ... code, test, commit ...
git tag v0.1-naive-rag
git checkout main
git merge module/01-naive-rag
```

Lợi ích:
- Người xem GitHub thấy rõ **tiến trình cải thiện** qua từng bước
- Dễ demo "trước/sau" khi phỏng vấn
- Có thể checkout bất kỳ module nào để reproduce kết quả

---

## 6. Monitoring (Basic)

| Gì | Cách |
|---|---|
| API logs | FastAPI logging → stdout → Docker logs |
| Error tracking | Python `logging` module, log to file |
| Health check | `GET /api/health` endpoint |
| Qdrant status | Qdrant dashboard: `http://localhost:6333/dashboard` |

```python
# api/main.py
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "qdrant": check_qdrant_connection(),
        "timestamp": datetime.utcnow().isoformat()
    }
```
