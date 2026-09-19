# AdmitAI

## Local development

1. Copy `.env.example` to `.env`, then configure `LLM_API_KEY` and the other backend settings.
2. Start Qdrant:

   ```bash
   docker compose up -d qdrant
   ```

3. Build or validate the hybrid Qdrant + BM25 index:

   ```bash
   python scripts/ingest.py --module 3
   ```

4. Start FastAPI from the repository root:

   ```bash
   uvicorn api.main:app --reload --port 8000
   ```

5. Copy `frontend/.env.local.example` to `frontend/.env.local`, then start Next.js:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

Before opening the chatbot, verify the backend directly at
`http://127.0.0.1:8000/api/health`. A healthy response reports `database`,
`local_index`, `qdrant`, `llm`, and `rag` separately.

The frontend proxy reads `BACKEND_URL` server-side. For a frontend running in a
container, use the backend service hostname (for example `http://backend:8000`),
not `localhost`.

### Troubleshooting socket resets on Windows

If the Uvicorn reloader cannot create its multiprocessing named pipe and logs
`PermissionError: [WinError 5]`, run the terminal with appropriate permissions
or start without `--reload`:

```bash
uvicorn api.main:app --port 8000
```

This is a process/reloader failure, not a Markdown rendering failure. The Next.js
Route Handler converts backend connection resets, refusal, and timeouts into
structured JSON errors so the chatbot remains usable and displays a safe message.
