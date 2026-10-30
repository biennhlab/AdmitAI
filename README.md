# AdmitAI

## Local development

The supported Windows profile runs **one API on port 8000**, one worker, BGE-M3,
Qdrant + BM25, domain-aware weighted RRF and Google AI. Only the current source
and validated 3,185-chunk index/collection pair are supported. There is no
automatic fallback to an older application or index.

1. Install backend dependencies with `python -m pip install -r requirements.txt`.
   Configure credentials privately in `.env`, using `.env.example` as a template.
   Revoke any previously exposed Hugging Face token before confirming the security
   gate. Do not put credentials in commands, reports or chat.
2. Start Qdrant:

   ```bash
   docker compose up -d qdrant
   ```

3. Check the existing validated index, Qdrant, local model cache and memory:

   ```powershell
   powershell -NoProfile -File scripts/start_local.ps1 -Action Check -SecurityGateConfirmed
   ```

   Use `-SecurityGateConfirmed` only after credentials are safe. The attestation
   contains no credentials. Check never stops or starts an API.
   The index and BGE-M3 cache must already be provisioned locally; they are not
   stored in Git. See [index integrity checks](docs/index-migration-runbook.md).

4. Start from the repository root:

   ```powershell
   powershell -NoProfile -File scripts/start_local.ps1
   ```

   The launcher applies `LOCAL_DEV_PROFILE=true`, overriding stale `.env` values
   together: `INDEX_DIR=data/index/module3-phase-a-20261019-223230`,
   `QDRANT_COLLECTION=admitai_chunks_phase_a_20261019_223230`,
   `RERANK_BACKEND=lite`, query rewrite off and local model cache only. It pins
   the existing embedding/chunking identity and retains M1 compatibility checks.
   It uses no Uvicorn reloader.

   The existing index has **3,185 chunks**, BGE-M3 dimension 1024. Do not rebuild
   it during normal development or fabricate fingerprints on an old index.

   RAM and commit are recorded for diagnosis; low or unavailable measurements
   do not block startup. BGE-M3 loads directly through SentenceTransformer from
   the local cache. Actual allocation failures (`MemoryError` or Windows error
   1455) are reported as `INSUFFICIENT_MEMORY`; health polling does not retry
   model loading after OOM or a missing-model failure. Free resources and restart
   explicitly after resolving the failure. Transient failures have a retry cooldown.

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

Normal Windows development uses `BACKEND_URL=http://127.0.0.1:8000`.

Stop only the recorded launcher API:

```powershell
powershell -NoProfile -File scripts/start_local.ps1 -Action Stop
```

The only lifecycle actions are Check, Start and Stop. Start validates the current
index/collection/model/security gates, verifies the existing API identity, stops
its reloader and owned workers, then measures free RAM and commit headroom again.
Memory measurements do not impose a fixed startup threshold. If a new candidate
fails startup, including an actual model allocation failure, it is stopped and
the exact failure is reported. No other release is started, and there is no dependency on an older
index, collection, archive or backup. Historical data is removed only after all
new-runtime acceptance checks pass; otherwise CLEANUP_PENDING is reported.

Local, non-secret results are in `unified-local-results/preflight.json`; logs
and PID state are in the same directory. `/api/health` includes PID, application
revision, backend and distinct startup codes such as `INDEX_FINGERPRINT_MISSING`,
`COLLECTION_MISMATCH`, `MODEL_UNAVAILABLE` and `INSUFFICIENT_MEMORY`. A running
process with `status=degraded` is not RAG ready.

JSON and SSE use the same provider response and presentation path. Generated
answers are returned without automatic factual, semantic or citation-support
validation. Source references are optional and map to real retrieved metadata
when present; absent or invalid reference IDs do not reject the answer.
Empty responses and malformed structured JSON return `INVALID_GENERATION`
(502), while retrieval unavailability remains a separate 503 error. Questions
requiring current inventory/availability abstain because the corpus is static.
See [answer generation and citations](docs/chat-generation-and-citations.md).

Run tests and actual BM25 golden evaluation without starting an API:

```powershell
python -m pytest tests -q
python scripts/evaluate_domain_lite.py --all-cases --runs 1 --output unified-local-results/evaluation-all.json
```

Run frontend regression tests from `frontend` with `npm test -- --maxWorkers=1`.
Local reports, logs, runtime state and temporary test artifacts are ignored by Git.

`--dense-qdrant` loads local BGE-M3 for real dense retrieval, while recording host
resources. Live Google AI JSON/SSE and frontend answer acceptance require safe
credentials and a healthy RAG runtime. Unit tests alone do not establish a
successful runtime cutover.

### Troubleshooting socket resets on Windows

The supported launcher runs one worker without `--reload`, avoiding the Windows
multiprocessing named-pipe failures that affect manual reloader commands.

The Next.js Route Handler converts backend connection resets, refusal, and timeouts into
structured JSON errors so the chatbot remains usable and displays a safe message.
