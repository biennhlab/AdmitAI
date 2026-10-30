Next.js frontend for AdmitAI.

## Getting Started

Copy `.env.local.example` to `.env.local`. When FastAPI runs locally, keep:

```text
BACKEND_URL=http://127.0.0.1:8000
```

Then run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

`/api/chat` and `/api/health` are controlled Route Handlers. They call FastAPI
with finite timeouts and return structured JSON when the backend is unavailable.
`BACKEND_URL` is server-only and must not use the `NEXT_PUBLIC_` prefix.

For Docker deployments, set `BACKEND_URL` to the backend service hostname rather
than `localhost`.
