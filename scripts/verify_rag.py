"""Manual verification helper for the real PTIT Phase-1 RAG index/API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import settings
from src.retrieval import Embedder, NaiveDenseSearch, load_dense_index


QUESTIONS = [
    "PTIT có những phương thức tuyển sinh nào trong năm 2026?",
    "Học phí chương trình liên kết quốc tế của PTIT năm học 2026-2027 là bao nhiêu?",
    "Trong nhóm ngành Kỹ thuật, Công nghệ miền Nam, ngành Trí tuệ nhân tạo có mã ngành nào?",
    "Điểm trúng tuyển năm 2025 của ngành Kỹ thuật Điện tử viễn thông mã 7520207 tại cơ sở phía Bắc là bao nhiêu?",
    "Thời gian đăng ký xét tuyển trực tuyến PTIT năm 2026 sau điều chỉnh là khi nào?",
]


def verify_retrieval() -> list[dict]:
    chunks, embeddings, manifest = load_dense_index(settings.INDEX_DIR)
    embedder = Embedder(settings.EMBEDDING_MODEL)
    retriever = NaiveDenseSearch(embedder, chunks, chunk_embeddings=embeddings)
    results = []
    for question in QUESTIONS:
        top = retriever.search(question, top_k=5)
        results.append(
            {
                "question": question,
                "top_chunks": [
                    {
                        "score": round(score, 6),
                        "doc_id": chunk.metadata.get("doc_id"),
                        "title": chunk.metadata.get("title"),
                        "page": chunk.metadata.get("page_number"),
                        "section": chunk.metadata.get("section"),
                        "source_url": chunk.metadata.get("source_url"),
                        "snippet": " ".join(chunk.content.split())[:240],
                    }
                    for chunk, score in top
                ],
            }
        )
    return [{"index": {key: manifest.get(key) for key in ("backend", "document_count", "chunk_count", "embedding_model")}}, *results]


def verify_api(api_url: str) -> list[dict]:
    session_id = None
    results = []
    with httpx.Client(timeout=120) as client:
        for question in QUESTIONS:
            payload = {"message": question}
            if session_id:
                payload["session_id"] = session_id
            response = client.post(api_url, json=payload)
            data = response.json()
            session_id = data.get("session_id", session_id)
            results.append(
                {
                    "question": question,
                    "status": response.status_code,
                    "answer": data.get("answer"),
                    "route_type": data.get("route_type"),
                    "session_id": data.get("session_id"),
                    "citations": [
                        {
                            key: citation.get(key)
                            for key in ("title", "page", "section", "source_url", "score")
                        }
                        for citation in data.get("citations", [])[:5]
                    ],
                }
            )
        follow_up = "Nguồn đầu tiên ở câu trả lời trước nói gì?"
        response = client.post(api_url, json={"message": follow_up, "session_id": session_id})
        data = response.json()
        results.append(
            {
                "question": follow_up,
                "status": response.status_code,
                "answer": data.get("answer"),
                "route_type": data.get("route_type"),
                "session_id": data.get("session_id"),
                "citations": data.get("citations", [])[:2],
            }
        )
        unknown = "PTIT có đào tạo ngành Thú y và lịch tiêm vaccine cho chó không?"
        response = client.post(api_url, json={"message": unknown})
        data = response.json()
        results.append(
            {
                "question": unknown,
                "status": response.status_code,
                "answer": data.get("answer"),
                "route_type": data.get("route_type"),
                "session_id": data.get("session_id"),
                "citations": data.get("citations", []),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", help="When set, verify POST /api/chat instead of direct retrieval")
    args = parser.parse_args()
    payload = verify_api(args.api_url) if args.api_url else verify_retrieval()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
