"""Build the canonical Phase-1 dense index from crawled PTIT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.ingestion import Chunk, load_documents, naive_chunk
from src.retrieval import Embedder, read_manifest, save_dense_index
from src.retrieval.dense_search import chunk_embedding_text


def corpus_fingerprint(documents: list, module: int) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "module": module,
                "embedding_model": settings.EMBEDDING_MODEL,
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "embedding_input": "title_section_content_v1",
            },
            sort_keys=True,
        ).encode()
    )
    for document in documents:
        digest.update(document.doc_id.encode("utf-8"))
        # Citation-only changes (URL/title/date/section) must also refresh the
        # persisted payload even when body text and vectors are unchanged.
        digest.update(
            json.dumps(document.metadata, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        )
    return digest.hexdigest()


def build_chunks(documents: list) -> tuple[list[Chunk], list[dict[str, str]]]:
    chunks: list[Chunk] = []
    errors: list[dict[str, str]] = []
    for document in documents:
        try:
            metadata = dict(document.metadata)
            metadata["doc_id"] = document.doc_id
            document_chunks = naive_chunk(
                document.content,
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP,
                metadata=metadata,
            )
            for chunk in document_chunks:
                chunk.metadata["document_chunk_count"] = len(document_chunks)
            chunks.extend(document_chunks)
        except Exception as exc:
            errors.append({"file": document.file_path, "error": str(exc)})
    return chunks, errors


def embed_chunks(embedder: Embedder, chunks: list[Chunk]) -> tuple[list[Chunk], np.ndarray, list[dict[str, str]]]:
    successful_chunks: list[Chunk] = []
    matrices: list[np.ndarray] = []
    errors: list[dict[str, str]] = []
    batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(chunks), batch_size), start=1):
        batch = chunks[start : start + batch_size]
        try:
            matrix = embedder.embed([chunk_embedding_text(chunk) for chunk in batch])
            if matrix.ndim != 2 or matrix.shape[0] != len(batch):
                raise ValueError("Embedding provider returned an invalid matrix")
            successful_chunks.extend(batch)
            matrices.append(matrix)
        except Exception as batch_exc:
            for chunk in batch:
                try:
                    vector = embedder.embed([chunk_embedding_text(chunk)])
                    successful_chunks.append(chunk)
                    matrices.append(vector)
                except Exception as exc:
                    errors.append({"chunk_id": chunk.chunk_id, "error": str(exc or batch_exc)})
        if batch_number == 1 or batch_number == total_batches or batch_number % 25 == 0:
            print(f"Embedded batch {batch_number}/{total_batches}", flush=True)
    if not matrices:
        return [], np.empty((0, 0), dtype=np.float32), errors
    return successful_chunks, np.vstack(matrices), errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest crawled PTIT data into the Phase-1 dense index")
    parser.add_argument("--data-dir", default="data/processed", help="Processed data or raw fallback directory")
    parser.add_argument("--module", type=int, default=1, help="Implementation-plan module (currently 1)")
    parser.add_argument("--index-dir", default=settings.INDEX_DIR)
    parser.add_argument("--force", action="store_true", help="Re-embed even when the corpus is unchanged")
    args = parser.parse_args()
    if args.module != 1:
        parser.error("This codebase is currently at Module 1; only naive dense ingestion is canonical")

    load_result = load_documents(args.data_dir, prefer_processed=True)
    fingerprint = corpus_fingerprint(load_result.documents, args.module)
    existing = read_manifest(args.index_dir)
    if (
        not args.force
        and existing
        and existing.get("corpus_fingerprint") == fingerprint
        and (Path(args.index_dir) / "embeddings.npy").exists()
    ):
        print(
            json.dumps(
                {
                    "status": "unchanged",
                    "source_dir": load_result.source_dir,
                    "documents": existing.get("document_count"),
                    "chunks": existing.get("chunk_count"),
                    "index_dir": str(Path(args.index_dir).resolve()),
                    "skipped_duplicates": len(load_result.skipped_duplicates),
                    "load_errors": load_result.errors,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    chunks, chunk_errors = build_chunks(load_result.documents)
    if not chunks:
        raise RuntimeError("No chunks were produced; refusing to replace the existing index")
    print(f"Loaded {len(load_result.documents)} documents and produced {len(chunks)} chunks")
    embedder = Embedder(settings.EMBEDDING_MODEL)
    indexed_chunks, embeddings, embedding_errors = embed_chunks(embedder, chunks)
    if not indexed_chunks:
        raise RuntimeError("No embeddings were produced; refusing to replace the existing index")

    indexed_doc_ids = {chunk.metadata.get("doc_id") for chunk in indexed_chunks}
    manifest = save_dense_index(
        args.index_dir,
        indexed_chunks,
        embeddings,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": load_result.source_dir,
            "module": args.module,
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "embedding_input": "title_section_content_v1",
            "corpus_fingerprint": fingerprint,
            "document_count": len(indexed_doc_ids),
            "load_errors": load_result.errors,
            "chunk_errors": chunk_errors,
            "embedding_errors": embedding_errors,
            "skipped_duplicates": load_result.skipped_duplicates,
        },
    )
    print(
        json.dumps(
            {
                "status": "indexed",
                "source_dir": load_result.source_dir,
                "documents": manifest["document_count"],
                "chunks": manifest["chunk_count"],
                "embedding_dimension": manifest["embedding_dimension"],
                "index_dir": str(Path(args.index_dir).resolve()),
                "errors": len(load_result.errors) + len(chunk_errors) + len(embedding_errors),
                "skipped_duplicates": len(load_result.skipped_duplicates),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
