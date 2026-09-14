"""Build the configured retrieval index from crawled PTIT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.ingestion import Chunk, ParsedPage, combined_chunk, load_documents, naive_chunk
from src.retrieval import (
    BM25Search,
    Embedder,
    QdrantDenseSearch,
    load_dense_index,
    read_manifest,
    save_dense_index,
)
from src.retrieval.dense_search import chunk_embedding_text


def corpus_fingerprint(documents: list, module: int) -> str:
    digest = hashlib.sha256()
    chunking_config = {
        "module": module,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "embedding_input": "title_section_content_v1",
    }
    if module in {2, 3}:
        chunking_config.update(
            {
                "table_chunk_size": settings.TABLE_CHUNK_SIZE,
                "parent_chunk_size": settings.PARENT_CHUNK_SIZE,
                "child_chunk_size": settings.CHILD_CHUNK_SIZE,
                "chunking_strategy": "combined_v2",
            }
        )
    digest.update(
        json.dumps(chunking_config, sort_keys=True).encode()
    )
    for document in documents:
        digest.update(document.doc_id.encode("utf-8"))
        # Citation-only changes (URL/title/date/section) must also refresh the
        # persisted payload even when body text and vectors are unchanged.
        digest.update(
            json.dumps(document.metadata, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        )
    return digest.hexdigest()


def build_chunks(
    documents: list,
    module: int = 1,
    parent_store: list[Chunk] | None = None,
) -> tuple[list[Chunk], list[dict[str, str]]]:
    if module not in {1, 2, 3}:
        raise ValueError("module must be 1 (naive), 2 (advanced), or 3 (hybrid)")
    chunks: list[Chunk] = []
    errors: list[dict[str, str]] = []
    for document in documents:
        try:
            metadata = dict(document.metadata)
            metadata["doc_id"] = document.doc_id
            if module == 1:
                document_chunks = naive_chunk(
                    document.content,
                    chunk_size=settings.CHUNK_SIZE,
                    overlap=settings.CHUNK_OVERLAP,
                    metadata=metadata,
                )
            else:
                pages = list(getattr(document, "pages", []) or [])
                if not pages:
                    page_number = int(metadata.get("page_number") or 1)
                    pages = [ParsedPage(page_number=page_number, text=document.content)]
                document_parents: list[Chunk] = []
                document_chunks = combined_chunk(
                    pages,
                    max_size=settings.TABLE_CHUNK_SIZE,
                    parent_size=settings.PARENT_CHUNK_SIZE,
                    child_size=settings.CHILD_CHUNK_SIZE,
                    metadata=metadata,
                    parent_chunks=document_parents,
                )
                if parent_store is not None:
                    parent_store.extend(document_parents)
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


def build_hybrid_indexes(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    embedder: Embedder | None,
    stale_chunk_ids: set[str] | None = None,
) -> tuple[BM25Search, int]:
    """Upsert dense vectors and validate the matching in-memory BM25 corpus."""
    dense_search = QdrantDenseSearch(
        QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT),
        settings.QDRANT_COLLECTION,
        embedder,
    )
    try:
        # Reuse the vectors produced by embed_chunks: module 3 must not embed the
        # same corpus a second time merely to hand it to Qdrant.
        batch_size = max(1, settings.QDRANT_UPSERT_BATCH_SIZE)
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        for batch_number, start in enumerate(range(0, len(chunks), batch_size), start=1):
            batch_chunks = chunks[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]
            dense_search.index(batch_chunks, embeddings=batch_embeddings)
            if batch_number == 1 or batch_number == total_batches or batch_number % 10 == 0:
                print(f"Qdrant upsert batch {batch_number}/{total_batches}", flush=True)
        if stale_chunk_ids:
            dense_search.qdrant_client.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points_selector=[
                    dense_search._point_id(chunk_id) for chunk_id in sorted(stale_chunk_ids)
                ],
                wait=True,
            )
        qdrant_count = dense_search.qdrant_client.count(
            collection_name=settings.QDRANT_COLLECTION,
            exact=True,
        ).count
    finally:
        dense_search.qdrant_client.close()

    sparse_search = BM25Search()
    sparse_search.index(chunks)
    return sparse_search, int(qdrant_count)


def validate_unchanged_hybrid_index(index_dir: str | Path) -> tuple[dict, int] | None:
    """Return an unchanged manifest only when both retrieval branches are ready."""
    chunks, _embeddings, manifest = load_dense_index(index_dir)
    sparse_search = BM25Search()
    sparse_search.index(chunks)

    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    try:
        if not client.collection_exists(settings.QDRANT_COLLECTION):
            return None
        qdrant_count = int(
            client.count(collection_name=settings.QDRANT_COLLECTION, exact=True).count
        )
    finally:
        client.close()

    if qdrant_count != len(chunks):
        return None
    return manifest, len(sparse_search.chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest crawled PTIT data into the retrieval index")
    parser.add_argument("--data-dir", default="data/processed", help="Processed data or raw fallback directory")
    parser.add_argument(
        "--module",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Module: 1=naive dense, 2=advanced dense, 3=Qdrant + BM25 hybrid",
    )
    parser.add_argument("--index-dir", default=settings.INDEX_DIR)
    parser.add_argument("--force", action="store_true", help="Re-embed even when the corpus is unchanged")
    args = parser.parse_args()
    load_result = load_documents(args.data_dir, prefer_processed=True)
    page_count = sum(len(getattr(document, "pages", []) or []) or 1 for document in load_result.documents)
    fingerprint = corpus_fingerprint(load_result.documents, args.module)
    existing = read_manifest(args.index_dir)
    unchanged = (
        not args.force
        and existing
        and existing.get("corpus_fingerprint") == fingerprint
        and (Path(args.index_dir) / "embeddings.npy").exists()
        and not existing.get("load_errors")
        and not existing.get("chunk_errors")
        and not existing.get("embedding_errors")
    )
    if unchanged and args.module == 3:
        ready = validate_unchanged_hybrid_index(args.index_dir)
        if ready is not None:
            unchanged_manifest, bm25_chunk_count = ready
            print(
                json.dumps(
                    {
                        "status": "unchanged",
                        "source_dir": load_result.source_dir,
                        "documents": unchanged_manifest.get("document_count"),
                        "pages": unchanged_manifest.get("page_count", page_count),
                        "chunks": unchanged_manifest.get("chunk_count"),
                        "qdrant_collection": settings.QDRANT_COLLECTION,
                        "qdrant_chunks": unchanged_manifest.get("chunk_count"),
                        "bm25_chunks": bm25_chunk_count,
                        "index_dir": str(Path(args.index_dir).resolve()),
                        "skipped_duplicates": len(load_result.skipped_duplicates),
                        "load_errors": load_result.errors,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        print("Hybrid index snapshot is unchanged but Qdrant is missing or out of sync; rebuilding")
    elif unchanged:
        print(
            json.dumps(
                {
                    "status": "unchanged",
                    "source_dir": load_result.source_dir,
                    "documents": existing.get("document_count"),
                    "pages": existing.get("page_count", page_count),
                    "chunks": existing.get("chunk_count"),
                    "table_chunks": existing.get("table_chunk_count", 0),
                    "parent_chunks": existing.get("parent_chunk_count", 0),
                    "child_chunks": existing.get("child_chunk_count", 0),
                    "index_dir": str(Path(args.index_dir).resolve()),
                    "skipped_duplicates": len(load_result.skipped_duplicates),
                    "load_errors": load_result.errors,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    parent_chunks: list[Chunk] = []
    chunks, chunk_errors = build_chunks(load_result.documents, args.module, parent_chunks)
    if not chunks:
        raise RuntimeError("No chunks were produced; refusing to replace the existing index")
    table_chunk_count = sum(chunk.metadata.get("chunk_type") == "table" for chunk in chunks)
    child_chunk_count = sum(chunk.metadata.get("chunk_type") == "child" for chunk in chunks)
    print(
        f"Loaded {len(load_result.documents)} documents / {page_count} pages; "
        f"produced {len(chunks)} chunks ({table_chunk_count} table, "
        f"{len(parent_chunks)} parent stored separately, {child_chunk_count} child)"
    )
    staging_dir = Path(args.index_dir) / ".module3-staging"
    staged = read_manifest(staging_dir) if args.module == 3 and not args.force else None
    if (
        staged
        and staged.get("corpus_fingerprint") == fingerprint
        and (staging_dir / "embeddings.npy").exists()
    ):
        indexed_chunks, embeddings, staged = load_dense_index(staging_dir)
        embedding_errors = list(staged.get("embedding_errors") or [])
        embedder: Embedder | None = None
        print(f"Reusing {len(indexed_chunks)} staged embeddings after an interrupted module 3 run")
    else:
        embedder = Embedder(settings.EMBEDDING_MODEL)
        indexed_chunks, embeddings, embedding_errors = embed_chunks(embedder, chunks)
        if not indexed_chunks:
            raise RuntimeError("No embeddings were produced; refusing to replace the existing index")

    indexed_doc_ids = {chunk.metadata.get("doc_id") for chunk in indexed_chunks}
    manifest_metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": load_result.source_dir,
        "module": args.module,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "embedding_input": "title_section_content_v1",
        "retrieval_backend": (
            "hybrid_qdrant_bm25_rrf" if args.module == 3 else "naive_dense_numpy"
        ),
        "qdrant_collection": settings.QDRANT_COLLECTION if args.module == 3 else None,
        "corpus_fingerprint": fingerprint,
        "document_count": len(indexed_doc_ids),
        "page_count": page_count,
        "table_chunk_count": sum(
            chunk.metadata.get("chunk_type") == "table" for chunk in indexed_chunks
        ),
        "parent_chunk_count": len(parent_chunks),
        "child_chunk_count": sum(
            chunk.metadata.get("chunk_type") == "child" for chunk in indexed_chunks
        ),
        "parent_chunks": [
            {"chunk_id": chunk.chunk_id, "content": chunk.content, "metadata": chunk.metadata}
            for chunk in parent_chunks
        ],
        "load_errors": load_result.errors,
        "chunk_errors": chunk_errors,
        "embedding_errors": embedding_errors,
        "skipped_duplicates": load_result.skipped_duplicates,
    }

    if args.module == 3 and embedder is not None:
        save_dense_index(staging_dir, indexed_chunks, embeddings, manifest_metadata)
        print(f"Checkpointed {len(indexed_chunks)} embeddings before Qdrant indexing")

    sparse_search: BM25Search | None = None
    qdrant_count: int | None = None
    if args.module == 3:
        stale_chunk_ids: set[str] = set()
        if existing and not (load_result.errors or chunk_errors or embedding_errors):
            previous_ids = {
                item.get("chunk_id")
                for item in existing.get("chunks", [])
                if isinstance(item, dict) and isinstance(item.get("chunk_id"), str)
            }
            stale_chunk_ids = previous_ids - {chunk.chunk_id for chunk in indexed_chunks}
        sparse_search, qdrant_count = build_hybrid_indexes(
            indexed_chunks,
            embeddings,
            embedder,
            stale_chunk_ids=stale_chunk_ids,
        )
        if qdrant_count != len(indexed_chunks):
            raise RuntimeError(
                f"Qdrant contains {qdrant_count} points after indexing "
                f"{len(indexed_chunks)} chunks; refusing to persist an out-of-sync BM25 snapshot"
            )
        print(
            f"Hybrid indexes ready: {len(indexed_chunks)} chunks upserted into "
            f"Qdrant collection '{settings.QDRANT_COLLECTION}', "
            f"{len(sparse_search.chunks)} chunks loaded into BM25"
        )

    manifest = save_dense_index(
        args.index_dir,
        indexed_chunks,
        embeddings,
        manifest_metadata,
    )
    print(
        json.dumps(
            {
                "status": "indexed",
                "source_dir": load_result.source_dir,
                "documents": manifest["document_count"],
                "pages": manifest["page_count"],
                "chunks": manifest["chunk_count"],
                "table_chunks": manifest["table_chunk_count"],
                "parent_chunks": manifest["parent_chunk_count"],
                "child_chunks": manifest["child_chunk_count"],
                "embedding_dimension": manifest["embedding_dimension"],
                "index_dir": str(Path(args.index_dir).resolve()),
                "qdrant_collection": settings.QDRANT_COLLECTION if args.module == 3 else None,
                "qdrant_chunks": qdrant_count,
                "bm25_chunks": len(sparse_search.chunks) if sparse_search is not None else None,
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
