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
from src.local_profile import LOCAL_COLLECTION, LOCAL_INDEX_DIR
from src.ingestion import Chunk, ParsedPage, combined_chunk, load_documents, naive_chunk
from src.retrieval import (
    BM25Search,
    Embedder,
    QdrantDenseSearch,
    load_dense_index,
    read_manifest,
    save_dense_index,
)
from src.retrieval.document import chunk_search_text
from src.retrieval.index_version import (
    INDEX_BUILD_REVISION,
    index_build_fingerprint,
    manifest_build_stale_reasons,
)


def corpus_fingerprint(documents: list, module: int) -> str:
    """Fingerprint corpus inputs while preserving the existing golden-data value."""
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


def build_fingerprint(module: int, revision: str = INDEX_BUILD_REVISION) -> str:
    """Fingerprint build code/config independently from the corpus inputs."""
    return index_build_fingerprint(
        module,
        embedding_model=settings.EMBEDDING_MODEL,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        table_chunk_size=settings.TABLE_CHUNK_SIZE,
        parent_chunk_size=settings.PARENT_CHUNK_SIZE,
        child_chunk_size=settings.CHILD_CHUNK_SIZE,
        revision=revision,
    )


def load_reusable_snapshot(
    index_dir: str | Path,
    *,
    expected_corpus_fingerprint: str,
    expected_build_fingerprint: str,
) -> tuple[list[Chunk], np.ndarray, dict] | None:
    """Load a complete current snapshot, or return None when it is stale/missing."""
    manifest = read_manifest(index_dir)
    stale_reasons = manifest_build_stale_reasons(
        manifest,
        expected_corpus_fingerprint=expected_corpus_fingerprint,
        expected_build_fingerprint=expected_build_fingerprint,
    )
    if stale_reasons or manifest is None:
        return None
    if any(
        manifest.get(field)
        for field in ("load_errors", "chunk_errors", "embedding_errors")
    ):
        return None
    # This validates schema, vector existence, checksum, and row alignment before
    # an unchanged or interrupted build is reused.
    return load_dense_index(index_dir)


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
            matrix = embedder.embed([chunk_search_text(chunk) for chunk in batch])
            if matrix.ndim != 2 or matrix.shape[0] != len(batch):
                raise ValueError("Embedding provider returned an invalid matrix")
            successful_chunks.extend(batch)
            matrices.append(matrix)
        except Exception as batch_exc:
            for chunk in batch:
                try:
                    vector = embedder.embed([chunk_search_text(chunk)])
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
    *,
    collection: str | None = None,
) -> tuple[BM25Search, int]:
    """Upsert dense vectors and validate the matching in-memory BM25 corpus."""
    collection = collection or settings.QDRANT_COLLECTION
    dense_search = QdrantDenseSearch(
        QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT),
        collection,
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
                collection_name=collection,
                points_selector=[
                    dense_search._point_id(chunk_id) for chunk_id in sorted(stale_chunk_ids)
                ],
                wait=True,
            )
        qdrant_count = dense_search.qdrant_client.count(
            collection_name=collection,
            exact=True,
        ).count
    finally:
        dense_search.qdrant_client.close()

    sparse_search = BM25Search()
    sparse_search.index(chunks)
    return sparse_search, int(qdrant_count)


def validate_unchanged_hybrid_index(
    index_dir: str | Path, *, collection: str | None = None,
) -> tuple[dict, int] | None:
    """Return an unchanged manifest only when both retrieval branches are ready."""
    chunks, _embeddings, manifest = load_dense_index(index_dir)
    sparse_search = BM25Search()
    sparse_search.index(chunks)

    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    collection = collection or settings.QDRANT_COLLECTION
    try:
        if not client.collection_exists(collection):
            return None
        qdrant_count = int(
            client.count(collection_name=collection, exact=True).count
        )
    finally:
        client.close()

    if qdrant_count != len(chunks):
        return None
    return manifest, len(sparse_search.chunks)


def _paths_overlap(left: Path, right: Path) -> bool:
    """Compare resolved trees, also recognizing Windows extended/short aliases."""
    if left == right or left.is_relative_to(right) or right.is_relative_to(left):
        return True
    if right.exists():
        for ancestor in (left, *left.parents):
            if ancestor.exists() and ancestor.samefile(right):
                return True
    if left.exists():
        for ancestor in right.parents:
            if ancestor.exists() and ancestor.samefile(left):
                return True
    return False


def validate_ingest_destination(module: int, index_dir: str, collection: str | None) -> Path:
    """Fail closed before corpus loading, embedding, staging, or Qdrant access.

    The pinned snapshot stays protected even with the local profile disabled or
    when the command is invoked outside the repository root. Resolve junctions,
    symlinks, case aliases and '..' before checking the destination and its tree.
    --force only controls embedding reuse; it never bypasses this policy.
    """
    if not index_dir.strip():
        raise ValueError("--index-dir must name an explicit safe output directory")
    destination = Path(index_dir).resolve()
    protected_paths = {
        (PROJECT_ROOT / LOCAL_INDEX_DIR).resolve(),
        Path(settings.INDEX_DIR).resolve(),
        (PROJECT_ROOT / settings.INDEX_DIR).resolve(),
    }
    for protected in protected_paths:
        if _paths_overlap(destination, protected):
            raise ValueError(
                f"Protected active Module 3 index: {protected}; "
                "select a separate --index-dir. --force cannot override protection"
            )
    if destination.exists() and not destination.is_dir():
        raise ValueError("--index-dir must be a directory")
    # Existing output files can themselves alias the protected snapshot, even
    # when their enclosing directory is separate (including hard links).
    artifacts = ("manifest.json", "embeddings.npy", "manifest.json.tmp")
    output_names = (*artifacts, ".module3-staging", *(f".module3-staging/{name}" for name in artifacts))
    for name in output_names:
        output = destination / name
        resolved = output.resolve()
        for protected in protected_paths:
            if _paths_overlap(resolved, protected):
                raise ValueError(f"Output artifact aliases protected Module 3 index: {output}")
            if output.exists():
                for source_name in output_names:
                    source = protected / source_name
                    if source.exists() and output.samefile(source):
                        raise ValueError(f"Output artifact aliases protected Module 3 index: {output}")

    existing = read_manifest(destination)
    if existing is not None and not isinstance(existing, dict):
        raise ValueError("Existing index manifest must be a JSON object; select a new --index-dir")
    if existing is not None and existing.get("module") != module:
        raise ValueError("Existing index belongs to a different module; select a new --index-dir")
    if module == 3:
        if not collection or not collection.strip():
            raise ValueError("Module 3 requires an explicit --collection for the migration destination")
        if collection != collection.strip():
            raise ValueError("--collection must not contain surrounding whitespace")
        if collection in {LOCAL_COLLECTION, settings.QDRANT_COLLECTION}:
            raise ValueError("Protected active Qdrant collection; select a separate --collection")
        if existing is not None and existing.get("qdrant_collection") != collection:
            raise ValueError("Existing index targets a different Qdrant collection; select a new --index-dir")
    elif collection is not None:
        raise ValueError("--collection is only supported for Module 3")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest crawled PTIT data into the retrieval index")
    parser.add_argument("--data-dir", default="data/processed", help="Processed data or raw fallback directory")
    parser.add_argument(
        "--module",
        type=int,
        choices=(1, 2, 3),
        required=True,
        help="Module: 1=naive dense, 2=advanced dense, 3=Qdrant + BM25 hybrid",
    )
    parser.add_argument("--index-dir", required=True, help="Explicit output directory, separate from the active index")
    parser.add_argument("--collection", help="Module 3 migration collection, separate from the active collection")
    parser.add_argument("--force", action="store_true", help="Re-embed even when the corpus is unchanged")
    args = parser.parse_args()
    try:
        args.index_dir = validate_ingest_destination(args.module, args.index_dir, args.collection)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    load_result = load_documents(args.data_dir, prefer_processed=True)
    page_count = sum(len(getattr(document, "pages", []) or []) or 1 for document in load_result.documents)
    fingerprint = corpus_fingerprint(load_result.documents, args.module)
    expected_build_fingerprint = build_fingerprint(args.module)
    existing = read_manifest(args.index_dir)
    reusable = None
    if not args.force:
        reusable = load_reusable_snapshot(
            args.index_dir,
            expected_corpus_fingerprint=fingerprint,
            expected_build_fingerprint=expected_build_fingerprint,
        )
    unchanged = reusable is not None
    if unchanged and args.module == 3:
        ready = validate_unchanged_hybrid_index(args.index_dir, collection=args.collection)
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
                        "qdrant_collection": args.collection,
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
    staged_snapshot = None
    if args.module == 3 and not args.force:
        staged_snapshot = load_reusable_snapshot(
            staging_dir,
            expected_corpus_fingerprint=fingerprint,
            expected_build_fingerprint=expected_build_fingerprint,
        )
    if staged_snapshot is not None:
        indexed_chunks, embeddings, staged = staged_snapshot
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
        "qdrant_collection": args.collection if args.module == 3 else None,
        "corpus_fingerprint": fingerprint,
        "index_build_revision": INDEX_BUILD_REVISION,
        "build_fingerprint": expected_build_fingerprint,
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
            collection=args.collection,
        )
        if qdrant_count != len(indexed_chunks):
            raise RuntimeError(
                f"Qdrant contains {qdrant_count} points after indexing "
                f"{len(indexed_chunks)} chunks; refusing to persist an out-of-sync BM25 snapshot"
            )
        print(
            f"Hybrid indexes ready: {len(indexed_chunks)} chunks upserted into "
            f"Qdrant collection '{args.collection}', "
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
                "qdrant_collection": args.collection if args.module == 3 else None,
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
