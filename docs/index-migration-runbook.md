# Current local index integrity

Normal development supports exactly this pair:

- Index: `data/index/module3-phase-a-20261019-223230`.
- Collection: `admitai_chunks_phase_a_20261019_223230`.
- Searchable chunks: 3,185; embedding: BGE-M3, dimension 1024.
- Build revision: `phase-a-provenance-v1`.

Keep the persisted chunk IDs, provenance, build fingerprint and embedding checksum
unchanged. The manifest's `naive_dense_numpy` backend identifier describes the
snapshot format; it is not a reference to an obsolete index directory.

Run read-only validation from the repository root:

```powershell
powershell -NoProfile -File scripts/start_local.ps1 -Action Check
python scripts/preflight_index_migration.py --index-dir data/index/module3-phase-a-20261019-223230 --collection admitai_chunks_phase_a_20261019_223230
python scripts/validate_golden_dataset.py
```

These commands do not rebuild an index or change a collection. The golden
validator defaults to the same current manifest. A missing revision/fingerprint
must not be repaired by inventing metadata. Resolve the exact configuration or
integrity failure first; a verified schema/embedding incompatibility is required
before considering a separate rebuild.

The ingestion CLI requires both `--module` and `--index-dir`; it never uses the
runtime index as a default output. The pinned and configured active index trees
are protected for every module, including with `--force`. Relative paths,
resolved aliases, directory junctions and linked output artifacts are checked
before loading the corpus or initializing embedding/Qdrant work. An existing
snapshot from another module must also be preserved in a separate directory.

For a separately approved ingestion or migration, choose a new destination.
Examples below are write commands, not read-only validation:

```powershell
python scripts/ingest.py --module 1 --index-dir data/index/module1-candidate
python scripts/ingest.py --module 2 --index-dir data/index/module2-candidate
python scripts/ingest.py --module 3 --index-dir data/index/module3-candidate --collection admitai_chunks_migration_candidate
```

Module 3 additionally requires an explicit collection separate from both the
pinned and configured active collection. Staging, upserts, stale-point removal,
unchanged checks and manifest metadata use that selected collection. Interrupted
candidate builds may reuse their own staging directory. `--force` only disables
embedding reuse and grants no permission to overwrite protected destinations.
Validate the candidate before any separately reviewed runtime configuration
switch:

```powershell
python scripts/preflight_index_migration.py --index-dir data/index/module3-candidate --collection admitai_chunks_migration_candidate
```

Ingestion never switches the runtime index or collection automatically.

Use only Check, Start and Stop through `scripts/start_local.ps1`. Start checks
index/collection/model/security conditions before replacing a verified API,
then measures RAM and commit headroom after stopping its reloader and workers.
Memory measurements are diagnostic only and impose no startup threshold. Actual
OOM failures block automatic model retries until an explicit restart.
Only the current source tree may be launched, with lite ranking and one worker.
An unsuccessful candidate is stopped and its failure is reported as BLOCKED.
No older release is started.

Legacy artifact deletion is conditional on a healthy current runtime, real
JSON/SSE citation support and frontend chat acceptance. Until all checks pass,
report CLEANUP_PENDING and retain the old artifacts untouched. No wildcard or
automatic recursive cleanup is part of normal startup. Never remove the
current index/collection, corpus, database, golden dataset, BGE-M3 model or source.
