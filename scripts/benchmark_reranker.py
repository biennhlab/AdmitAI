"""Manual CPU benchmark for CrossEncoder candidate batch sizes."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.ingestion.chunker import Chunk
from src.retrieval.document import chunk_search_text
from src.retrieval.reranker import Reranker


QUERY = "PTIT có những phương thức tuyển sinh nào trong năm 2026?"
DEFAULT_COUNTS = (5, 6, 8, 10)


def benchmark_candidates(max_count: int) -> list[tuple[Chunk, float]]:
    paragraphs = [
        (
            "PTIT công bố các phương thức tuyển sinh năm 2026, gồm xét tuyển thẳng, "
            "xét kết quả thi tốt nghiệp THPT và các phương thức xét tuyển kết hợp. "
        )
        * (1 + index % 4)
        for index in range(max_count)
    ]
    return [
        (
            Chunk(
                chunk_id=f"benchmark-{index:02d}",
                content=content,
                metadata={"heading": f"Phương thức tuyển sinh {index + 1}"},
            ),
            1.0 - index / 100,
        )
        for index, content in enumerate(paragraphs)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local CrossEncoder latency by candidate count."
    )
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_COUNTS),
        help="Positive candidate counts to benchmark (default: 5 6 8 10).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured runs per candidate count (default: 3).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up runs before measurement (default: 1).",
    )
    args = parser.parse_args()
    if any(count <= 0 for count in args.counts):
        parser.error("--counts values must be greater than 0")
    if args.runs <= 0:
        parser.error("--runs must be greater than 0")
    if args.warmup < 0:
        parser.error("--warmup must be 0 or greater")
    return args


def main() -> None:
    args = parse_args()
    counts = tuple(dict.fromkeys(args.counts))
    candidates = benchmark_candidates(max(counts))
    reranker = Reranker()
    warmup_batch = candidates[: min(counts)]
    for _ in range(args.warmup):
        reranker.rerank(QUERY, warmup_batch, top_k=min(5, len(warmup_batch)))

    print(
        "candidate_count\tavg_chars\tmax_chars\truns\t"
        "median_ms\tmin_ms\tmax_ms\tmedian_ms_per_candidate"
    )
    for count in counts:
        batch = candidates[:count]
        char_counts = [len(chunk_search_text(chunk)) for chunk, _ in batch]
        durations = []
        for _ in range(args.runs):
            started_at = time.perf_counter()
            reranker.rerank(QUERY, batch, top_k=min(5, count))
            durations.append((time.perf_counter() - started_at) * 1000)
        median_ms = statistics.median(durations)
        print(
            f"{count}\t{sum(char_counts) / count:.1f}\t{max(char_counts)}\t{args.runs}\t"
            f"{median_ms:.3f}\t{min(durations):.3f}\t{max(durations):.3f}\t"
            f"{median_ms / count:.3f}"
        )


if __name__ == "__main__":
    main()
