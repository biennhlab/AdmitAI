import sys
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.retrieval import Embedder, NaiveDenseSearch, BM25Search, HybridRetriever, load_dense_index

EXPECTED_MAJORS = [
    "Kỹ thuật Điện tử viễn thông",
    "Công nghệ kỹ thuật Điện, điện tử",
    "Công nghệ thông tin",
    "An toàn thông tin",
    "Khoa học máy tính",
    "Mạng máy tính và truyền thông dữ liệu",
    "Kỹ thuật Điều khiển và tự động hóa",
    "Công nghệ đa phương tiện",
    "Truyền thông đa phương tiện",
    "Quản trị kinh doanh",
    "Marketing",
    "Kế toán",
    "Thương mại điện tử",
    "Báo chí",
    "Công nghệ tài chính",
    "Kỹ thuật máy tính",
    "Kỹ thuật dữ liệu",
    "Kỹ thuật thiết kế vi mạch",
    "Công nghệ Internet vạn vật",
    "IoT",
    "Fintech",
]

def check_majors(content: str) -> list[str]:
    content_lower = content.lower()
    found = []
    for major in EXPECTED_MAJORS:
        if major.lower() in content_lower:
            found.append(major)
    return found

async def main():
    print("Loading index...")
    chunks, embeddings, manifest = load_dense_index(settings.INDEX_DIR)
    print(f"Loaded {len(chunks)} chunks.")
    
    embedder = Embedder(model_name=settings.EMBEDDING_MODEL)
    dense_search = NaiveDenseSearch(embedder, chunks, chunk_embeddings=embeddings)
    bm25_search = BM25Search()
    bm25_search.index(chunks)
    hybrid_search = HybridRetriever(dense_search, bm25_search)
    
    queries = [
        "PTIT có những ngành đào tạo nào?",
        "các ngành PTIT",
        "PTIT đào tạo ngành gì",
        "danh sách ngành tuyển sinh PTIT"
    ]
    
    report_file = PROJECT_ROOT / "eval_results" / "ptit_retrieval_eval.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"EXPECTED MAJORS IN CORPUS: {', '.join(EXPECTED_MAJORS)}\n\n")
        
        for q in queries:
            f.write(f"=== Query: {q} ===\n")
            
            dense_results = list(dense_search.search(q, top_k=20))
            bm25_results = list(bm25_search.search(q, top_k=20))
            hybrid_results = list(hybrid_search.search(q, top_k=20))
            
            for name, results in [("Dense", dense_results), ("BM25", bm25_results), ("RRF", hybrid_results)]:
                f.write(f"--- {name} top-20 ---\n")
                all_found_majors = set()
                
                for rank, (chunk, score) in enumerate(results, 1):
                    found = check_majors(chunk.content)
                    all_found_majors.update(found)
                    
                    doc_id = chunk.metadata.get("doc_id", "N/A")
                    page_number = chunk.metadata.get("page_number", "N/A")
                    heading = chunk.metadata.get("heading", "N/A")
                    chunk_type = chunk.metadata.get("chunk_type", "N/A")
                    parent_chunk_id = chunk.metadata.get("parent_chunk_id", "N/A")
                    
                    f.write(f"[{rank}] doc_id={doc_id}, page={page_number}, heading={heading}, chunk_type={chunk_type}, parent_chunk_id={parent_chunk_id} | score={score:.4f}\n")
                    
                missed_majors = set(EXPECTED_MAJORS) - all_found_majors
                f.write(f"-> Majors found ({len(all_found_majors)}): {', '.join(sorted(all_found_majors))}\n")
                f.write(f"-> Majors missed ({len(missed_majors)}): {', '.join(sorted(missed_majors))}\n\n")

if __name__ == "__main__":
    asyncio.run(main())
