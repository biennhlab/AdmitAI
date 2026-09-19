import sys
import asyncio
import time
import re
from typing import Any, List, Dict
from pathlib import Path
from contextlib import contextmanager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.retrieval import Embedder, NaiveDenseSearch, BM25Search, HybridRetriever, load_dense_index
import src.retrieval.fusion as fusion
from src.generation.assembler import ContextAssembler

# --- GROUND TRUTH DATA ---
BROAD_QUERIES = [
    {
        "query": "PTIT có những ngành đào tạo nào?",
        "expected_entities": [
            "Kỹ thuật Điện tử viễn thông", "Công nghệ kỹ thuật Điện, điện tử", "Công nghệ thông tin",
            "An toàn thông tin", "Khoa học máy tính", "Mạng máy tính và truyền thông dữ liệu",
            "Kỹ thuật Điều khiển và tự động hóa", "Công nghệ đa phương tiện", "Truyền thông đa phương tiện",
            "Quản trị kinh doanh", "Marketing", "Kế toán", "Thương mại điện tử", "Báo chí",
            "Công nghệ tài chính", "Kỹ thuật máy tính", "Kỹ thuật dữ liệu", "Kỹ thuật thiết kế vi mạch",
            "Công nghệ Internet vạn vật", "IoT", "Fintech"
        ]
    },
    {
        "query": "các phương thức tuyển sinh của PTIT",
        "expected_entities": [
            "Xét tuyển tài năng", "Xét tuyển dựa vào kết quả thi THPT", 
            "Xét điểm thi đánh giá năng lực", "Xét tuyển kết hợp"
        ]
    },
    {
        "query": "danh sách chương trình đào tạo",
        "expected_entities": [
            "chuẩn", "chất lượng cao", "liên kết quốc tế"
        ]
    }
]

NARROW_QUERIES = [
    {
        "query": "Điểm chuẩn ngành Công nghệ thông tin năm 2023 là bao nhiêu?",
        "expected_facts": ["26.59"]
    },
    {
        "query": "Học phí chương trình chất lượng cao là bao nhiêu?",
        "expected_facts": ["390.000", "790.000"] # 390k credit for standard, 790k for high quality, or global facts
    },
    {
        "query": "So sánh ngành CNTT và An toàn thông tin",
        "expected_facts": ["Công nghệ thông tin", "An toàn thông tin"] # Need context about both
    },
    {
        "query": "Điểm chuẩn ngành Y đa khoa năm 2023 là bao nhiêu?",
        "expected_facts": [] # Should have low scores or no facts found
    }
]

def contains_entity(text: str, entity: str) -> bool:
    return entity.lower() in text.lower()

@contextmanager
def mock_coverage_mode(enabled: bool):
    original_is_coverage = fusion._is_coverage_intent
    def mock_intent(query):
        return enabled if original_is_coverage(query) else False # If forced disabled, return False. If enabled, use actual logic
    
    # Actually, we want to force disable it for Baseline.
    fusion._is_coverage_intent = lambda q: enabled and original_is_coverage(q)
    yield
    fusion._is_coverage_intent = original_is_coverage

class RetrievalEvaluator:
    def __init__(self):
        print("Loading index...")
        chunks, embeddings, manifest = load_dense_index(settings.INDEX_DIR)
        print(f"Loaded {len(chunks)} chunks.")
        
        embedder = Embedder(model_name=settings.EMBEDDING_MODEL)
        dense_search = NaiveDenseSearch(embedder, chunks, chunk_embeddings=embeddings)
        bm25_search = BM25Search()
        bm25_search.index(chunks)
        self.retriever = HybridRetriever(dense_search, bm25_search)
        self.assembler = ContextAssembler(max_chars=12000)
        
    def _run_pipeline(self, query: str, top_k: int = 5) -> List[Any]:
        results = self.retriever.search(query, top_k=top_k)
        final_retrieved = self.assembler.assemble(results)
        return [chunk for chunk, score in final_retrieved]

    def evaluate_broad(self, queries: List[Dict]) -> Dict:
        metrics = {"recall": 0.0, "source_coverage": 0.0, "duplicate_ratio": 0.0, "latency_ms": 0.0}
        total = len(queries)
        if total == 0: return metrics
        
        total_latency = 0
        total_recall = 0
        total_sources = 0
        total_dup_ratio = 0
        
        for q_data in queries:
            start_time = time.time()
            chunks = self._run_pipeline(q_data["query"])
            total_latency += (time.time() - start_time) * 1000
            
            combined_text = "\n".join(getattr(c, "content", "") for c in chunks)
            found_entities = sum(1 for e in q_data["expected_entities"] if contains_entity(combined_text, e))
            total_recall += found_entities / len(q_data["expected_entities"])
            
            sources = set(getattr(c, "metadata", {}).get("doc_id", "") for c in chunks)
            total_sources += len(sources)
            
            # Count duplicates
            seen_norm = set()
            duplicates = 0
            for c in chunks:
                norm = self.assembler._normalize(getattr(c, "content", ""))
                if norm in seen_norm:
                    duplicates += 1
                seen_norm.add(norm)
            if len(chunks) > 0:
                total_dup_ratio += duplicates / len(chunks)
                
        return {
            "recall": total_recall / total,
            "source_coverage": total_sources / total,
            "duplicate_ratio": total_dup_ratio / total,
            "latency_ms": total_latency / total
        }

    def evaluate_narrow(self, queries: List[Dict]) -> Dict:
        metrics = {"precision_at_k": 0.0, "irrelevant_ratio": 0.0, "latency_ms": 0.0}
        total = len(queries)
        if total == 0: return metrics
        
        total_latency = 0
        total_precision = 0
        total_irrelevant = 0
        
        for q_data in queries:
            start_time = time.time()
            chunks = self._run_pipeline(q_data["query"])
            total_latency += (time.time() - start_time) * 1000
            
            if len(q_data["expected_facts"]) == 0:
                # Empty query - if we return anything with high score, it's irrelevant. 
                # But our pipeline returns chunks anyway unless RAG filters by threshold.
                # Assuming top-k is fetched. Precision here might be 0, which is correct (it's not relevant).
                total_precision += 1.0 if len(chunks) == 0 else 0.0
                total_irrelevant += 1.0 if len(chunks) > 0 else 0.0
                continue
                
            relevant_chunks = 0
            irrelevant_chunks = 0
            for chunk in chunks:
                content = getattr(chunk, "content", "")
                is_rel = any(contains_entity(content, fact) for fact in q_data["expected_facts"])
                if is_rel:
                    relevant_chunks += 1
                else:
                    irrelevant_chunks += 1
                    
            if len(chunks) > 0:
                total_precision += relevant_chunks / len(chunks)
                total_irrelevant += irrelevant_chunks / len(chunks)
                
        return {
            "precision_at_k": total_precision / total,
            "irrelevant_ratio": total_irrelevant / total,
            "latency_ms": total_latency / total
        }

async def run_eval():
    evaluator = RetrievalEvaluator()
    
    print("\n--- RUNNING BASELINE (Coverage Mode DISABLED) ---")
    with mock_coverage_mode(enabled=False):
        base_broad = evaluator.evaluate_broad(BROAD_QUERIES)
        base_narrow = evaluator.evaluate_narrow(NARROW_QUERIES)
        
    print("\n--- RUNNING OPTIMIZED (Coverage Mode ENABLED) ---")
    with mock_coverage_mode(enabled=True):
        opt_broad = evaluator.evaluate_broad(BROAD_QUERIES)
        opt_narrow = evaluator.evaluate_narrow(NARROW_QUERIES)
        
    # Write report
    report_file = PROJECT_ROOT / "eval_results" / "retrieval_qa_report.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== RETRIEVAL EVALUATION REPORT ===\n\n")
        
        f.write("--- BROAD / LIST QUERIES ---\n")
        f.write(f"Recall (Entity Coverage): BASELINE={base_broad['recall']:.2%} -> OPTIMIZED={opt_broad['recall']:.2%}\n")
        f.write(f"Source Coverage (Docs per query): BASELINE={base_broad['source_coverage']:.2f} -> OPTIMIZED={opt_broad['source_coverage']:.2f}\n")
        f.write(f"Duplicate Ratio: BASELINE={base_broad['duplicate_ratio']:.2%} -> OPTIMIZED={opt_broad['duplicate_ratio']:.2%}\n")
        f.write(f"Latency: BASELINE={base_broad['latency_ms']:.2f}ms -> OPTIMIZED={opt_broad['latency_ms']:.2f}ms\n\n")
        
        f.write("--- NARROW / FACTUAL QUERIES ---\n")
        f.write(f"Precision@K: BASELINE={base_narrow['precision_at_k']:.2%} -> OPTIMIZED={opt_narrow['precision_at_k']:.2%}\n")
        f.write(f"Irrelevant Context Ratio: BASELINE={base_narrow['irrelevant_ratio']:.2%} -> OPTIMIZED={opt_narrow['irrelevant_ratio']:.2%}\n")
        f.write(f"Latency: BASELINE={base_narrow['latency_ms']:.2f}ms -> OPTIMIZED={opt_narrow['latency_ms']:.2f}ms\n")
        
    print(f"\nReport saved to {report_file}")

if __name__ == "__main__":
    asyncio.run(run_eval())
