import sys
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.retrieval import load_dense_index

def main():
    chunks, _, _ = load_dense_index(settings.INDEX_DIR)
    print(f"Loaded {len(chunks)} chunks.")
    
    # Simple search for chunks with table or listing of majors
    for chunk in chunks:
        content = chunk.content.lower()
        if "ngành" in content and ("mã" in content or "chỉ tiêu" in content):
            print("="*40)
            print(f"doc_id: {chunk.metadata.get('doc_id')}")
            print(f"heading: {chunk.metadata.get('heading')}")
            print(f"content:\n{chunk.content[:500]}...")

if __name__ == "__main__":
    main()
