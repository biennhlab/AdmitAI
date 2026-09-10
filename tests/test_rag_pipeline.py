import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from api.routers import chat

async def main():
    print("Initializing RAG...")
    chat.initialize_rag()
    
    if chat.rag_chain is None:
        print(f"Failed to initialize RAG: {chat.rag_initialization_error}")
        return
        
    questions = [
        "PTIT có những phương thức tuyển sinh nào?",
        "Học phí của chương trình Kỹ sư tài năng là bao nhiêu?",
        "Ngành Công nghệ thông tin có mã ngành gì?",
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        response = await asyncio.to_thread(chat.rag_chain.answer, q)
        print(f"A: {response.answer}")
        print(f"Route: {response.route_type}")
        print("Citations:")
        for c in response.citations:
            print(f" - {c.get('source_url')} (Score: {c.get('score')})")

if __name__ == "__main__":
    asyncio.run(main())
