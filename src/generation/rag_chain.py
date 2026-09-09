from dataclasses import dataclass
from typing import List, Optional, Any
from .prompts import SYSTEM_PROMPT, build_rag_prompt

@dataclass
class RAGResponse:
    answer: str
    citations: List[str]
    route_type: str

class RAGChain:
    """End-to-end RAG Generation Chain (Phase 1)"""
    
    def __init__(self, retriever: Any, llm_client: Any):
        self.retriever = retriever
        self.llm_client = llm_client

    def answer(self, question: str, session_history: Optional[List[dict]] = None) -> RAGResponse:
        """
        Process the user question through the RAG pipeline.
        Steps:
        1. Retrieve top-k chunks
        2. Check for empty context
        3. Build RAG prompt with history
        4. Call LLM
        5. Build citations
        6. Return RAGResponse
        """
        # 1. Retrieve top-k chunks
        # Assuming the naive retriever returns a list of tuples: (Chunk, score)
        retrieved_results = self.retriever.search(question, top_k=5)
        
        # Extract just the chunks for prompting (ignoring scores for simple generation)
        chunks = [result[0] for result in retrieved_results] if retrieved_results else []
        
        # Log top-k chunks to terminal
        print(f"\\n--- TOP {len(chunks)} CHUNKS RETRIEVED ---")
        for i, chunk in enumerate(chunks):
            chunk_id = getattr(chunk, 'chunk_id', 'unknown')
            snippet = chunk.content[:150].replace('\\n', ' ') + "..." if hasattr(chunk, 'content') else 'N/A'
            print(f"[Chunk {i+1}] ID: {chunk_id} | Preview: {snippet}")
        print("-----------------------------------\\n")
        
        # 2. Check for empty context
        if not chunks:
            return RAGResponse(
                answer="Xin lỗi, tôi không tìm thấy thông tin phù hợp. Vui lòng kết nối trực tiếp với nhân viên tuyển sinh để được hỗ trợ chi tiết hơn.",
                citations=[],
                route_type="general"
            )
            
        # 3. Build RAG prompt
        # We start with any session history if provided
        messages = []
        if session_history:
            messages.extend(session_history)
            
        # Add the new RAG prompt message
        rag_messages = build_rag_prompt(chunks, question)
        messages.extend(rag_messages)
        
        # 4. Call LLM
        llm_answer = self.llm_client.generate(
            system_prompt=SYSTEM_PROMPT,
            messages=messages
        )
        
        # 5. Build citations list
        citations = []
        for chunk in chunks:
            if hasattr(chunk, "metadata"):
                source = chunk.metadata.get("source", f"Chunk {chunk.chunk_id}")
            else:
                source = f"Chunk {getattr(chunk, 'chunk_id', 'unknown')}"
                
            if source not in citations:
                citations.append(source)
                
        # 6. Return response
        return RAGResponse(
            answer=llm_answer,
            citations=citations,
            route_type="general"
        )
