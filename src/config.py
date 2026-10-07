from pathlib import Path

from pydantic_settings import BaseSettings


DEFAULT_HUGGINGFACE_CACHE_DIR = str(
    Path(__file__).resolve().parents[1] / "models" / "huggingface" / "hub"
)


class Settings(BaseSettings):
    LLM_API_KEY: str = "placeholder"
    LLM_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    LLM_MODEL: str = "gemma-4-26b-a4b-it"
    LLM_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_RETRIES: int = 1
    JWT_SECRET_KEY: str = "secret"
    JWT_EXPIRY_HOURS: int = 24
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/admitai.db"
    
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "admitai_chunks"
    QDRANT_UPSERT_BATCH_SIZE: int = 128
    
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    HUGGINGFACE_CACHE_DIR: str = DEFAULT_HUGGINGFACE_CACHE_DIR
    
    RETRIEVAL_TOP_K: int = 20
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = 3
    RERANK_TOP_K: int = 5
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    TABLE_CHUNK_SIZE: int = 500
    PARENT_CHUNK_SIZE: int = 1500
    CHILD_CHUNK_SIZE: int = 300
    EMBEDDING_BATCH_SIZE: int = 32
    INDEX_DIR: str = "data/index/naive_dense"
    RETRIEVAL_MIN_SCORE: float = 0.68
    RETRIEVAL_MIN_LEXICAL_COVERAGE: float = 0.50

    class Config:
        env_file = ".env"

settings = Settings()
