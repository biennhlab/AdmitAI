from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    NVIDIA_API_KEY: str = "placeholder"
    LLM_MODEL: str = "meta/llama-3.1-70b-instruct"
    JWT_SECRET_KEY: str = "secret"
    JWT_EXPIRY_HOURS: int = 24
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/admitai.db"
    
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "admitai_chunks"
    
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    
    RETRIEVAL_TOP_K: int = 20
    RERANK_TOP_K: int = 5
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    EMBEDDING_BATCH_SIZE: int = 32
    INDEX_DIR: str = "data/index/naive_dense"
    RETRIEVAL_MIN_SCORE: float = 0.68
    RETRIEVAL_MIN_LEXICAL_COVERAGE: float = 0.50

    class Config:
        env_file = ".env"

settings = Settings()
