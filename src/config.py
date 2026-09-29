import math

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    LLM_API_KEY: str = "placeholder"
    LLM_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    LLM_MODEL: str = "gemma-4-26b-a4b-it"
    LLM_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_RETRIES: int = 1
    JWT_SECRET_KEY: str = "secret"
    JWT_EXPIRY_HOURS: int = 24
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/admitai.db"
    CORS_ORIGINS: str = "http://localhost:3000"
    
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "admitai_chunks"
    QDRANT_UPSERT_BATCH_SIZE: int = 128
    
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_BACKEND: str = "local"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_REMOTE_PROVIDER: str = "huggingface_endpoint"
    RERANKER_REMOTE_URL: str = ""
    RERANKER_REMOTE_API_TOKEN: SecretStr = SecretStr("")
    RERANKER_REMOTE_TIMEOUT_SECONDS: float = 30.0
    RERANKER_REMOTE_MAX_RETRIES: int = 1
    RERANKER_REMOTE_TOP_K_LIMIT: int = 20
    
    RETRIEVAL_TOP_K: int = 20
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = 3
    RERANK_TOP_K: int = 5
    RERANK_BATCH_SIZE: int = 16
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    TABLE_CHUNK_SIZE: int = 500
    PARENT_CHUNK_SIZE: int = 1500
    CHILD_CHUNK_SIZE: int = 300
    EMBEDDING_BATCH_SIZE: int = 32
    INDEX_DIR: str = "data/index/naive_dense"
    RETRIEVAL_MIN_SCORE: float = 0.68
    RETRIEVAL_MIN_LEXICAL_COVERAGE: float = 0.50

    @model_validator(mode="after")
    def validate_reranker_settings(self) -> "Settings":
        self.RERANKER_BACKEND = self.RERANKER_BACKEND.strip().lower()
        self.RERANKER_REMOTE_PROVIDER = self.RERANKER_REMOTE_PROVIDER.strip().lower()
        self.RERANKER_REMOTE_URL = self.RERANKER_REMOTE_URL.strip()

        if self.RERANKER_BACKEND not in {"local", "remote"}:
            raise ValueError("RERANKER_BACKEND must be 'local' or 'remote'")
        if self.RERANKER_REMOTE_PROVIDER != "huggingface_endpoint":
            raise ValueError(
                "RERANKER_REMOTE_PROVIDER must be 'huggingface_endpoint'"
            )
        if not math.isfinite(self.RERANKER_REMOTE_TIMEOUT_SECONDS) or self.RERANKER_REMOTE_TIMEOUT_SECONDS <= 0:
            raise ValueError("RERANKER_REMOTE_TIMEOUT_SECONDS must be greater than 0")
        if not 0 <= self.RERANKER_REMOTE_MAX_RETRIES <= 5:
            raise ValueError("RERANKER_REMOTE_MAX_RETRIES must be between 0 and 5")
        if not 1 <= self.RERANKER_REMOTE_TOP_K_LIMIT <= 20:
            raise ValueError("RERANKER_REMOTE_TOP_K_LIMIT must be between 1 and 20")
        if self.RERANKER_BACKEND == "remote":
            if not self.RERANKER_REMOTE_URL:
                raise ValueError(
                    "RERANKER_REMOTE_URL is required when RERANKER_BACKEND=remote"
                )
            if not self.RERANKER_REMOTE_API_TOKEN.get_secret_value().strip():
                raise ValueError(
                    "RERANKER_REMOTE_API_TOKEN is required when RERANKER_BACKEND=remote"
                )
        return self

    class Config:
        env_file = ".env"

settings = Settings()
