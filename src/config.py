from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings
from src.local_profile import LOCAL_INDEX_DIR, LOCAL_COLLECTION, LOCAL_EMBEDDING_MODEL


DEFAULT_HUGGINGFACE_CACHE_DIR = str(
    Path(__file__).resolve().parents[1] / "models" / "huggingface" / "hub"
)


class Settings(BaseSettings):
    LOCAL_DEV_PROFILE: bool = True
    MODELS_LOCAL_ONLY: bool = True
    RAG_RECOVERY_COOLDOWN_SECONDS: float = Field(default=60.0, gt=0, allow_inf_nan=False)
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
    QDRANT_COLLECTION: str = LOCAL_COLLECTION
    QDRANT_UPSERT_BATCH_SIZE: int = 128
    
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_BACKEND: Literal["neural", "lite"] = "lite"
    QUERY_REWRITE_ENABLED: bool = False
    HUGGINGFACE_CACHE_DIR: str = DEFAULT_HUGGINGFACE_CACHE_DIR
    HF_TOKEN: str | None = None
    
    RETRIEVAL_TOP_K: int = 20
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = 3
    RERANK_CANDIDATE_K: int = Field(default=8, gt=0)
    RERANK_TOP_K: int = 5
    LITE_CANDIDATE_K: int = Field(default=60, gt=0, le=200)
    LITE_MAX_BRANCH_CANDIDATES: int = Field(default=360, gt=0, le=2000)
    LITE_CONTEXT_MAX_CHARS: int = Field(default=12000, ge=1000, le=32000)
    LITE_RRF_K: float = Field(default=60.0, ge=0, allow_inf_nan=False)
    LITE_DENSE_WEIGHT: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    LITE_SPARSE_WEIGHT: float = Field(default=1.5, ge=0, allow_inf_nan=False)
    LITE_DOMAIN_MIN_FACTOR: float = Field(default=0.25, gt=0, allow_inf_nan=False)
    LITE_DOMAIN_MAX_FACTOR: float = Field(default=4.0, gt=0, allow_inf_nan=False)
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    TABLE_CHUNK_SIZE: int = 500
    PARENT_CHUNK_SIZE: int = 1500
    CHILD_CHUNK_SIZE: int = 300
    EMBEDDING_BATCH_SIZE: int = 32
    INDEX_DIR: str = LOCAL_INDEX_DIR
    RETRIEVAL_MIN_SCORE: float = 0.68
    RETRIEVAL_MIN_LEXICAL_COVERAGE: float = 0.50

    @model_validator(mode="after")
    def validate_lite_parameters(self):
        if self.LOCAL_DEV_PROFILE:
            if self.EMBEDDING_MODEL != LOCAL_EMBEDDING_MODEL:
                raise ValueError("The local profile requires the existing BGE-M3 embedding model")
            from urllib.parse import urlsplit
            if urlsplit(self.LLM_BASE_URL).hostname != "generativelanguage.googleapis.com":
                raise ValueError("The local profile requires the configured Google AI endpoint")
            self.INDEX_DIR = LOCAL_INDEX_DIR
            self.QDRANT_COLLECTION = LOCAL_COLLECTION
            self.RERANK_BACKEND = "lite"
            self.QUERY_REWRITE_ENABLED = False
            self.MODELS_LOCAL_ONLY = True
        if self.LITE_DENSE_WEIGHT == self.LITE_SPARSE_WEIGHT == 0:
            raise ValueError("At least one lite RRF weight must be positive")
        if self.LITE_DOMAIN_MIN_FACTOR > self.LITE_DOMAIN_MAX_FACTOR:
            raise ValueError("Lite domain factor bounds are reversed")
        return self

    class Config:
        env_file = ".env"

settings = Settings()
