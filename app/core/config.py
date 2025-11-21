import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, PostgresDsn, RedisDsn, field_validator
import secrets

# Explicitly load the .env file from the project root.
# This ensures that environment variables are available for Alembic and other CLI tools.
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

class Settings(BaseSettings):
    # API Configuration
    PROJECT_NAME: str = "MEMSHADOW"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Database Configuration
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    DATABASE_URL: Optional[PostgresDsn] = None

    @field_validator("DATABASE_URL", mode="before")
    def assemble_db_connection(cls, v: Optional[str], values) -> Any:
        if isinstance(v, str):
            return v
        if 'POSTGRES_USER' not in values.data:
            return v
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=values.data.get("POSTGRES_USER"),
            password=values.data.get("POSTGRES_PASSWORD"),
            host=values.data.get("POSTGRES_SERVER"),
            path=values.data.get("POSTGRES_DB") or "",
        )

    # Redis Configuration
    REDIS_URL: RedisDsn

    # ChromaDB Configuration
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "memshadow_memories"

    # Celery Configuration
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    # Security Configuration
    ALGORITHM: str = "HS256"
    BCRYPT_ROUNDS: int = 12
    FIELD_ENCRYPTION_KEY: str

    # MFA Configuration
    MFA_ISSUER: str = "MEMSHADOW"
    FIDO2_RP_ID: str = "localhost"
    FIDO2_RP_NAME: str = "MEMSHADOW"

    # SDAP Configuration
    SDAP_BACKUP_PATH: str = "/var/backups/memshadow"
    SDAP_ARCHIVE_SERVER: str = "backup.memshadow.internal"
    SDAP_GPG_KEY_ID: str = ""

    # Corpus Import Configuration
    CORPUS_IMPORT_DIR: str = "/data/corpus"  # Autoscan directory (hourly)

    # Local AI Configuration (OpenAI-compatible backend: ollama, llama.cpp, vllm)
    LOCAL_AI_URL: str = "http://localhost:11434/v1"
    LOCAL_AI_MODEL: str = "llama3.2"

    # Embedding Configuration
    # Backend options: "sentence-transformers", "openai", "cohere"
    EMBEDDING_BACKEND: str = "sentence-transformers"

    # Sentence-Transformers Models:
    # - "sentence-transformers/all-mpnet-base-v2" (768d) - Balanced, fast
    # - "BAAI/bge-large-en-v1.5" (1024d) - High quality, recommended
    # - "thenlper/gte-large" (1024d) - Alternative high quality
    # - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" (768d→2048d with projection)
    #
    # OpenAI Models:
    # - "text-embedding-3-small" (1536d) - Cost effective
    # - "text-embedding-3-large" (3072d, configurable) - Highest quality
    #
    # For 2048d: Set EMBEDDING_MODEL to any base model and EMBEDDING_DIMENSION to 2048
    # The system will add a projection layer if needed
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSION: int = 2048
    EMBEDDING_CACHE_TTL: int = 3600
    EMBEDDING_USE_PROJECTION: bool = True  # Auto-project to EMBEDDING_DIMENSION if model != target

    # OpenAI Configuration (if EMBEDDING_BACKEND="openai")
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"

    # Advanced NLP Configuration
    USE_ADVANCED_NLP: bool = True
    NLP_QUERY_EXPANSION: bool = True
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.7

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()