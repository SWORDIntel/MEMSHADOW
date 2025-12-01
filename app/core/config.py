import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, PostgresDsn, RedisDsn, field_validator
import secrets

# Explicitly load the .env file from the project root.
# This ensures that environment variables are available for Alembic and other CLI tools.
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)


class MemoryOperationMode(str, Enum):
    """
    Operation modes for memory processing.

    - LOCAL: Full enrichment, all features enabled (highest quality, slowest)
    - ONLINE: Balanced speed/features (moderate enrichment, good performance)
    - LIGHTWEIGHT: Minimal processing (fastest, basic features only)
    """
    LOCAL = "local"
    ONLINE = "online"
    LIGHTWEIGHT = "lightweight"


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

    # Memory Operation Mode Configuration
    MEMORY_OPERATION_MODE: MemoryOperationMode = MemoryOperationMode.LOCAL

    # =============================================================
    # Neural Storage Configuration - Brain-like Multi-Tiered System
    # =============================================================

    # Enable the neural storage system (multi-tiered brain-like storage)
    NEURAL_STORAGE_ENABLED: bool = True

    # Tiered Database Configuration
    # Tier 0: Ultra-High Dimensional (4096d) - Archival, maximum fidelity
    # Tier 1: High Dimensional (2048d) - Long-term storage
    # Tier 2: Medium Dimensional (1024d) - Warm storage
    # Tier 3: Low Dimensional (512d) - Hot compressed
    # Tier 4: RAMDISK (256d) - Working memory
    NEURAL_STORAGE_ENABLE_ULTRA_HIGH_TIER: bool = True
    NEURAL_STORAGE_RAMDISK_MAX_MB: int = 512
    NEURAL_STORAGE_RAMDISK_MIN_MB: int = 64

    # Dynamic CPU Management
    NEURAL_STORAGE_MIN_WORKERS: int = 1
    NEURAL_STORAGE_MAX_WORKERS: Optional[int] = None  # None = auto-detect CPU count
    NEURAL_STORAGE_TARGET_CPU_UTILIZATION: float = 0.7

    # Memory Migration (Hot/Cold Storage)
    NEURAL_STORAGE_PROMOTE_TEMPERATURE: float = 0.8  # Promote to faster tier if temp > this
    NEURAL_STORAGE_DEMOTE_TEMPERATURE: float = 0.2   # Demote to slower tier if temp < this
    NEURAL_STORAGE_MIN_AGE_FOR_DEMOTION_HOURS: float = 24.0
    NEURAL_STORAGE_MAX_IDLE_HOURS_ARCHIVE: float = 168.0  # 1 week

    # Neural Connection Engine (Brain-like pattern discovery)
    NEURAL_STORAGE_SIMILARITY_THRESHOLD: float = 0.7
    NEURAL_STORAGE_MAX_CONNECTIONS_PER_MEMORY: int = 100
    NEURAL_STORAGE_HEBBIAN_LEARNING_RATE: float = 0.1
    NEURAL_STORAGE_CONNECTION_DECAY_RATE: float = 0.01

    # Cross-tier Deduplication
    NEURAL_STORAGE_SEMANTIC_DEDUP_THRESHOLD: float = 0.95
    NEURAL_STORAGE_NEAR_DUPLICATE_THRESHOLD: float = 0.85
    NEURAL_STORAGE_AUTO_DEDUP_INTERVAL_MINUTES: int = 30

    # Background Tasks
    NEURAL_STORAGE_ENABLE_BACKGROUND_TASKS: bool = True
    NEURAL_STORAGE_MAINTENANCE_INTERVAL_MINUTES: int = 15
    NEURAL_STORAGE_CONNECTION_DISCOVERY_INTERVAL_MINUTES: int = 5

    # Persistence
    NEURAL_STORAGE_PERSISTENCE_PATH: str = "/var/lib/memshadow/neural_storage"
    NEURAL_STORAGE_ENABLE_PERSISTENCE: bool = True

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()