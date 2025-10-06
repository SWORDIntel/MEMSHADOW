from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, PostgresDsn, RedisDsn, field_validator
import secrets

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

    # MFA Configuration
    MFA_ISSUER: str = "MEMSHADOW"
    FIDO2_RP_ID: str = "localhost"
    FIDO2_RP_NAME: str = "MEMSHADOW"

    # SDAP Configuration
    SDAP_BACKUP_PATH: str = "/var/backups/memshadow"
    SDAP_ARCHIVE_SERVER: str = "backup.memshadow.internal"
    SDAP_GPG_KEY_ID: str = ""

    # Embedding Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIMENSION: int = 768
    EMBEDDING_CACHE_TTL: int = 3600

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()