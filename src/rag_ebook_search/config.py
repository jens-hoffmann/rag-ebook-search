"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/rag_ebooks"

    # LMStudio OpenAI-compatible API
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "lm-studio"

    # Models
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    llm_model: str = "llama-3.2-3b-instruct"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000


settings = Settings()
