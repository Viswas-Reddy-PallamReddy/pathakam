from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    """All settings come from environment variables prefixed PATHAKAM_ (or the repo's .env file)."""

    model_config = SettingsConfigDict(
        env_prefix="PATHAKAM_", env_file=REPO_ROOT / ".env", extra="ignore"
    )

    # Chat model
    llm_provider: Literal["ollama", "openrouter"] = "ollama"
    llm_model: str = "gemma3:4b"
    temperature: float = 0.1

    # Embeddings ("none" = keyword search only)
    embed_provider: Literal["ollama", "none"] = "ollama"
    embed_model: str = "bge-m3"

    ollama_url: str = "http://127.0.0.1:11434"
    openrouter_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""

    # Storage and retrieval
    db_path: Path = DATA_DIR / "pathakam.db"
    top_k: int = 6
    max_history_turns: int = 6

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
