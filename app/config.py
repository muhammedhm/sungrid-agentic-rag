"""
Central configuration for the SunGrid Cooperative Copilot.

All values can be overridden via environment variables or a `.env` file. Nothing here should ever contain a real secret --
`groq_api_key` is read from the environment and never given a default.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (Groq) ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0

    # --- Embeddings (local, no API key required) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Retrieval / vector store ---
    docs_dir: str = str(REPO_ROOT / "docs")
    chroma_persist_dir: str = str(REPO_ROOT / ".chroma")
    collection_name: str = "sungrid_docs"
    top_k: int = 4
    max_chunk_characters: int = 1200
    new_chunk_after_characters: int = 1000
    combine_chunk_under_characters: int = 200

    # --- Agent control flow ---
    max_retrieval_retries: int = 2  # how many times the agent may rewrite+retry a query
    max_agent_steps: int = 6        # hard cap on graph iterations, avoids runaway loops

    # --- App ---
    log_level: str = "INFO"


settings = Settings()
