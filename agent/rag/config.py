"""RAG layer configuration, read from the environment and the root `.env`."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# agent/rag/config.py -> agent/rag -> agent -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """RAG layer settings.

    Every value has a working default: `pnpm ingest` runs without a `.env`,
    provided a Qdrant instance listens on the default port.
    """

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Embeddings -----------------------------------------------------
    # `fastembed` (local, offline) or `mistral` (cloud API, EU-hosted).
    embedding_provider: str = "fastembed"
    # 0.22 GB, 384 dim, multilingual, apache-2.0 — chosen to keep the demo
    # lightweight and fully offline.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    mistral_api_key: str | None = None

    # --- Qdrant ---------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    # Suffixed with the model at use time (see `collection_name`).
    qdrant_collection_prefix: str = "regwatch"

    # --- Corpus ---------------------------------------------------------
    corpus_dir: Path = REPO_ROOT / "data" / "corpus"
    manifest_path: Path = REPO_ROOT / "corpus-manifest.json"

    # --- Retrieval ------------------------------------------------------
    top_k: int = 8
    # Below this score, no chunk is considered grounded enough to answer from,
    # and the caller is expected to refuse explicitly.
    score_threshold: float = 0.5

    # --- LLM --------------------------------------------------------------
    llm_model: str = "mistral-small-2603"
    llm_temperature: float = 0.0
    llm_timeout: int = 60

    @property
    def collection_name(self) -> str:
        """Collection name suffixed with the active model.

        Switching models changes the vector dimension; without this suffix a
        re-ingestion would silently overwrite an incompatible collection.
        """
        return f"{self.qdrant_collection_prefix}_{slugify_model(self.embedding_model)}"


def slugify_model(model_id: str) -> str:
    """Reduce a model identifier to a readable collection suffix.

    >>> slugify_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    'paraphrase_multilingual_minilm_l12_v2'
    """
    tail = model_id.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9]+", "_", tail).strip("_")


def get_settings() -> Settings:
    return Settings()
