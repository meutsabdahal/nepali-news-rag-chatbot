from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    project_root: Path
    raw_csv_path: Path
    chunks_pkl_path: Path
    vector_store_dir: Path
    embedding_model: str
    llm_provider: str
    response_max_tokens: int
    retriever_k: int
    trusted_local_index: bool
    ollama_host: str
    ollama_model: str
    groq_api_key: str | None
    groq_model: str
    hf_api_key: str | None
    hf_model: str


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parent.parent

    # Load project-level .env so local runs pick up provider/model settings.
    load_dotenv(project_root / ".env", override=False)

    return Settings(
        project_root=project_root,
        raw_csv_path=project_root / "data" / "raw" / "np20ng.csv",
        chunks_pkl_path=project_root / "data" / "processed" / "nepali_news_chunks.pkl",
        vector_store_dir=project_root / "data" / "vector_store" / "faiss_index",
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
        response_max_tokens=_env_int("MAX_TOKENS", 256),
        retriever_k=_env_int("TOP_K_RAG", 3),
        trusted_local_index=_env_bool("TRUST_LOCAL_INDEX", True),
        ollama_host=(
            os.getenv("OLLAMA_HOST")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        hf_api_key=os.getenv("HF_API_KEY"),
        hf_model=os.getenv("HF_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
    )
