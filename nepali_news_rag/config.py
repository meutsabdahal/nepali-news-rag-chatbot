from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    raw_csv_path: Path
    chunks_pkl_path: Path
    vector_store_dir: Path
    model_cache_dir: Path
    embedding_model: str
    llm_provider: str
    response_max_tokens: int
    retriever_k: int
    trusted_local_index: bool
    llama_gguf_repo: str
    llama_gguf_file: str
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

    return Settings(
        project_root=project_root,
        raw_csv_path=project_root / "data" / "raw" / "np20ng.csv",
        chunks_pkl_path=project_root / "data" / "processed" / "nepali_news_chunks.pkl",
        vector_store_dir=project_root / "data" / "vector_store" / "faiss_index",
        model_cache_dir=project_root / "models",
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        llm_provider=os.getenv("LLM_PROVIDER", "llama_cpp").strip().lower(),
        response_max_tokens=_env_int("MAX_TOKENS", 256),
        retriever_k=_env_int("TOP_K_RAG", 3),
        trusted_local_index=_env_bool("TRUST_LOCAL_INDEX", True),
        llama_gguf_repo=os.getenv(
            "LLAMA_GGUF_REPO", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
        ),
        llama_gguf_file=os.getenv(
            "LLAMA_GGUF_FILE", "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
        ),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "tinyllama:latest"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        hf_api_key=os.getenv("HF_API_KEY"),
        hf_model=os.getenv("HF_LLM_MODEL", "HuggingFaceH4/zephyr-7b-beta"),
    )
