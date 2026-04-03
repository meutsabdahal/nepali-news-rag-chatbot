from __future__ import annotations

from nepali_news_rag.config import get_settings


def _state(exists: bool) -> str:
    return "OK" if exists else "MISSING"


def main() -> None:
    settings = get_settings()

    checks = {
        "raw_csv": settings.raw_csv_path.exists(),
        "chunks_pkl": settings.chunks_pkl_path.exists(),
        "vector_store_dir": settings.vector_store_dir.exists(),
    }

    print("Project doctor report")
    print("-" * 24)
    print(f"provider: {settings.llm_provider}")
    print(f"raw_csv: {_state(checks['raw_csv'])} ({settings.raw_csv_path})")
    print(f"chunks: {_state(checks['chunks_pkl'])} ({settings.chunks_pkl_path})")
    print(
        f"vector_store: {_state(checks['vector_store_dir'])} ({settings.vector_store_dir})"
    )

    if settings.llm_provider == "groq" and not settings.groq_api_key:
        print("warning: GROQ_API_KEY is empty")
    if settings.llm_provider in {"hf", "huggingface"} and not settings.hf_api_key:
        print("warning: HF_API_KEY is empty")

    if checks["raw_csv"] and not checks["chunks_pkl"]:
        print("hint: run news-build-db")
    if checks["chunks_pkl"] and not checks["vector_store_dir"]:
        print("hint: run news-refresh-news")


if __name__ == "__main__":
    main()
