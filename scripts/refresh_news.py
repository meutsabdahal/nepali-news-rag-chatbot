from __future__ import annotations

from nepali_news_rag.config import get_settings
from nepali_news_rag.data_prep import load_chunks
from nepali_news_rag.index_builder import build_faiss_index


def main() -> None:
    settings = get_settings()

    if not settings.chunks_pkl_path.exists():
        raise FileNotFoundError(
            "Missing chunks artifact. Run scripts/build_db.py before refresh_news.py"
        )

    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(str(settings.chunks_pkl_path))
    build_faiss_index(
        chunks=chunks,
        embedding_model=settings.embedding_model,
        output_dir=str(settings.vector_store_dir),
    )
    print("News refresh and vector indexing completed.")


if __name__ == "__main__":
    main()
