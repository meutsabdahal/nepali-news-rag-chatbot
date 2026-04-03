from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import Settings
from .embeddings import get_embeddings


class Retriever:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embeddings = get_embeddings(settings.embedding_model)

        if not settings.vector_store_dir.exists():
            raise FileNotFoundError(
                f"Vector store not found at {settings.vector_store_dir}. Build index first."
            )

        self._vectorstore = FAISS.load_local(
            str(settings.vector_store_dir),
            self._embeddings,
            allow_dangerous_deserialization=settings.trusted_local_index,
        )

        if hasattr(self._vectorstore.index, "nprobe"):
            self._vectorstore.index.nprobe = 16

    def retrieve(self, query: str, k: int) -> list[Document]:
        retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
        return retriever.invoke(query)
