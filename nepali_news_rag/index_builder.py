from __future__ import annotations

import random

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

from .embeddings import get_embeddings


def build_faiss_index(
    chunks: list,
    embedding_model: str,
    output_dir: str,
    n_clusters: int = 256,
    batch_size: int = 5000,
) -> FAISS:
    embeddings = get_embeddings(embedding_model)

    sample_pool = [c.page_content for c in chunks]
    random.shuffle(sample_pool)
    sample_texts = sample_pool[: min(50_000, len(sample_pool))]

    vectors = []
    for i in tqdm(range(0, len(sample_texts), batch_size), desc="Embedding training sample"):
        vectors.extend(embeddings.embed_documents(sample_texts[i : i + batch_size]))

    sample_vectors = np.asarray(vectors, dtype=np.float32)
    dim = sample_vectors.shape[1]

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_L2)
    index.train(sample_vectors)

    vectorstore = FAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        vectorstore.add_documents(chunks[i : i + batch_size])

    vectorstore.save_local(output_dir)
    print(f"Saved FAISS index at {output_dir} with {index.ntotal} vectors")
    return vectorstore
