import os
import pickle

import faiss
import numpy as np
import torch
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore

from data_prep import load_chunks

# paths (relative to project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index_nepali")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "faiss_index_nepali_checkpoint")

# hyper-parameters
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # MiniLM output dimension
N_CLUSTERS = 256  # number of IVF clusters
BATCH_SIZE = 5000
CHECKPOINT_EVERY = 10  # save every N batches


# helpers
def get_hf_embeddings() -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(chunks: list, hf_embeddings: HuggingFaceEmbeddings) -> FAISS:

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FINAL_INDEX_DIR, exist_ok=True)

    # embed a training sample (first 50k chunks)
    sample_texts = [c.page_content for c in chunks[:50_000]]
    all_vectors = []

    for i in tqdm(
        range(0, len(sample_texts), BATCH_SIZE), desc="Embedding training sample"
    ):
        batch = sample_texts[i : i + BATCH_SIZE]
        batch_vectors = hf_embeddings.embed_documents(batch)
        all_vectors.extend(batch_vectors)

    sample_vectors = np.array(all_vectors, dtype=np.float32)
    print(f"Training sample shape: {sample_vectors.shape}")

    # build and train the IVF index
    quantizer = faiss.IndexFlatL2(EMBEDDING_DIM)
    index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, N_CLUSTERS, faiss.METRIC_L2)
    index.train(sample_vectors)
    print(f"IVF index trained. {index.ntotal} vectors in index (training only).")

    # wrap in LangChain FAISS vectorstore
    vectorstore = FAISS(
        embedding_function=hf_embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    print("Vectorstore ready with trained IVF index.")

    # batch-insert all chunks with periodic checkpoints
    num_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(
        range(0, len(chunks), BATCH_SIZE),
        total=num_batches,
        desc="Embedding batches",
    ):
        batch = chunks[i : i + BATCH_SIZE]
        vectorstore.add_documents(batch)

        batch_num = (i // BATCH_SIZE) + 1

        if batch_num % CHECKPOINT_EVERY == 0:
            vectorstore.save_local(CHECKPOINT_DIR)
            tqdm.write(
                f"Batch {batch_num:>4d} | chunks {i:>7d} → {i + len(batch):>7d} "
                f"| Checkpoint saved | total: {index.ntotal}"
            )
        else:
            tqdm.write(
                f"Batch {batch_num:>4d} | chunks {i:>7d} → {i + len(batch):>7d} "
                f"| total: {index.ntotal}"
            )

    print(f"\nDone inserting. Final index size: {index.ntotal} vectors.")

    # save final index
    vectorstore.save_local(FINAL_INDEX_DIR)
    print(f"Final FAISS index saved to {FINAL_INDEX_DIR}")

    return vectorstore


# CLI entry-point
def main() -> None:
    chunks = load_chunks()
    hf_embeddings = get_hf_embeddings()
    build_index(chunks, hf_embeddings)


if __name__ == "__main__":
    main()
