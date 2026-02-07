import os

import torch
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download
from langdetect import detect

# paths / constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index_nepali")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# GGUF model – TinyLlama-1.1B-Chat
GGUF_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
GGUF_FILE = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
MODELS_DIR = os.path.join(BASE_DIR, "models")

# prompt template (ChatML format for TinyLlama-Chat)
PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful AI assistant for Nepali News.
Use the following pieces of retrieved context to answer the user's question.
If the answer is not in the context, strictly say "I cannot find the answer in the provided news."
Keep the answer concise and factual.

Context:
{context}

Answer in the following language: {target_language}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""


# loaders (each called once, results are cached at module level)
def load_embeddings() -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={},
    )


def load_vectorstore(embeddings: HuggingFaceEmbeddings) -> FAISS:
    print("Loading FAISS index …")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    # set nprobe for IVF index (search 16 of 256 clusters)
    vectorstore.index.nprobe = 16
    print(
        f"Index loaded. {vectorstore.index.ntotal} vectors, "
        f"nprobe={vectorstore.index.nprobe}"
    )
    return vectorstore


def load_llm() -> LlamaCpp:
    """Download the GGUF file (once) and return a LlamaCpp LLM."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, GGUF_FILE)

    # optional: log in to Hugging Face via env var
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(hf_token)

    # download GGUF if not already cached locally
    if not os.path.exists(model_path):
        print(f"Downloading {GGUF_FILE} from {GGUF_REPO} …")
        hf_hub_download(
            repo_id=GGUF_REPO,
            filename=GGUF_FILE,
            local_dir=MODELS_DIR,
        )
        print("Download complete.")

    print(f"Loading GGUF model from {model_path} …")
    try:
        llm = LlamaCpp(
            model_path=model_path,
            n_ctx=2048,  # context window (kept modest to save RAM)
            n_batch=512,  # prompt-eval batch size
            n_gpu_layers=0,  # CPU-only
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.1,
            max_tokens=256,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load GGUF model at {model_path}. "
            f"Ensure llama-cpp-python is installed correctly: {exc}"
        ) from exc
    print("GGUF model loaded.")
    return llm


# RAG pipeline class
class RAGChain:

    def __init__(self) -> None:
        self.embeddings = load_embeddings()
        self.vectorstore = load_vectorstore(self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )
        self.llm = load_llm()

        # char budgets tuned per language (n_ctx=2048, max_tokens=256 → 1792 prompt tokens)
        # Nepali ≈ 0.55 tok/char  →  1400 chars ≈ 770 ctx tokens + ~250 overhead ≈ 1020 total
        # English ≈ 0.24 tok/char →  3200 chars ≈ 768 ctx tokens + ~150 overhead ≈  918 total
        self._max_prompt_chars = {"Nepali": 1400, "English": 3200}

    # public API
    def get_answer(
        self, query: str, language: str | None = None
    ) -> tuple[str, list[str]]:

        # input validation
        if not query or not query.strip():
            return "Please provide a question.", []

        # auto-detect language if not specified
        if language is None:
            try:
                detected = detect(query)
                language = "English" if detected == "en" else "Nepali"
            except Exception:
                language = "Nepali"

        # retrieve relevant documents
        docs = self.retriever.invoke(query)

        if not docs:
            return "I don't have information on this topic.", []

        # build context from retrieved chunks (truncate to fit context window)
        # reserve chars for the template shell + question
        template_overhead = len(PROMPT_TEMPLATE) + len(query) + 40
        budget = self._max_prompt_chars.get(
            language, 1400
        )  # default to Nepali (stricter)
        max_context_chars = budget - template_overhead

        context_parts: list[str] = []
        char_count = 0
        for d in docs:
            if char_count + len(d.page_content) > max_context_chars:
                break
            context_parts.append(d.page_content)
            char_count += len(d.page_content) + 2  # +2 for "\n\n" separator
        context_text = "\n\n".join(context_parts)

        # only include the docs we actually used for source attribution
        docs = docs[: len(context_parts)]

        # format prompt
        final_prompt = PROMPT_TEMPLATE.format(
            context=context_text,
            question=query,
            target_language=language,
        )

        # generate response
        response = self.llm.invoke(final_prompt)

        # extract assistant's answer
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]

        # strip special tokens
        response = response.replace("<|im_end|>", "").strip()

        # "I don't know" guardrail
        if "cannot find" in response.lower() or "don't have" in response.lower():
            response = "I cannot find the answer in the provided news."

        # format sources
        sources = [
            f"{d.metadata.get('source', 'Unknown')} - {d.metadata.get('heading', 'N/A')}"
            for d in docs
        ]

        return response, sources


# convenience for direct CLI testing
def main() -> None:
    rag = RAGChain()

    test_queries = [
        "What are the recent developments in education?",
        "नेपाल को प्रधानमन्त्री को हुन्?",
    ]

    for q in test_queries:
        answer, sources = rag.get_answer(q)
        print(f"\nQ: {q}")
        print(f"A: {answer}")
        for i, src in enumerate(sources[:3], 1):
            print(f"   {i}. {src}")


if __name__ == "__main__":
    main()
