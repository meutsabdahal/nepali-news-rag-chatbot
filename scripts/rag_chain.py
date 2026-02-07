import os

import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
from langdetect import detect

# paths / constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index_nepali")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

# prompt template (Phi-3 instruct format)
PROMPT_TEMPLATE = """<|system|>
You are a helpful AI assistant for Nepali News.
Use the following pieces of retrieved context to answer the user's question.
If the answer is not in the context, strictly say "I cannot find the answer in the provided news."
Keep the answer concise and factual.

Context:
{context}

Answer in the following language: {target_language}<|end|>
<|user|>
{question}<|end|>
<|assistant|>
"""


# loaders (each called once, results are cached at module level) ───────────
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


def load_llm() -> HuggingFacePipeline:
    # optional: log in to Hugging Face via env var
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(hf_token)

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {LLM_MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print("Model loaded.")

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    print("Pipeline ready.")
    return llm


# RAG pipeline class
class RAGChain:

    def __init__(self) -> None:
        self.embeddings = load_embeddings()
        self.vectorstore = load_vectorstore(self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15},
        )
        self.llm = load_llm()

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

        # build context from retrieved chunks
        context_text = "\n\n".join([d.page_content for d in docs])

        # format prompt
        final_prompt = PROMPT_TEMPLATE.format(
            context=context_text,
            question=query,
            target_language=language,
        )

        # generate response
        response = self.llm.invoke(final_prompt)

        # extract assistant's answer
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]

        # strip special tokens
        response = response.replace("<|end|>", "").strip()

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
