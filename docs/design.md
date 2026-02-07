### 1. System Overview

**Project Name:** Nepali News RAG Chatbot <br>
**Goal:** A retrieval-augmented generation system capable of answering queries in English or Nepali based on a dataset of 10k+ Nepali news articles. <br>

**Stack:**
* **Environment:** Local CPU (8 GB RAM, no GPU)
* **LLM:** `TinyLlama-1.1B-Chat-v1.0` (Q8_0 GGUF via `llama-cpp-python`)
* **Orchestration:** LangChain
* **Vector Store:** FAISS (CPU)
* **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* **UI:** Streamlit

### 2. Architecture Diagram
![Architecture Diagram](./diagram/architecture_diagram.svg)

### 3. Data Pipeline & Logic

1. **Ingestion:**
* Load `np20ng` dataset.
* **Cleaning:** Fix encoding artifacts (`¥`  `र्`), normalize Unicode (`NFKC`), collapse whitespace.


2. **Chunking Strategy:**
* **Splitter:** `RecursiveCharacterTextSplitter`
* **Separators:** `["\n\n", "\n", " ।", "।", "|", " ", ""]` (Priority given to Nepali Purna Viram).


3. **Indexing:**
* Embed chunks using `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions).
* Store in FAISS (FlatL2 index for accuracy, IVF for speed if >50k docs).



### 4. Design Constraints (The Math)

*Based on experiments (Feb 2026):*

* **Token Ratio:** 0.55 tokens/char (Nepali) vs 0.24 tokens/char (English).
* **Max Context Window:** 2048 tokens (TinyLlama).
* **Chunk Size:** 1000 characters (~550 tokens Nepali).
* **Chunk Overlap:** 200 characters.
* **Retrieval Count (k):** 3 documents.
* **Max Generation Tokens:** 256.
* **Prompt Token Budget:** ~1792 tokens (2048 − 256).
* **Char Budget (Nepali):** 1400 chars (~770 tokens context) — fits within budget.
* **Char Budget (English):** 3200 chars (~768 tokens context) — fits within budget.
* **Note:** Language-aware truncation ensures Nepali prompts (with their higher token density) never exceed the context window.

### 5. Prompt Engineering

We use the ChatML instruct format expected by TinyLlama-Chat.

**Template:**

```text
<|im_start|>system
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
```

### 6. User Interface (Streamlit)

**Inputs:**

1. `Query` — Chat input box.
2. `Response Language` — Sidebar radio: "Auto-detect", "Nepali", "English". Controls the `{target_language}` variable in prompt.

**Outputs:**

1. `Answer` — Rendered as Markdown in a chat bubble.
2. `Sources` — Expandable list of outlet names + article headlines.

### 7. Limitations & Future Scalability

* **Nepali Generation Quality:** TinyLlama-1.1B was primarily trained on English; its Nepali output is noticeably weaker. Upgrading to a larger multilingual model (e.g., Phi-3.5-mini, Llama-3.1-8B) on a machine with a GPU or more RAM would significantly improve Nepali answers.
* **Context Window:** TinyLlama's 2048-token limit restricts retrieval to k=3 chunks. A model with a larger context (e.g., 8k–128k) would allow more evidence per query.
* **Storage:** Current in-memory FAISS works for <100k docs. For >1M, migrate to Pinecone or Weaviate.
* **Latency:** CPU-only inference is slow (~15–30 s per answer). A GPU would reduce this to <3 s.
* **Multi-turn:** Currently stateless (single QA). Conversation memory can be added in a future phase.
