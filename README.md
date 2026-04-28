# Nepali News RAG Chatbot

A cross-lingual Retrieval-Augmented Generation (RAG) chatbot for Nepali news.

Ask questions in English or Nepali, retrieve grounded evidence from the `np20ng` corpus, and get answers with source references.

## What It Does

This project provides a practical RAG pipeline focused on Nepali-language news:

- Cross-lingual query support (English/Nepali input).
- Retrieval over a local FAISS vector store built from Nepali news chunks.
- Safety guardrail for speculative prediction prompts.
- Language-aware responses with source transparency.
- Streamlit chat interface for quick experimentation.

## Current Status

Implemented:

- End-to-end `RAG`, `DIRECT`, and `OOS` route handling.
- Dataset preparation and chunking pipeline from raw CSV.
- FAISS index build and local retrieval flow.
- Multiple LLM providers (`ollama`, `groq`, `huggingface`).
- Benchmark runner and smoke-check script.
- Contract and guardrail regression tests.

Not in scope in this repository:

- Real-time web/news crawling.
- Price/fundamental SQL analytics (this repo is news-RAG-only).

## Architecture

High-level flow:

`Streamlit UI (app/main.py) -> NepaliNewsPipeline -> Router/Guardrails -> Retriever (FAISS) -> LLM -> Answer + Sources`

Core behavior:

- `DIRECT`: greetings/general factual queries that do not need corpus retrieval.
- `RAG`: news-grounded queries answered from retrieved context.
- `OOS`: prediction-style requests blocked with a safe refusal.

## Repository Layout

- `app/main.py`: Streamlit app entrypoint.
- `app/components.py`: response/source rendering helpers.
- `nepali_news_rag/pipeline.py`: core orchestration for routing, retrieval, and generation.
- `nepali_news_rag/router.py`: route decision logic (`DIRECT`, `RAG`, `OOS`).
- `nepali_news_rag/guardrails.py`: prediction-guardrail policy.
- `nepali_news_rag/data_prep.py`: cleaning, validation, chunking, and chunk artifact IO.
- `nepali_news_rag/index_builder.py`: FAISS index training/building.
- `nepali_news_rag/retriever.py`: FAISS load and retrieval wrapper.
- `nepali_news_rag/llm.py`: provider adapters.
- `scripts/build_db.py`: build chunk artifact from raw CSV.
- `scripts/refresh_news.py`: build/refresh vector index from chunks.
- `scripts/evaluate_benchmark.py`: benchmark questions runner.
- `scripts/smoke_check.py`: compile + tests (+ optional benchmark) sanity checks.
- `tests/`: unit and contract tests.

## Quick Start (uv-native)

Prerequisites:

- Python 3.14+
- `uv` installed
- Raw dataset file available at `data/raw/np20ng.csv`

1. Sync dependencies

```bash
uv sync
```

2. Configure environment

```bash
cp .env.example .env
```

Then set values in `.env` as needed:

- `LLM_PROVIDER=ollama|groq|hf`
- `OLLAMA_HOST`, `OLLAMA_MODEL` (for local Ollama)
- `GROQ_API_KEY`, `GROQ_MODEL` (for Groq)
- `HF_API_KEY`, `HF_LLM_MODEL` (for Hugging Face)

3. Build local artifacts

```bash
uv run news-build-db
uv run news-refresh-news
```

These commands create:

- `data/processed/nepali_news_chunks.pkl`
- `data/vector_store/faiss_index/`

4. Run the app

```bash
uv run news-app
```

Streamlit should open locally and serve the chat UI.

## Useful Commands

Project health and diagnostics:

```bash
uv run news-doctor
```

Run smoke checks (compile + tests):

```bash
uv run news-smoke
```

Include a small benchmark in smoke checks:

```bash
uv run news-smoke --with-benchmark --benchmark-limit 2
```

Run benchmark directly:

```bash
uv run news-evaluate --limit 20 --output latest_results.json
```

Results are written to `evaluation/results/`.

## Evaluation

Benchmark questions live in `evaluation/benchmark_questions.json`.

The evaluation script reports:

- Route match count/rate against expected route labels.
- Per-question outputs, sources, and error capture.
- Timestamped run summary metadata.

Example:

```bash
uv run python scripts/evaluate_benchmark.py --limit 10 --output report_local.json
```

## Tests

Run the test suite:

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

Current coverage focus:

- Guardrail detection behavior.
- Provider factory behavior.
- Pipeline response contract shape and routing behavior.

## Troubleshooting

- `Vector store not found ...`:
  Build artifacts first with `uv run news-build-db` then `uv run news-refresh-news`.

- `GROQ_API_KEY is required ...`:
  Set `GROQ_API_KEY` when using `LLM_PROVIDER=groq`.

- Ollama memory/provider failures:
  Switch to Groq/HF provider in `.env`, or use a smaller/local model that fits your machine.

- No answer from retrieval:
  Confirm raw data quality and rebuild chunks/index.

## Data Notes

- This project assumes a static local news corpus (`np20ng.csv`).
- Source data ownership remains with original publishers/dataset maintainers.
- Use this repository for educational/research purposes.

## License

MIT License. See `LICENSE`.
