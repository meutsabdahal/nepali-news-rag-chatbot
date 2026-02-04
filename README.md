# Product Requirements Document (Lite)

**Project Name:** Nepali News RAG (Cross-Lingual Q&A Bot) <br>
**Version:** 0.1.0 (MVP) <br>
**Status:** In Development

## 1. Executive Summary

This project aims to build a **Retrieval-Augmented Generation (RAG)** system that allows users to query a massive static dataset of Nepali news articles (`np20ng`) using natural language. The system features **cross-lingual capabilities**, enabling users to ask questions in **English or Nepali** and receive accurate, grounded answers in their preferred language, bridging the gap between global users and local information.

## 2. Problem Statement

* **The Search Gap:** Standard keyword search fails to capture context or synonyms in the Nepali language (e.g., searching "disaster" might miss "badhipairo").
* **The Language Barrier:** Non-Nepali speakers (researchers, tourists, expats) cannot access valuable local knowledge locked inside Nepali-only documents.
* **Hallucinations:** Generic LLMs (like ChatGPT) often hallucinate facts about niche local events when not grounded in specific data.

## 3. Target Audience

* **Primary:** Recruiters/Hiring Managers (Proof of RAG & NLP skills).
* **Secondary:** Researchers, NGOs, or Tourists needing specific information from local Nepali archives without needing to read the script.

## 4. Key Features (MVP)

| Feature | Description | Priority |
| --- | --- | --- |
| **Cross-Lingual Search** | Users can query in English; the system retrieves relevant *Nepali* documents using semantic mapping. | **P0 (Critical)** |
| **Bilingual Response** | The bot automatically detects the query language (Eng/Nep) and responds in the same language. | **P0 (Critical)** |
| **Source Citation** | Every answer must cite the specific document/headline used (e.g., *"Source: Kantipur, 2018"*). | **P1 (High)** |
| **"I Don't Know" Guardrail** | If the answer isn't in the dataset, the bot must refuse to answer rather than making things up. | **P1 (High)** |

## 5. Technical Architecture

* **Knowledge Base:** `np20ng` Dataset (10k+ Nepali news articles).
* **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (supports Nepali & English).
* **Vector Database:** **FAISS** (Local, fast, CPU-optimized).
* **LLM (Reasoning):** **Llama-3.1-70B-Versatile** (via Groq API for sub-second latency).
* **Orchestrator:** **LangChain**.
* **Frontend:** **Gradio** or **Streamlit**.

## 6. User Flow

1. **User** opens the web interface.
2. **User** types a question (e.g., *"What is the impact of tourism in Pokhara?"*).
3. **System** detects language (English) and embeds the query.
4. **Retriever** searches the Nepali vector index for semantically similar news.
5. **LLM** reads the retrieved Nepali text and synthesizes an answer *in English*.
6. **UI** displays the Answer + Source Documents.

## 7. Success Metrics

* **Retrieval Accuracy:** The top 3 retrieved documents must contain the answer 80% of the time (verified manually).
* **Latency:** Full response (Retrieval + Generation) should take **< 3 seconds**.
* **Language Consistency:** The bot must never switch languages mid-sentence (e.g., answering an English question in Nepali).

## 8. Out of Scope (For Now)

* Real-time internet browsing (Live news).
* Voice input/output.
* User account/history storage.