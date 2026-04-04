import streamlit as st
from pathlib import Path

from app.components import render_sources
from app.example_questions import EXAMPLE_QUESTIONS
from nepali_news_rag.config import get_settings
from nepali_news_rag.pipeline import NepaliNewsPipeline


st.set_page_config(page_title="Nepali News RAG", page_icon="📰", layout="centered")
st.title("Nepali News RAG")
st.caption("Ask Nepali news questions in English or Nepali.")


def _pipeline_cache_key() -> tuple:
    settings = get_settings()

    index_sig = "missing"
    index_dir = Path(settings.vector_store_dir)
    if index_dir.exists():
        parts: list[str] = []
        for artifact in sorted(index_dir.glob("*")):
            try:
                stat = artifact.stat()
                parts.append(f"{artifact.name}:{stat.st_mtime_ns}:{stat.st_size}")
            except FileNotFoundError:
                continue
        if parts:
            index_sig = "|".join(parts)

    return (
        settings.llm_provider,
        settings.ollama_host,
        settings.ollama_model,
        settings.groq_model,
        settings.hf_model,
        settings.retriever_k,
        index_sig,
    )


@st.cache_resource(show_spinner="Loading models and index...")
def get_pipeline(_cache_key: tuple) -> NepaliNewsPipeline:
    return NepaliNewsPipeline()


pipeline = get_pipeline(_pipeline_cache_key())

with st.sidebar:
    st.subheader("Settings")
    response_language = st.radio(
        "Response Language", ["Auto-detect", "Nepali", "English"], index=0
    )
    if st.button("Reload Model + Index"):
        get_pipeline.clear()
        st.rerun()
    st.subheader("Try these")
    for q in EXAMPLE_QUESTIONS:
        st.caption(f"- {q}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

if query := st.chat_input("Type your question"):
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    language = None if response_language == "Auto-detect" else response_language

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = pipeline.run(query, language=language)

        st.markdown(result["answer"])
        render_sources(result.get("sources", []))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        }
    )
