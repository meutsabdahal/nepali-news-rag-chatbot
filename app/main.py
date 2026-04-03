import streamlit as st

from app.components import render_sources
from app.example_questions import EXAMPLE_QUESTIONS
from nepali_news_rag.pipeline import NepaliNewsPipeline


st.set_page_config(page_title="Nepali News RAG", page_icon="📰", layout="centered")
st.title("Nepali News RAG")
st.caption("Ask Nepali news questions in English or Nepali.")


@st.cache_resource(show_spinner="Loading models and index...")
def get_pipeline() -> NepaliNewsPipeline:
    return NepaliNewsPipeline()


pipeline = get_pipeline()

with st.sidebar:
    st.subheader("Settings")
    response_language = st.radio(
        "Response Language", ["Auto-detect", "Nepali", "English"], index=0
    )
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
