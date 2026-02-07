import streamlit as st
from rag_chain import RAGChain

# page config
st.set_page_config(
    page_title="Nepali News RAG Chatbot",
    page_icon="📰",
    layout="centered",
)

st.title("📰 Nepali News RAG Chatbot")
st.caption("Ask questions about Nepali news in English or Nepali.")


# load RAG chain once (cached across reruns)
@st.cache_resource(show_spinner="Loading models … this may take a minute ⏳")
def get_rag_chain() -> RAGChain:
    return RAGChain()


rag = get_rag_chain()


# sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    response_language = st.radio(
        "Response Language",
        options=["Auto-detect", "Nepali", "English"],
        index=0,
        help="Choose the language for the answer. 'Auto-detect' infers from your query.",
    )

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.write(f"{i}. {src}")

# user input
if query := st.chat_input("Type your question here …"):
    # display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # determine language
    lang = None if response_language == "Auto-detect" else response_language

    # generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            answer, sources = rag.get_answer(query, language=lang)

        st.markdown(answer)

        if sources:
            with st.expander("📄 Sources"):
                for i, src in enumerate(sources, 1):
                    st.write(f"{i}. {src}")

    # persist assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
