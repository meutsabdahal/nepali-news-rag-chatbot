import streamlit as st


def render_sources(sources: list[str]) -> None:
    if not sources:
        return

    with st.expander("Sources"):
        for i, source in enumerate(sources, 1):
            st.write(f"{i}. {source}")
