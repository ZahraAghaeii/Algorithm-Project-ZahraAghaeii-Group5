from __future__ import annotations

import streamlit as st

from src.extractive.textrank import textrank_summarize
from src.abstractive.llm_summarizer import MetisLLMSummarizer
from src.merge.merge_engine import merge_summaries
from src.utils.text_splitter import split_sentences


st.set_page_config(page_title="Hybrid Summarization Demo", layout="wide")

st.title("Hybrid Text Summarization System")
st.markdown("TextRank + LLM + Merge Engine")


# Sidebar controls
st.sidebar.header("Settings")

k_extractive = st.sidebar.slider("Extractive sentences (TextRank)", 1, 5, 2)
k_final = st.sidebar.slider("Final summary sentences (Hybrid)", 1, 6, 4)
llm_target = st.sidebar.slider("LLM target sentences", 1, 5, 3)

model_type = st.sidebar.selectbox(
    "Choose mode",
    ["Hybrid", "TextRank Only", "LLM Only"]
)


# Input text
text = st.text_area(
    "Enter your text below:",
    height=200,
    placeholder="Paste an English or Persian paragraph here..."
)


if st.button("Generate Summary"):

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Processing..."):

        sentences = split_sentences(text)

        # TextRank
        extractive_summary, _ = textrank_summarize(sentences, k=k_extractive)
        extractive_text = " ".join(extractive_summary)

        # LLM
        summarizer = MetisLLMSummarizer()
        abstractive_text = summarizer.summarize(text, target_sentences=llm_target)

        # Hybrid
        final_summary = merge_summaries(
            extractive_summary=extractive_summary,
            abstractive_text=abstractive_text,
            k=k_final,
        )
        hybrid_text = " ".join(final_summary)

    st.success("Summary generated!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("TextRank")
        st.write(extractive_text)

    with col2:
        st.subheader("LLM")
        st.write(abstractive_text)

    with col3:
        st.subheader("Hybrid")
        st.write(hybrid_text)
