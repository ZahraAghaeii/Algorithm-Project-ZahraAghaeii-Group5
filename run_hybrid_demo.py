from __future__ import annotations

from src.abstractive.llm_summarizer import MetisLLMSummarizer
from src.extractive.textrank import textrank_summarize
from src.merge.merge_engine import merge_summaries
from src.utils.text_splitter import split_sentences

print("STARTING DEMO...")
import sys; sys.stdout.flush()


def main() -> None:
    # You can replace this text with any Persian/English input later.
    text = (
        "Artificial intelligence is transforming medicine. "
        "Machine learning models help doctors detect diseases early. "
        "Early diagnosis improves treatment outcomes. "
        "Today it is raining and traffic is heavy."
    )

    k_extractive = 2
    k_final = 3

    # 1) Split text into sentences for TextRank
    sentences = split_sentences(text)

    # 2) Extractive summary (TextRank)
    extractive_summary, scores = textrank_summarize(sentences, k=k_extractive)

    # 3) Abstractive summary (LLM via Metis)
    summarizer = MetisLLMSummarizer()
    abstractive_text = summarizer.summarize(text, target_sentences=3)

    # 4) Merge extractive + abstractive into final summary
    final_summary = merge_summaries(
        extractive_summary=extractive_summary,
        abstractive_text=abstractive_text,
        k=k_final,
    )

    print("\n================ INPUT TEXT ================\n")
    print(text)

    print("\n================ TEXTRANK (EXTRACTIVE) ================\n")
    for i, s in enumerate(extractive_summary, 1):
        print(f"{i}. {s}")

    print("\n================ LLM (ABSTRACTIVE) ================\n")
    print(abstractive_text)

    print("\n================ FINAL (MERGED) ================\n")
    for i, s in enumerate(final_summary, 1):
        print(f"{i}. {s}")


if __name__ == "__main__":
    main()
