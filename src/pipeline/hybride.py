from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from src.abstractive.llm_summarizer import MetisLLMSummarizer
from src.extractive.textrank import textrank_summarize
from src.merge.merge_engine import MergeConfig, merge_summaries
from src.utils.text_splitter import split_sentences


class Summarizer(Protocol):
    """
    Minimal interface for an abstractive summarizer.
    This makes the pipeline testable without real API calls.
    """

    def summarize(self, text: str, target_sentences: int = 5) -> str:
        ...


@dataclass(frozen=True)
class HybridConfig:
    """
    Configuration for the hybrid summarization pipeline.

    k_final:
        Number of sentences in the final merged summary.

    k_extractive:
        Number of sentences to take from TextRank (extractive summary).

    llm_target_sentences:
        Approximate number of sentences requested from the LLM.

    merge_config:
        Configuration for redundancy filtering and selection behavior.
    """
    k_final: int = 5
    k_extractive: int = 5
    llm_target_sentences: int = 5
    merge_config: MergeConfig = MergeConfig()


def hybrid_summarize(
    text: str,
    config: HybridConfig = HybridConfig(),
    summarizer: Optional[Summarizer] = None,
) -> Tuple[List[str], List[str], str]:
    """
    Run the full hybrid pipeline:

      1) Split input text into sentences
      2) Extractive summary via TextRank -> list[str]
      3) Abstractive summary via LLM -> str
      4) Merge both -> final list[str]

    Returns:
        final_summary_sentences, extractive_summary_sentences, abstractive_summary_text
    """
    text = (text or "").strip()
    if not text:
        return [], [], ""

    # 1) Sentence split for extractive stage
    sentences = split_sentences(text)
    sentences = [s.strip() for s in sentences if s and s.strip()]
    if not sentences:
        return [], [], ""

    # 2) Extractive: TextRank
    extractive_summary, _ = textrank_summarize(sentences, k=config.k_extractive)

    # 3) Abstractive: LLM (real or injected)
    if summarizer is None:
        summarizer = MetisLLMSummarizer()

    abstractive_text = summarizer.summarize(text, target_sentences=config.llm_target_sentences)

    # 4) Merge
    final_summary = merge_summaries(
        extractive_summary=extractive_summary,
        abstractive_text=abstractive_text,
        k=config.k_final,
        config=config.merge_config,
    )

    return final_summary, extractive_summary, abstractive_text
