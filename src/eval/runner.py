from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

from src.abstractive.llm_summarizer import MetisLLMSummarizer
from src.eval.metrics import coverage_score, redundancy_score, word_count
from src.extractive.textrank import textrank_summarize
from src.merge.merge_engine import merge_summaries
from src.utils.text_splitter import split_sentences


@dataclass
class EvalResult:
    name: str
    runtime_sec: float
    words: int
    redundancy: float
    coverage: float
    summary: str


def _run_textrank(text: str, k: int) -> str:
    sentences = split_sentences(text)
    summary_sents, _ = textrank_summarize(sentences, k=k)
    return " ".join(summary_sents).strip()


def _run_llm(text: str, target_sentences: int) -> str:
    summarizer = MetisLLMSummarizer()
    return summarizer.summarize(text, target_sentences=target_sentences).strip()


def _run_hybrid(text: str, k_extractive: int, k_final: int, llm_target_sentences: int) -> str:
    sentences = split_sentences(text)
    extractive_sents, _ = textrank_summarize(sentences, k=k_extractive)
    llm_text = _run_llm(text, target_sentences=llm_target_sentences)
    final_sents = merge_summaries(extractive_summary=extractive_sents, abstractive_text=llm_text, k=k_final)
    return " ".join(final_sents).strip()


def evaluate_texts(
    texts: List[str],
    k_extractive: int = 3,
    k_final: int = 4,
    llm_target_sentences: int = 3,
) -> Dict[str, List[EvalResult]]:
    
    out: Dict[str, List[EvalResult]] = {}

    for idx, text in enumerate(texts, 1):
        key = f"text_{idx}"
        out[key] = []

        t0 = time.perf_counter()
        tr_sum = _run_textrank(text, k=k_extractive)
        t1 = time.perf_counter()
        out[key].append(
            EvalResult(
                name="textrank",
                runtime_sec=t1 - t0,
                words=word_count(tr_sum),
                redundancy=redundancy_score(split_sentences(tr_sum)),
                coverage=coverage_score(text, tr_sum),
                summary=tr_sum,
            )
        )

        # LLM
        t0 = time.perf_counter()
        llm_sum = _run_llm(text, target_sentences=llm_target_sentences)
        t1 = time.perf_counter()
        out[key].append(
            EvalResult(
                name="llm",
                runtime_sec=t1 - t0,
                words=word_count(llm_sum),
                redundancy=redundancy_score(split_sentences(llm_sum)),
                coverage=coverage_score(text, llm_sum),
                summary=llm_sum,
            )
        )

        # Hybrid
        t0 = time.perf_counter()
        hyb_sum = _run_hybrid(
            text,
            k_extractive=k_extractive,
            k_final=k_final,
            llm_target_sentences=llm_target_sentences,
        )
        t1 = time.perf_counter()
        out[key].append(
            EvalResult(
                name="hybrid",
                runtime_sec=t1 - t0,
                words=word_count(hyb_sum),
                redundancy=redundancy_score(split_sentences(hyb_sum)),
                coverage=coverage_score(text, hyb_sum),
                summary=hyb_sum,
            )
        )

    return out
