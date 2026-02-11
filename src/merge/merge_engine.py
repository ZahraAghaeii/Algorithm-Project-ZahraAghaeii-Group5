from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.extractive.similarity import build_tfidf_vectors
from src.utils.text_splitter import split_sentences


@dataclass(frozen=True)
class MergeConfig:
    """
    Configuration for the hybrid merge engine.

    redundancy_threshold:
        If a candidate sentence is too similar to any already-selected sentence
        (cosine similarity >= threshold), it will be skipped.

    prefer_extractive:
        If True, try to keep extractive sentences (TextRank output) as anchors first.

    max_abstractive_sentences:
        Limit how many sentences from the abstractive text we consider (for efficiency).
    """
    redundancy_threshold: float = 0.75
    prefer_extractive: bool = True
    max_abstractive_sentences: int = 60


def _cosine_sparse(v1: dict[str, float], v2: dict[str, float]) -> float:
    """
    Compute cosine similarity between two sparse TF-IDF vectors.
    Vectors are stored as dicts: {term: weight}.
    """
    if not v1 or not v2:
        return 0.0

    # Dot product over intersection
    dot = 0.0
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    for term, w1 in v1.items():
        w2 = v2.get(term)
        if w2 is not None:
            dot += w1 * w2

    # Norms
    norm1 = sum(w * w for w in v1.values()) ** 0.5
    norm2 = sum(w * w for w in v2.values()) ** 0.5
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot / (norm1 * norm2)


def _dedupe_keep_order(sentences: List[str]) -> List[str]:
    """
    Remove exact duplicate sentences while preserving order.
    """
    seen = set()
    out: List[str] = []
    for s in sentences:
        s = (s or "").strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _build_candidate_pool(
    extractive_summary: List[str],
    abstractive_text: Optional[str],
    config: MergeConfig,
) -> Tuple[List[str], List[bool]]:
    """
    Build candidates from:
      - extractive_summary (list of sentences)
      - abstractive_text (string -> split into sentences)

    Returns:
        candidates: list of unique sentences (order preserved)
        is_extractive: parallel list indicating source of each candidate
    """
    ext = _dedupe_keep_order(extractive_summary)

    abs_sents: List[str] = []
    if abstractive_text and abstractive_text.strip():
        abs_sents = split_sentences(abstractive_text)
        abs_sents = _dedupe_keep_order(abs_sents)
        abs_sents = abs_sents[: config.max_abstractive_sentences]

    candidates: List[str] = []
    is_extractive: List[bool] = []

    # 1) Put extractive anchors first
    for s in ext:
        candidates.append(s)
        is_extractive.append(True)

    # 2) Add abstractive sentences after (avoid duplicates)
    existing = set(candidates)
    for s in abs_sents:
        if s in existing:
            continue
        candidates.append(s)
        is_extractive.append(False)
        existing.add(s)

    return candidates, is_extractive


def _centrality_scores(vectors: List[dict[str, float]]) -> List[float]:
    """
    Lightweight coverage heuristic:
    score[i] = sum_j cosine(i, j) for j != i

    Higher score => more semantically connected to others => good for coverage.
    """
    n = len(vectors)
    scores = [0.0] * n
    for i in range(n):
        total = 0.0
        for j in range(n):
            if i == j:
                continue
            total += _cosine_sparse(vectors[i], vectors[j])
        scores[i] = total
    return scores


def merge_summaries(
    extractive_summary: List[str],
    abstractive_text: Optional[str],
    k: int,
    config: MergeConfig = MergeConfig(),
    extra_stopwords: Optional[List[str]] = None,
) -> List[str]:
    """
    Merge extractive and abstractive summaries into a final k-sentence summary.

    Version 1 strategy:
      1) Build candidate pool: extractive anchors + abstractive sentences.
      2) Vectorize candidates using TF-IDF.
      3) Rank candidates by a centrality (coverage) heuristic.
      4) Select up to k sentences with redundancy filtering:
         - Optionally select extractive anchors first
         - Fill remaining slots by centrality score
         - If still short, relax redundancy constraint as a fallback

    Notes:
      - This module is independent of the actual LLM.
      - You can pass a mock abstractive_text during development/testing.
    """
    if k <= 0:
        return []

    # If no abstractive text exists, fall back to extractive
    if not abstractive_text or not abstractive_text.strip():
        return _dedupe_keep_order(extractive_summary)[:k]

    candidates, is_extractive = _build_candidate_pool(extractive_summary, abstractive_text, config)
    if not candidates:
        return []

    if len(candidates) <= k:
        return candidates

    vectors, _ = build_tfidf_vectors(candidates, extra_stopwords=extra_stopwords)
    centrality = _centrality_scores(vectors)

    selected: List[int] = []

    def is_redundant(i: int) -> bool:
        for j in selected:
            if _cosine_sparse(vectors[i], vectors[j]) >= config.redundancy_threshold:
                return True
        return False

    # Step A: pick extractive anchors first (optional)
    if config.prefer_extractive:
        for i, flag in enumerate(is_extractive):
            if not flag:
                continue
            if len(selected) >= k:
                break
            if is_redundant(i):
                continue
            selected.append(i)

    # Step B: fill remaining slots by centrality rank
    ranked = sorted(range(len(candidates)), key=lambda i: (-centrality[i], i))
    for i in ranked:
        if len(selected) >= k:
            break
        if i in selected:
            continue
        if is_redundant(i):
            continue
        selected.append(i)

    # Step C: fallback (if redundancy was too strict)
    if len(selected) < k:
        for i in ranked:
            if len(selected) >= k:
                break
            if i in selected:
                continue
            selected.append(i)

    return [candidates[i] for i in selected[:k]]
