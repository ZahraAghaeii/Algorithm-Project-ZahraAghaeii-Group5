from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.extractive.similarity import build_tfidf_vectors, cosine_similarity_matrix


@dataclass(frozen=True)
class TextRankConfig:
    """
    Configuration for TextRank.

    damping:
        Damping factor (typically 0.85).
    max_iter:
        Maximum number of iterations for convergence.
    eps:
        Convergence threshold based on max absolute score change.
    edge_threshold:
        Optional threshold to drop weak edges (improves speed / reduces noise).
        If None, keep all positive edges.
    """
    damping: float = 0.85
    max_iter: int = 50
    eps: float = 1e-6
    edge_threshold: Optional[float] = None


def _row_outgoing_sum(weights: List[List[float]], j: int) -> float:
    return sum(weights[j])


def _apply_edge_threshold(sim: List[List[float]], threshold: Optional[float]) -> List[List[float]]:
    if threshold is None:
        return sim
    n = len(sim)
    out = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = sim[i][j]
            if w > threshold:
                out[i][j] = w
    return out


def textrank_summarize(
    sentences: List[str],
    k: int,
    config: TextRankConfig = TextRankConfig(),
    extra_stopwords: List[str] | None = None,
) -> Tuple[List[str], List[float]]:
    """
    Run TextRank (graph-based extractive summarization).

    Tie-break policy (as specified in Phase 1):
    1) Prefer earlier sentence index if scores are equal.
    2) If still tied, prefer shorter sentence length.

    Returns:
        summary_sentences: selected sentences (in original order)
        scores: sentence scores aligned with input sentence order
    """
    if k <= 0:
        return [], []

    n = len(sentences)
    if n == 0:
        return [], []

    # If k >= n, return all sentences (no crash / no trimming).
    if k >= n:
        return sentences[:], [1.0 / n for _ in range(n)]

    vectors, _ = build_tfidf_vectors(sentences, extra_stopwords=extra_stopwords)
    sim = cosine_similarity_matrix(vectors)
    weights = _apply_edge_threshold(sim, config.edge_threshold)

    # Initialize scores uniformly.
    scores = [1.0 / n for _ in range(n)]

    # Precompute outgoing weight sums to avoid repeated sums.
    outgoing_sums = [_row_outgoing_sum(weights, j) for j in range(n)]

    for _ in range(config.max_iter):
        new_scores = [1.0 - config.damping for _ in range(n)]
        max_change = 0.0

        for i in range(n):
            acc = 0.0
            # Incoming edges: j -> i
            for j in range(n):
                w_ji = weights[j][i]
                if w_ji <= 0.0:
                    continue

                denom = outgoing_sums[j]
                if denom <= 0.0:
                    continue

                acc += (w_ji / denom) * scores[j]

            new_scores[i] += config.damping * acc
            diff = abs(new_scores[i] - scores[i])
            if diff > max_change:
                max_change = diff

        scores = new_scores
        if max_change < config.eps:
            break

    # Rank indices by score desc, then by earlier index, then by shorter length.
    # (The last two implement the requested tie-break rule deterministically.)
    ranked = sorted(
        range(n),
        key=lambda idx: (-scores[idx], idx, len(sentences[idx])),
    )

    selected = ranked[:k]
    selected_set = set(selected)

    # Return in original order for readability.
    summary = [sentences[i] for i in range(n) if i in selected_set]
    return summary, scores
