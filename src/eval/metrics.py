from __future__ import annotations

import re
from typing import List, Set

from src.extractive.similarity import build_tfidf_vectors


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _unique_tokens(text: str) -> Set[str]:
    return set(_tokenize(text))


def word_count(text: str) -> int:
    return len(_tokenize(text))


def sentence_count(sentences: List[str]) -> int:
    return len([s for s in sentences if (s or "").strip()])


def redundancy_score(sentences: List[str]) -> float:
    """
    Average pairwise cosine similarity between summary sentences (TF-IDF based).
    Higher => more redundancy.
    """
    sentences = [s.strip() for s in sentences if s and s.strip()]
    n = len(sentences)
    if n <= 1:
        return 0.0

    vectors, _ = build_tfidf_vectors(sentences)
    # vectors: list[dict[str, float]]

    def cosine_sparse(v1: dict[str, float], v2: dict[str, float]) -> float:
        if not v1 or not v2:
            return 0.0
        dot = 0.0
        if len(v1) > len(v2):
            v1, v2 = v2, v1
        for k, a in v1.items():
            b = v2.get(k)
            if b is not None:
                dot += a * b
        norm1 = sum(a * a for a in v1.values()) ** 0.5
        norm2 = sum(b * b for b in v2.values()) ** 0.5
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += cosine_sparse(vectors[i], vectors[j])
            pairs += 1

    return total / pairs if pairs else 0.0


def coverage_score(source_text: str, summary_text: str) -> float:
    """
    Token coverage: fraction of unique summary tokens that appear in the source.
    Higher => summary uses more tokens grounded in the source.
    """
    src = _unique_tokens(source_text)
    summ = _unique_tokens(summary_text)
    if not summ:
        return 0.0
    overlap = len(summ.intersection(src))
    return overlap / len(summ)
