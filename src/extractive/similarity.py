import math
from collections import Counter
from typing import Dict, List, Set, Tuple

from src.utils.preprocessing import build_stopwords, filter_tokens, tokenize


def _tf(tokens: List[str]) -> Counter:
    return Counter(tokens)


def _idf(docs: List[List[str]]) -> Dict[str, float]:
    """
    Compute IDF with smoothing:
      idf(t) = log((1 + N) / (1 + df(t))) + 1
    """
    n_docs = len(docs)
    df: Counter = Counter()
    for doc in docs:
        df.update(set(doc))

    idf: Dict[str, float] = {}
    for term, dfi in df.items():
        idf[term] = math.log((1.0 + n_docs) / (1.0 + dfi)) + 1.0
    return idf


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = _tf(tokens)
    vec: Dict[str, float] = {}
    for term, freq in tf.items():
        if term in idf:
            vec[term] = float(freq) * idf[term]
    return vec


def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0

    dot = 0.0
    # Iterate over smaller dict for efficiency
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    for k, a in v1.items():
        b = v2.get(k)
        if b is not None:
            dot += a * b

    norm1 = math.sqrt(sum(a * a for a in v1.values()))
    norm2 = math.sqrt(sum(b * b for b in v2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def build_tfidf_vectors(
    sentences: List[str],
    extra_stopwords: List[str] | None = None,
) -> Tuple[List[Dict[str, float]], Set[str]]:
    """
    Convert sentences into TF-IDF sparse vectors.

    Returns:
        vectors: list of {term: weight} dictionaries
        stopwords: the stopword set used (useful for reproducibility)
    """
    stopwords = build_stopwords(extra_stopwords)

    docs_tokens: List[List[str]] = []
    for s in sentences:
        tokens = tokenize(s)
        tokens = filter_tokens(tokens, stopwords=stopwords)
        docs_tokens.append(tokens)

    idf = _idf(docs_tokens)
    vectors = [_tfidf_vector(toks, idf) for toks in docs_tokens]
    return vectors, stopwords


def cosine_similarity_matrix(vectors: List[Dict[str, float]]) -> List[List[float]]:
    """
    Build an NxN cosine similarity matrix.
    Diagonal is 0.0 (we do not connect a sentence to itself).
    """
    n = len(vectors)
    sim = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sim[i][j] = _cosine(vectors[i], vectors[j])
    return sim
