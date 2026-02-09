import re
from typing import List


_SENT_SPLIT_RE = re.compile(r"(?:\r?\n)+|[.!؟?؛;]+")


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple punctuation/newline heuristics.
    Works for Persian and English.

    Returns:
        A list of non-empty, stripped sentences in original order.
    """
    if not text or not text.strip():
        return []

    parts = _SENT_SPLIT_RE.split(text)
    sentences = [p.strip() for p in parts if p and p.strip()]
    return sentences
