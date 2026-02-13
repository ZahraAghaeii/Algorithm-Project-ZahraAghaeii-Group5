import re
from typing import List


_SENT_SPLIT_RE = re.compile(r"(?:\r?\n)+|[.!؟?؛;]+")


def split_sentences(text: str) -> List[str]:
    
    if not text or not text.strip():
        return []

    parts = _SENT_SPLIT_RE.split(text)
    sentences = [p.strip() for p in parts if p and p.strip()]
    return sentences
