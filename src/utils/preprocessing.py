import re
from typing import Iterable, List, Set


_DEFAULT_STOPWORDS_FA: Set[str] = {
    "و", "یا", "به", "از", "در", "را", "که", "این", "آن", "برای", "با", "تا",
    "است", "بود", "باشد", "می", "شود", "شد", "کن", "کند", "کرد", "کرده",
    "هم", "اما", "اگر", "پس", "بر", "چون", "یک", "نه", "من", "تو", "او", "ما", "شما", "آنها",
}

_DEFAULT_STOPWORDS_EN: Set[str] = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "as",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
}


_PUNCT_RE = re.compile(r"[^\w\u0600-\u06FF]+", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    
    text = text.strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    
    text = normalize_text(text)
    if not text:
        return []
    return text.split()


def build_stopwords(extra_stopwords: Iterable[str] | None = None) -> Set[str]:
    
    sw = set()
    sw.update(_DEFAULT_STOPWORDS_FA)
    sw.update(_DEFAULT_STOPWORDS_EN)
    if extra_stopwords:
        sw.update({s.strip().lower() for s in extra_stopwords if s and s.strip()})
    return sw


def filter_tokens(tokens: List[str], stopwords: Set[str] | None = None) -> List[str]:
    
    if stopwords is None:
        stopwords = build_stopwords()
    return [t for t in tokens if t not in stopwords and len(t) > 1]
