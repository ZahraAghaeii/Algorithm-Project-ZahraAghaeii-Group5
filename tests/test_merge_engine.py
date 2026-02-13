from src.merge.merge_engine import MergeConfig, merge_summaries


def test_merge_falls_back_to_extractive_when_no_abstractive_text():
    extractive = ["A", "B", "C"]
    out = merge_summaries(extractive, abstractive_text=None, k=2)
    assert out == ["A", "B"]


def test_merge_respects_k_sentences():
    extractive = ["Sentence A", "Sentence B"]
    abstractive = "Sentence C. Sentence D. Sentence E."
    out = merge_summaries(extractive, abstractive_text=abstractive, k=3)
    assert len(out) == 3


def test_merge_keeps_extractive_anchors_first_by_default():
    extractive = ["Core fact 1", "Core fact 2"]
    abstractive = "Core fact 1. Nice rewrite sentence."
    out = merge_summaries(extractive, abstractive_text=abstractive, k=2)
    # Must contain the first extractive anchor
    assert "Core fact 1" in out


def test_merge_deduplicates_exact_duplicates():
    extractive = ["X", "Y"]
    abstractive = "X. X. Y. Z."
    out = merge_summaries(extractive, abstractive_text=abstractive, k=3)
    # Should not contain exact duplicates
    assert len(out) == len(set(out))
