from src.extractive.textrank import textrank_summarize


def test_empty_input_returns_empty():
    summary, scores = textrank_summarize([], k=2)
    assert summary == []
    assert scores == []


def test_k_zero_returns_empty():
    sentences = ["Sentence one", "Sentence two"]
    summary, scores = textrank_summarize(sentences, k=0)
    assert summary == []
    assert scores == []


def test_k_greater_or_equal_than_n_returns_all():
    sentences = ["S1", "S2"]
    summary, scores = textrank_summarize(sentences, k=5)
    assert summary == sentences
    assert len(scores) == 2
