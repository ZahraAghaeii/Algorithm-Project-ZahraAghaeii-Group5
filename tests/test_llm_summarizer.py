import os

import pytest

from src.abstractive.llm_summarizer import MetisLLMSummarizer


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = type("msg", (), {"content": content})


class _FakeResp:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


def test_llm_summarizer_returns_empty_for_empty_text(monkeypatch):
    monkeypatch.setenv("METISAI_API_KEY", "dummy")
    s = MetisLLMSummarizer()
    assert s.summarize("") == ""


def test_llm_summarizer_parses_response_content(monkeypatch):
    monkeypatch.setenv("METISAI_API_KEY", "dummy")

    s = MetisLLMSummarizer()

    # Patch the network call so no real API request happens.
    def _fake_create(*args, **kwargs):
        return _FakeResp("This is a summary.")

    monkeypatch.setattr(s.client.chat.completions, "create", _fake_create)

    out = s.summarize("Some long text here", target_sentences=3)
    assert out == "This is a summary."
