import pytest
from src.extractive.textrank import textrank_summarize, TextRankConfig

def test_hard_case_long_text_with_low_centrality():
   
    sentences = [
        "The economy is facing global inflation due to supply chain issues.",
        "Biology researchers found a new species of insects in the rainforest.",
        "SpaceX successfully launched a satellite to the Moon's orbit today.",
        "Cooking pasta requires a high temperature and salted boiling water.",
        "The history of ancient Rome is filled with political transitions.",
        "Heavy rainfall caused minor flooding in the downtown area."
    ]
 
    summary, scores = textrank_summarize(sentences, k=3)
    assert len(summary) == 3
    assert all(s >= 0 for s in scores)

def test_hard_case_redundant_information_overlap():
   
    sentences = [
        "Artificial intelligence is a branch of computer science.",
        "AI is a specific branch of computer science focused on learning.",
        "Computer science includes several branches like AI and security.",
        "Machine learning is a subset of artificial intelligence."
    ]
    summary, _ = textrank_summarize(sentences, k=2)
    assert len(summary) == 2

def test_hard_case_persian_complexity():
   
    sentences = [
        "کتاب‌خانه‌های عمومی در حال توسعه‌ی زیرساخت‌های دیجیتال هستند.",
        "توسعه زیرساخت دیجیتال باعث بهبود یادگیری دانش‌آموزان می‌شود.",
        "سیستم‌های هوشمند در کتاب‌خانه به مدیریت منابع کمک می‌کنند.",
        "فناوری‌های نوین یادگیری را برای همه‌ی جوامع تسهیل کرده است."
    ]
    summary, _ = textrank_summarize(sentences, k=2)
    assert any("دیجیتال" in s or "یادگیری" in s for s in summary)

def test_hard_case_tie_break_determinism():
    
    sentences = [
        "Identical score text A.",
        "Identical score text B.",
        "This is a longer sentence but with identical score."
    ]

    summary, _ = textrank_summarize(sentences, k=1)

    assert summary[0] == sentences[0]

def test_hard_case_max_iter_convergence():
   
    sentences = ["Sentence number " + str(i) for i in range(20)]
    config = TextRankConfig(max_iter=5, eps=1e-1) 
    summary, scores = textrank_summarize(sentences, k=5, config=config)
    assert len(summary) == 5
    assert abs(sum(scores) - 1.0) < 1e-5 