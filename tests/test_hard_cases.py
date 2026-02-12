import pytest
from src.extractive.textrank import textrank_summarize, TextRankConfig

def test_hard_case_long_text_with_low_centrality():
    """
    Worst-Case: متون طولانی با جملات کاملاً بی‌ربط به هم.
    هدف: اطمینان از اینکه الگوریتم در گراف‌های پراکنده (Sparse) دچار خطا نمی‌شود.
    """
    sentences = [
        "The economy is facing global inflation due to supply chain issues.",
        "Biology researchers found a new species of insects in the rainforest.",
        "SpaceX successfully launched a satellite to the Moon's orbit today.",
        "Cooking pasta requires a high temperature and salted boiling water.",
        "The history of ancient Rome is filled with political transitions.",
        "Heavy rainfall caused minor flooding in the downtown area."
    ]
    # رتبه بندی باید بدون خطا انجام شود حتی اگر شباهت‌ها نزدیک به صفر باشد
    summary, scores = textrank_summarize(sentences, k=3)
    assert len(summary) == 3
    assert all(s >= 0 for s in scores)

def test_hard_case_redundant_information_overlap():
    """
    Edge Case: جملات با همپوشانی کلمات بسیار زیاد (Redundancy).
    هدف: بررسی نحوه امتیازدهی وقتی جملات تقریباً مشابه هستند (چالش فاز 1).
    """
    sentences = [
        "Artificial intelligence is a branch of computer science.",
        "AI is a specific branch of computer science focused on learning.",
        "Computer science includes several branches like AI and security.",
        "Machine learning is a subset of artificial intelligence."
    ]
    summary, _ = textrank_summarize(sentences, k=2)
    assert len(summary) == 2

def test_hard_case_persian_complexity():
    """
    Hard Case: متون فارسی با ساختارهای پیچیده و نیم‌فاصله.
    هدف: تست پایداری توکن‌ساز و TF-IDF روی کلمات کلیدی فارسی.
    """
    sentences = [
        "کتاب‌خانه‌های عمومی در حال توسعه‌ی زیرساخت‌های دیجیتال هستند.",
        "توسعه زیرساخت دیجیتال باعث بهبود یادگیری دانش‌آموزان می‌شود.",
        "سیستم‌های هوشمند در کتاب‌خانه به مدیریت منابع کمک می‌کنند.",
        "فناوری‌های نوین یادگیری را برای همه‌ی جوامع تسهیل کرده است."
    ]
    summary, _ = textrank_summarize(sentences, k=2)
    assert any("دیجیتال" in s or "یادگیری" in s for s in summary)

def test_hard_case_tie_break_determinism():
    """
    Requirement Test: اجرای دقیق قانون Tie-break.
    طبق داکیومنت: اگر امتیاز برابر بود -> ایندکس کمتر و سپس طول بیشتر[cite: 714].
    """
    sentences = [
        "Identical score text A.",
        "Identical score text B.",
        "This is a longer sentence but with identical score."
    ]
    # در صورت نبود شباهت، امتیازها برابر (1/n) خواهند بود
    summary, _ = textrank_summarize(sentences, k=1)
    # طبق پیاده‌سازی شما در textrank.py، اولین جمله (Index 0) باید انتخاب شود
    assert summary[0] == sentences[0]

def test_hard_case_max_iter_convergence():
    """
    Worst-Case: بررسی همگرایی در متون سنگین با تکرار بالا.
    """
    sentences = ["Sentence number " + str(i) for i in range(20)]
    config = TextRankConfig(max_iter=5, eps=1e-1) # محدود کردن تکرار برای تست سرعت
    summary, scores = textrank_summarize(sentences, k=5, config=config)
    assert len(summary) == 5
    assert abs(sum(scores) - 1.0) < 1e-5 # مجموع امتیازها باید 1 باشد