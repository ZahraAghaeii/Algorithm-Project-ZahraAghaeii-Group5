from src.extractive.textrank import textrank_summarize


def test_phase1_case_1_outlier_sentence_should_not_be_selected():
    """
    Phase 1 - Test 1:
    The outlier sentence (about rain/traffic) should not appear in the summary.
    We assert properties rather than exact sentence set to avoid brittle tests.
    """
    sentences = [
        "هوش مصنوعی در پزشکی برای تشخیص بیماری استفاده می‌شود",
        "مدل‌های یادگیری ماشین می‌توانند الگوهای داده‌های پزشکی را پیدا کنند",
        "تشخیص زودهنگام بیماری باعث افزایش شانس درمان می‌شود",
        "امروز هوا باران است و ترافیک زیاد شده است",
    ]

    summary, _ = textrank_summarize(sentences, k=2)

    # Must return exactly k sentences
    assert len(summary) == 2

    # The outlier should not be selected
    assert "امروز هوا باران است و ترافیک زیاد شده است" not in summary

    # At least one medical-related sentence should be present
    medical_markers = ("پزشکی", "بیماری", "تشخیص", "درمان")
    assert any(any(m in s for m in medical_markers) for s in summary)


def test_phase1_case_2_outlier_sentence_should_not_be_selected():
    """
    Phase 1 - Test 2:
    The weather sentence should not appear in the summary.
    """
    sentences = [
        "دانشگاه‌ها برای ثبت‌نام، مدارک هویتی و ریزنمرات را بررسی می‌کنند",
        "بررسی مدارک باعث جلوگیری از ثبت‌نام غیرمجاز می‌شود",
        "فرآیند ثبت‌نام شامل پرداخت شهریه و انتخاب واحد است",
        "انتخاب واحد باید بر اساس پیش‌نیازها انجام شود",
        "هوای شهر امروز خیلی سرد است",
    ]

    summary, _ = textrank_summarize(sentences, k=2)

    assert len(summary) == 2
    assert "هوای شهر امروز خیلی سرد است" not in summary

    # Should contain at least one registration-related sentence
    markers = ("ثبت", "مدارک", "شهریه", "انتخاب", "واحد")
    assert any(any(m in s for m in markers) for s in summary)


def test_phase1_edge_case_near_zero_similarity_should_be_deterministic():
    """
    Phase 1 - Edge case:
    When similarities are (nearly) zero, scores become close and tie-break rules matter.
    We cannot guarantee exact internal similarity values, but we can enforce determinism:
    calling the function twice must return the same output.
    """
    sentences = [
        "گربه روی دیوار پرید",
        "اقتصاد کشور رشد کرد",
        "برنامه‌نویسی تمرین می‌خواهد",
    ]

    summary1, _ = textrank_summarize(sentences, k=1)
    summary2, _ = textrank_summarize(sentences, k=1)

    assert summary1 == summary2
    assert len(summary1) == 1


def test_phase1_k_greater_than_n_edge_case():
    """
    Phase 1 - Edge case:
    If k > n, return all sentences without crashing.
    """
    sentences = [
        "این متن فقط دو جمله دارد",
        "پس خلاصه باید همان دو جمله باشد",
    ]

    summary, _ = textrank_summarize(sentences, k=5)
    assert summary == sentences
