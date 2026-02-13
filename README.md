# Hybrid Text Summarization System (Phase 2)
**Algorithm Design Course Project – Group 5**  
**Path:** Algorithmic Summarization Engine (Classic Algorithm + LLM + Merge)

---

## 1) معرفی موضوع انتخابی
هدف این پروژه ساخت یک سیستم **خلاصه‌سازی متن** است که همزمان از:
- یک روش **کلاسیک و الگوریتمی** (Extractive)  
- و یک روش **مدل زبانی** (Abstractive)  
استفاده کند و در نهایت با یک **الگوریتم ادغام (Merge)**، خروجی نهایی را تولید کند. :contentReference[oaicite:1]{index=1}

---

## 2) تعریف مسئله
ورودی: یک متن (فارسی یا انگلیسی)  
خروجی: یک خلاصه کوتاه‌تر از متن

در فاز ۲، سه خروجی را تولید و مقایسه می‌کنیم:
1) خلاصه‌ی Extractive با **TextRank**  
2) خلاصه‌ی Abstractive با **LLM**  
3) خلاصه‌ی نهایی **Hybrid** با ادغام دو خروجی بالا :contentReference[oaicite:2]{index=2}

---

## 3) توضیح الگوریتم‌ها

### 3.1) TextRank (Extractive)
در TextRank متن به جمله‌ها شکسته می‌شود و یک گراف از جمله‌ها ساخته می‌شود که یال‌ها بر اساس شباهت جمله‌ها وزن‌دار هستند. سپس مشابه PageRank، امتیاز هر جمله به‌صورت تکرارشونده به‌روزرسانی می‌شود و در نهایت Top-k جمله انتخاب می‌شود. (پیاده‌سازی: TF-IDF + Cosine Similarity)

در کد این ریپو:
- بردارسازی TF-IDF و ماتریس شباهت: `src/extractive/similarity.py` :contentReference[oaicite:3]{index=3}  
- اجرای TextRank با پارامترهای damping/max_iter/eps و tie-break: `src/extractive/textrank.py` :contentReference[oaicite:4]{index=4}  

**Tie-break (برای امتیازهای برابر):**
1) جمله‌ای که زودتر در متن آمده باشد  
2) اگر باز هم برابر بود، جمله کوتاه‌تر :contentReference[oaicite:5]{index=5}

---

### 3.2) LLM Summarization (Abstractive)
یک خلاصه‌ی بازنویسی‌شده و روان با استفاده از API سازگار با OpenAI ساخته می‌شود.

در کد این ریپو:
- کلاس `MetisLLMSummarizer` در `src/abstractive/llm_summarizer.py` :contentReference[oaicite:6]{index=6}  
- کلید از متغیر محیطی `METISAI_API_KEY` خوانده می‌شود :contentReference[oaicite:7]{index=7}  
- آدرس پیش‌فرض:
  - `https://api.metisai.ir/openai/v1` (پیش‌فرض)
  - برای محیط‌هایی مثل Colab/Kaggle که ممکن است مشکل داشته باشند: `https://api.tapsage.com/openai/v1` :contentReference[oaicite:8]{index=8}  

---

### 3.3) Merge Engine (Hybrid)
هدف این بخش تولید یک خلاصه‌ی نهایی با **ادغام** خروجی‌های TextRank و LLM است.

استراتژی فعلی (Version 1):
1) ساخت pool کاندیداها (اول جمله‌های Extractive به عنوان anchor، بعد جمله‌های LLM)
2) بردارسازی TF-IDF برای کاندیداها
3) امتیازدهی با heuristic پوشش/مرکزیت (centrality)
4) انتخاب تا k جمله با **حذف تکرار** بر اساس آستانه‌ی شباهت (redundancy threshold)
5) اگر خیلی سخت‌گیرانه شد، یک fallback برای پر شدن خروجی دارد

کد: `src/merge/merge_engine.py` :contentReference[oaicite:9]{index=9}  

---

## 4) نقش LLM در مسیر
LLM در این پروژه نقش **کمکی/Oracle** دارد و جایگزین بخش الگوریتمی نمی‌شود؛ یعنی تصمیم‌گیری اصلی (رتبه‌بندی و انتخاب جمله‌ها) در مسیر الگوریتمی است و LLM صرفاً یک خلاصه‌ی دوم تولید می‌کند تا برای مقایسه/ادغام استفاده شود. :contentReference[oaicite:10]{index=10}

---

## 5) ساختار فولدرها
ساختار کلی branch `phase-2`:

- `app.py` : وب‌دمو با Streamlit (نمایش TextRank / LLM / Hybrid) :contentReference[oaicite:11]{index=11}  
- `run_hybrid_demo.py` : دمو خط فرمان برای اجرای سریع Hybrid :contentReference[oaicite:12]{index=12}  
- `run_evaluation.py` : اجرای ارزیابی و خروجی گرفتن `evaluation_results.csv` :contentReference[oaicite:13]{index=13}  
- `plot_evaluation.py` : رسم نمودارها از روی CSV و ذخیره در `plots/` :contentReference[oaicite:14]{index=14}  
- `src/`
  - `extractive/` : TF-IDF + Similarity + TextRank :contentReference[oaicite:15]{index=15}  
  - `abstractive/` : LLM Summarizer :contentReference[oaicite:16]{index=16}  
  - `merge/` : الگوریتم ادغام و حذف تکرار :contentReference[oaicite:17]{index=17}  
  - `eval/` : متریک‌ها + runner ارزیابی :contentReference[oaicite:18]{index=18}  
  - `utils/` : پیش‌پردازش و sentence splitting (فارسی/انگلیسی) :contentReference[oaicite:19]{index=19}  

---

## 6) نمونه ورودی/خروجی

### نمونه ورودی
متن انگلیسی/فارسی را وارد می‌کنید (مثلاً در Streamlit یا اسکریپت‌ها).

### نمونه خروجی
سیس:contentReference[oaicite:20]{index=20}ank (Extractive)**: چند جمله از خود متن
- **LLM (Abstractive)**: خلاصه بازنویسی‌شده
- **Hybrid**: ترکیب و فیلتر شده‌ی دو خروجی بالا :contentReference[oaicite:21]{index=21}  

---

## 7) نحوه اجرای پروژه

### 7.1) نصب پیش‌نیازها
```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt
