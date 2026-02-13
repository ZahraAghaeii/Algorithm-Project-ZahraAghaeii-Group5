#  Hybrid Summarization Engine (TextRank + LLM)

## Overview
This project implements a Hybrid Text Summarization System that combines the structural reliability of graph-based algorithms with the semantic intelligence of Large Language Models (LLMs). The engine uses a custom TextRank implementation for extractive importance and GPT-4o-mini for abstractive refinement.

---

## Project Structure
```
.
├── src/
│   ├── extractive/          # Extractive Summarization (Phase 1 core)
│   │   ├── similarity.py    # Cosine similarity & TF-IDF calculations
│   │   └── textrank.py      # TextRank graph iteration & ranking logic
│   ├── abstractive/         # Abstractive Summarization
│   │   └── llm_summarizer.py # MetisAI/GPT API integration logic
│   ├── merge/               # Hybrid Integration
│   │   └── merge_engine.py  # Redundancy filtering & summary blending
│   ├── eval/                # Evaluation Framework
│   │   ├── metrics.py       # Coverage, Redundancy, and Word Count metrics
│   │   └── runner.py        # Logic for running batch evaluations
│   └── utils/               # Helper Modules
│       ├── preprocessing.py # Persian/English text cleaning
│       └── text_splitter.py # Sentence segmentation logic
├── tests/                   # Comprehensive Test Suite
│   ├── test_textrank_sanity.py      # Basic functional tests
│   ├── test_textrank_phase1_cases.py# Validation against Phase 1 requirements
│   ├── test_hard_cases.py           # Edge cases (empty, short, or unrelated text)
│   ├── test_llm_summarizer.py       # API connectivity and response tests
│   └── test_merge_engine.py         # Blending and redundancy removal tests
├── plots/                   # Visual Analysis (Generated)
│   ├── coverage.png         # Comparison of content coverage
│   ├── redundancy.png       # Redundancy levels across methods
│   ├── runtime_sec.png      # Time complexity & scalability chart
│   └── words.png            # Summary length distribution
├── docs/                    # Project Documentation
│   └── report_phase2.pdf    # Detailed Phase 2 technical report
├── app.py                   # Main Streamlit Web Interface
├── run_hybrid_demo.py       # CLI-based demo for quick testing
├── run_evaluation.py        # Main script to run metrics on sample dataset
├── plot_evaluation.py       # Script to generate charts from results
└── requirements.txt         # Project dependencies
```
---
## Features & Modules
1. Extractive Module (TextRank)

* Graph Construction: Sentences are nodes; edges are weighted by TF-IDF Cosine Similarity.
* Ranking: Iterative power method until convergence.
* Tie-breaking: Implements strict rules (Lower Index > Shorter Length) as per project document requirements.

2. Abstractive Module (LLM)

* Uses gpt-4o-mini to rephrase and condense the core ideas of the text.
* Focused on fluency and grammatical coherence.

3. Merge Engine (Hybrid)

* Combines sentences from both methods.
* Redundancy Filter: Uses a similarity threshold (0.75) to ensure no two sentences in the final summary convey the same info.

---

## Installation
Clone the repository and install dependencies:
```
pip install -r requirements.txt
```
---
## testing
We use pytest for quality assurance. The suite covers everything from basic sanity to "hard cases":

pytest tests/   

---

## Team
Group 5 – Algorithm Design Course  
Semester 1404–1405

---
## Contributors
* Zahra Aghaeii (Student ID: 40214923)
* Yeganeh Zafarzadeh (Student ID: 40219303)
