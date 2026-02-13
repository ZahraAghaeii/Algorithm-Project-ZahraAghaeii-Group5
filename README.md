# Algorithmic Summarization Engine (Phase 1)

##  Project Overview
This project is the Phase 1 implementation of the final project for the **Algorithm Design** course.

The goal is to design and analyze a **Hybrid Text Summarization Engine** that combines:

- A **classic extractive summarization algorithm** (TextRank)
- A **Large Language Model (LLM)** for abstractive summarization
- A future merging strategy (Phase 2)

 Track: **Algorithmic Summarization Engine**  
 Phase: **Phase 1 — Problem Analysis & Algorithm Design**

---

##  Team Members
- Zahra Aghaeii — 40214923  
- Yeganeh Zafarzadeh — 40219303  

---

##  Problem Definition
Given an input text:

\[
T = \{s_1, s_2, ..., s_n\}
\]

The system generates a shorter summary **S** that:

- Covers the main concepts  
- Avoids redundancy  
- Maintains coherence and readability  

Outputs:

1. **Extractive Summary** (TextRank)
2. **Abstractive Summary** (LLM)
3. Final merged summary (Phase 2)

---

##  Algorithms Considered

### 1. Frequency-Based Summarization
Baseline approach based on keyword frequency.

### 2. Sentence Ranking (Feature-Based)
Sentences are scored using handcrafted rules such as:

- Length  
- Position  
- Keyword presence  

### 3. TextRank (Main Algorithm)
TextRank is a graph-based ranking algorithm inspired by PageRank:

- Sentences → graph nodes  
- Similarity → weighted edges  
- Iterative ranking assigns importance  

Top-k ranked sentences form the extractive summary.

---

##  Complexity Analysis

- Similarity Graph Construction: **O(n²)**
- Ranking Iterations: **O(iter · n²)**
- Sorting Sentences: **O(n log n)**
- Space Complexity: **O(n²)**

---

##  Role of LLM
LLM is used only as an oracle component:

- Produces a fluent abstractive summary  
- Does NOT affect similarity computation or ranking  
- Will be merged with TextRank output in Phase 2  

---

##  Manual Test Cases (Phase 1)

Includes:

- Simple case with irrelevant sentence removal  
- Medium case with clustered topics  
- High redundancy case  
- Zero-similarity edge case  
- Boundary case where k > n  
