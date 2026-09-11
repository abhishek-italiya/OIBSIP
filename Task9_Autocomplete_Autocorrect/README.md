# OIBSIP Task 9 – Autocomplete & Autocorrect

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green.svg)](https://www.nltk.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) project implementing **N-Gram Autocomplete Language Modeling** and **Levenshtein Distance Autocorrect** developed for the **Oasis InfoByte Data Analytics Internship Program (OIBSIP)** – Task 9 (Level 2 Task 5).

---

## 1. Project Overview
Text input assistance systems—such as predictive autocomplete and automatic spell-checking—are critical features of modern digital interfaces (mobile touch-screen keyboards, search engine boxes, IDE code editors, and messaging applications). 

This project demonstrates:
1. **Autocomplete:** N-gram language modeling ($1$-gram, $2$-gram, $3$-gram) with **Katz Backoff** to predict next words and complete prefixes.
2. **Autocorrect:** Dynamic programming **Levenshtein Edit Distance** combined with **Unigram Frequency Weighting** to correct misspelled words.
3. **Empirical Evaluation:** Comparative benchmarking of basic vs. frequency-weighted algorithms across accuracy, coverage, precision, recall, F1-score, and inference latency.

---

## 2. Project Objective
- **Corpus Processing & Vocabulary Building:** Tokenize and analyze a public-domain text corpus (*Moby Dick* via NLTK Gutenberg, containing 218,361 tokens and 16,948 unique vocabulary words).
- **N-Gram Language Modeling:** Build Unigram, Bigram, and Trigram frequency probability structures on an 80% training split (174,688 tokens).
- **Katz Backoff Implementation:** Implement a fallback hierarchy ($\text{Trigram} \to \text{Bigram} \to \text{Unigram}$) to achieve 100% prediction coverage.
- **Edit Distance Ranking:** Combine Levenshtein edit distance ($\le 2$) with log unigram frequency score:
  $$\text{Score}(w) = -\text{EditDistance}(\text{input}, w) \times 1000 + \log(\text{Frequency}(w))$$
- **Held-Out Evaluation & Benchmarking:** Evaluate next-word prediction on a 20% sequential held-out test split (43,673 tokens) and evaluate autocorrect on 25 realistic typographical error cases.

---

## 3. Corpus Description & Statistics
- **Source:** Project Gutenberg (*Moby Dick* by Herman Melville via NLTK Corpus API).
- **Total Character Count:** 1,242,990 characters.
- **Total Token Count:** 218,361 words.
- **Vocabulary Size (Unique Words):** 16,948 words.
- **Average Word Length:** 4.36 characters.

### Corpus Summary Table:
| Metric | Value |
| :--- | :---: |
| **Total Characters** | 1,242,990 |
| **Total Tokens** | 218,361 |
| **Unique Words (Vocabulary)** | 16,948 |
| **Average Word Length** | 4.36 chars |
| **Training Tokens (80%)** | 174,688 |
| **Testing Tokens (20%)** | 43,673 |

---

## 4. Technologies Used
- **Programming Language:** Python 3.10+
- **NLP & Text Tokenization:** `nltk`, `re`
- **Data Processing:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Environment:** Jupyter Notebook, VS Code

---

## 5. Project Structure

```text
Task9_Autocomplete_Autocorrect/
│
├── data/
│   ├── raw/
│   │   └── corpus.txt                         # Raw Project Gutenberg text corpus
│   └── processed/
│       ├── cleaned_corpus.txt                 # Lowercased tokenized cleaned text
│       ├── word_frequencies.csv               # Vocabulary word frequency counts
│       └── evaluation_dataset.csv             # Held-out sequential test pairs
│
├── notebooks/
│   └── Task9_Autocomplete_Autocorrect.ipynb   # Executed Jupyter Notebook
│
├── outputs/
│   ├── charts/                                # 6 High-resolution generated charts
│   │   ├── autocomplete_accuracy.png
│   │   ├── autocorrect_accuracy.png
│   │   ├── autocorrect_confusion_matrix.png
│   │   ├── ngram_coverage_performance.png
│   │   ├── top_20_words.png
│   │   └── word_length_distribution.png
│   └── reports/                               # 9 Generated CSV evaluation reports
│       ├── approach_comparison.csv
│       ├── autocomplete_evaluation.csv
│       ├── autocomplete_predictions.csv
│       ├── autocorrect_evaluation.csv
│       ├── autocorrect_predictions.csv
│       ├── corpus_statistics.csv
│       ├── key_insights.csv
│       ├── performance_metrics.csv
│       ├── top_20_words.csv
│       └── word_frequencies.csv
│
├── build_task9_notebook.py                    # Programmatic notebook builder
├── download_corpus_data.py                    # Automated corpus downloader script
├── run_autocomplete_autocorrect_pipeline.py    # End-to-end Python pipeline
├── .gitignore                                 # Git ignore rules
├── LICENSE                                    # MIT License
├── README.md                                  # Project documentation
└── requirements.txt                           # Dependency specifications
```

---

## 6. Autocomplete Evaluation (N-Gram Language Model)

Evaluated next-word prediction on 1,000 held-out test contexts from the 20% sequential test set:

| Approach | Top-1 Accuracy (%) | Top-3 Accuracy (%) | Coverage (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Approach A (Bigram Only)** | **11.90%** | **19.00%** | 94.90% | 11.90% | 12.54% | 12.21% |
| **Approach B (Trigram Katz Backoff)** | 10.80% | 18.00% | **100.00%** | 10.80% | 10.80% | 10.80% |

*Key Observation:* Katz Backoff (Approach B) eliminated Out-Of-Vocabulary (OOV) prediction outages, increasing prediction coverage from 94.90% to 100.00%.

---

## 7. Autocorrect Evaluation (Levenshtein Distance)

Evaluated spell-checking on 25 realistic typographical error test cases:

| Approach | Top-1 Accuracy (%) | Top-3 Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Approach A (Basic Edit Distance)** | 44.00% | 68.00% | 44.00% | 44.00% | 44.00% |
| **Approach B (Frequency-Weighted Edit Distance)** | **64.00%** | **84.00%** | **64.00%** | **64.00%** | **64.00%** |

*Key Observation:* Unigram frequency weighting (Approach B) resolves edit distance ties in favor of common English words, improving Top-1 accuracy from 44.00% to 64.00% (+20.00% absolute gain) and Top-3 accuracy to 84.00%.

---

## 8. System Performance & Inference Latency

Measured execution times on local hardware:

| Metric | Value |
| :--- | :---: |
| **Model Training Time** | 1.1088 seconds |
| **Avg Autocomplete Latency** | 0.223 ms / query |
| **Avg Autocorrect Latency** | 455.941 ms / query |
| **Evaluated Test Cases** | 25 words |

---

## 9. Visualizations Generated

All generated plots are saved in high resolution under `outputs/charts/`:

1. **Top 20 Word Frequencies (`top_20_words.png`):** Horizontal bar chart of the most frequent words.
2. **Word Length Distribution (`word_length_distribution.png`):** Histogram showing word character length distribution.
3. **N-Gram Coverage & Performance (`ngram_coverage_performance.png`):** Bar plot comparing Approach A vs. B across accuracy and coverage.
4. **Autocomplete Accuracy (`autocomplete_accuracy.png`):** Bar plot of Top-3 autocomplete accuracy.
5. **Autocorrect Accuracy (`autocorrect_accuracy.png`):** Bar plot comparing Basic vs Frequency-Weighted autocorrect accuracy.
6. **Autocorrect Confusion Matrix (`autocorrect_confusion_matrix.png`):** Heatmap of ground-truth vs predicted correction pairs.

---

## 10. Key Insights
1. **Backoff Guarantees Prediction Coverage:** Katz Backoff guarantees 100.0% coverage by falling back to unigram statistics when higher-order N-grams are absent.
2. **Frequency Weighting Resolves Edit Distance Ambiguity:** Incorporating log unigram frequency into candidate scoring improved Top-1 correction accuracy by +20.0% (from 44.0% to 64.0%).
3. **Sub-Millisecond Autocomplete Latency:** N-gram lookup latency averages 0.223 ms per query, enabling seamless real-time touch-screen keyboard autocomplete.
4. **Short Word Dominance:** Words of length 3–5 characters constitute over 60% of corpus tokens (mean word length: 4.36 characters).
5. **Top-3 Autocorrect Precision:** Frequency-weighted autocorrect achieved an 84.00% Top-3 suggestion accuracy on realistic typing misspellings.

---

## 11. Practical Applications & Business Recommendations
1. **Mobile Keyboard Autocomplete:** Deploy frequency-weighted unigram/bigram backoff models on mobile touch keyboards to reduce user keystrokes.
2. **Trie Index Acceleration:** Utilize prefix trie data structures for candidate retrieval to reduce Levenshtein search latency under 10 ms.
3. **Dynamic Contextual Thresholds:** Fall back dynamically to lower-order n-grams when higher-order contexts have low counts.

---

## 12. How to Run the Project

### Setup Steps
1. Navigate to project directory:
   ```bash
   cd Task9_Autocomplete_Autocorrect
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the automated dataset downloader:
   ```bash
   python download_corpus_data.py
   ```
4. Run the complete analysis pipeline:
   ```bash
   python run_autocomplete_autocorrect_pipeline.py
   ```
5. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/Task9_Autocomplete_Autocorrect.ipynb
   ```

---

## 13. Requirements & Licensing
- Dependencies specified in `requirements.txt`.
- Code and documentation released under the [MIT License](LICENSE).
- Corpus text attribution belongs to Project Gutenberg / NLTK.

---

## 14. Conclusion
This project demonstrates the core mechanics of predictive text autocomplete and spelling correction. By combining N-gram language modeling with Katz backoff and frequency-weighted Levenshtein edit distance, we achieved high coverage and accuracy with sub-millisecond autocomplete query performance.
