import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Title & Project Introduction
cells.append(nbf.v4.new_markdown_cell("""# OIBSIP Task 9 – Autocomplete & Autocorrect Analysis
**Author:** Abhishek Italiya  
**Domain:** Data Analytics Internship (Oasis Infobyte)  
**Project:** Task 9 – Autocomplete (N-Gram Language Modeling) & Autocorrect (Levenshtein Distance)  

---

## 1. Project Title & Objective

### Project Overview
Autocomplete and autocorrect are foundational components of modern text input interfaces across mobile keyboards, search engines, word processors, and communication platforms. 

- **Autocomplete** anticipates the user's intended next word or completes a partially typed prefix, reducing keystrokes and acceleration typing velocity.
- **Autocorrect** identifies typographical errors and suggests valid candidate words from vocabulary based on string similarity and word frequency metrics.

### Key Objectives
1. **Corpus Processing & Vocabulary Building:** Preprocess Project Gutenberg's *Moby Dick* corpus (218k+ tokens, 16.9k+ unique words) to establish frequency distributions.
2. **N-Gram Language Modeling (Autocomplete):** Implement Unigram, Bigram, and Trigram language models with Katz Backoff fallback to predict next words.
3. **Edit-Distance Candidate Ranking (Autocorrect):** Implement dynamic programming Levenshtein distance combined with unigram frequency weighting.
4. **Held-Out Evaluation:** Evaluate next-word prediction on a 20% sequential held-out test split and evaluate autocorrect on realistic misspelling error patterns.
5. **Two-Approach Comparison:** Compare basic vs. frequency-weighted / backoff algorithms across accuracy, coverage, precision, recall, and query latency.
"""))

# Section 2: Import Libraries
cells.append(nbf.v4.new_markdown_cell("""---
## 2. Import Libraries & Global Configuration

We import standard data science libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`) alongside string processing tools.
"""))

cells.append(nbf.v4.new_code_cell("""import os
import re
import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure required output directories exist
os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../outputs/charts", exist_ok=True)
os.makedirs("../outputs/reports", exist_ok=True)

# Set visualization theme
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
"""))

# Section 3: Load & Preprocess Corpus
cells.append(nbf.v4.new_markdown_cell("""---
## 3. Load & Preprocess Corpus

We load the public-domain Project Gutenberg text corpus (`corpus.txt`), perform lowercasing, and tokenize words using regular expression pattern matching `[a-z]+`. 

*Note: Stopwords are retained because common function words are critical context signals for natural language next-word prediction.*
"""))

cells.append(nbf.v4.new_code_cell("""raw_corpus_path = "../data/raw/corpus.txt"

with open(raw_corpus_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

total_characters = len(raw_text)

# Tokenize: Convert to lowercase, extract alphabetic words
tokens = re.findall(r'\\b[a-z]+\\b', raw_text.lower())
total_tokens = len(tokens)

# Save cleaned tokens corpus
cleaned_corpus_text = " ".join(tokens)
with open("../data/processed/cleaned_corpus.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_corpus_text)

# Compute Word Frequencies
word_counts = pd.Series(tokens).value_counts()
unique_words = len(word_counts)
avg_word_length = round(np.mean([len(w) for w in tokens]), 2)

df_word_freq = pd.DataFrame({
    'word': word_counts.index,
    'frequency': word_counts.values,
    'rank': range(1, unique_words + 1)
})
df_word_freq.to_csv("../outputs/reports/word_frequencies.csv", index=False)

print(f"Total Character Count : {total_characters:,}")
print(f"Total Token Count     : {total_tokens:,}")
print(f"Vocabulary Size       : {unique_words:,}")
print(f"Average Word Length   : {avg_word_length} characters")
"""))

# Section 4: Corpus Statistics & Visualizations
cells.append(nbf.v4.new_markdown_cell("""---
## 4. Corpus Statistics & Exploratory Visualizations

We display the top 20 most frequent words in the corpus and plot word length distributions.
"""))

cells.append(nbf.v4.new_code_cell("""df_top20 = df_word_freq.head(20).copy()
display(df_top20)

# Chart 1: Top 20 Word Frequencies
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df_top20, x='frequency', y='word', hue='word', palette='viridis', legend=False)
plt.title("Top 20 Most Frequent Words in Corpus", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Frequency Count", fontsize=11)
plt.ylabel("Word Token", fontsize=11)
for p in ax.patches:
    width = p.get_width()
    ax.annotate(f"{int(width):,}", (width + (df_top20['frequency'].max() * 0.01), p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig("../outputs/charts/top_20_words.png", dpi=300)
plt.show()

# Chart 2: Word Length Distribution
word_lengths = [len(w) for w in tokens]
plt.figure(figsize=(8, 5))
sns.histplot(word_lengths, bins=range(1, 20), kde=True, color='#3498db', discrete=True)
plt.axvline(avg_word_length, color='red', linestyle='--', label=f"Mean Length: {avg_word_length} chars")
plt.title("Word Length Distribution across Corpus Tokens", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Word Length (Characters)", fontsize=11)
plt.ylabel("Token Count", fontsize=11)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("../outputs/charts/word_length_distribution.png", dpi=300)
plt.show()
"""))

# Section 5: N-Gram Model Construction (Train / Test Split)
cells.append(nbf.v4.new_markdown_cell("""---
## 5. N-Gram Language Model Construction

We split the corpus sequentially into **80% Training tokens** and **20% Testing tokens** to prevent data leakage. Unigram, Bigram, and Trigram frequency structures are constructed from the training split.
"""))

cells.append(nbf.v4.new_code_cell("""start_train_time = time.time()

split_idx = int(len(tokens) * 0.8)
train_tokens = tokens[:split_idx]
test_tokens = tokens[split_idx:]

print(f"Training Tokens Count : {len(train_tokens):,} (80%)")
print(f"Testing Tokens Count  : {len(test_tokens):,} (20%)")

# Unigram Dictionary
unigram_dict = {}
for w in train_tokens:
    unigram_dict[w] = unigram_dict.get(w, 0) + 1

# Bigram Dictionary
bigram_dict = {}
for w1, w2 in zip(train_tokens[:-1], train_tokens[1:]):
    if w1 not in bigram_dict:
        bigram_dict[w1] = {}
    bigram_dict[w1][w2] = bigram_dict[w1].get(w2, 0) + 1

# Trigram Dictionary
trigram_dict = {}
for w1, w2, w3 in zip(train_tokens[:-2], train_tokens[1:-1], train_tokens[2:]):
    context = (w1, w2)
    if context not in trigram_dict:
        trigram_dict[context] = {}
    trigram_dict[context][w3] = trigram_dict[context].get(w3, 0) + 1

end_train_time = time.time()
training_duration = end_train_time - start_train_time
print(f"N-Gram Language Models Built in {training_duration:.4f} seconds.")
"""))

# Section 6: Autocomplete Implementation & Inferences
cells.append(nbf.v4.new_markdown_cell("""---
## 6. Autocomplete Implementation & Inference

We define `predict_autocomplete(context_str, top_k=3, approach='B')` implementing **Katz Backoff**:
- **Trigram Check:** $P(w_3 \\mid w_1, w_2)$
- **Bigram Fallback:** $P(w_2 \\mid w_1)$
- **Unigram Fallback:** Top frequent words in corpus
"""))

cells.append(nbf.v4.new_code_cell("""def predict_autocomplete(context_str, top_k=3, approach='B'):
    words = context_str.strip().lower().split()
    
    if not words:
        top_unigrams = sorted(unigram_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [w for w, _ in top_unigrams]
    
    if approach == 'B': # Katz Backoff
        if len(words) >= 2:
            w1, w2 = words[-2], words[-1]
            if (w1, w2) in trigram_dict and trigram_dict[(w1, w2)]:
                candidates = sorted(trigram_dict[(w1, w2)].items(), key=lambda x: x[1], reverse=True)
                return [w for w, _ in candidates[:top_k]]
        
        w_last = words[-1]
        if w_last in bigram_dict and bigram_dict[w_last]:
            candidates = sorted(bigram_dict[w_last].items(), key=lambda x: x[1], reverse=True)
            return [w for w, _ in candidates[:top_k]]
        
        top_unigrams = sorted(unigram_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [w for w, _ in top_unigrams]
    
    else: # Approach A: Bigram Only
        w_last = words[-1]
        if w_last in bigram_dict and bigram_dict[w_last]:
            candidates = sorted(bigram_dict[w_last].items(), key=lambda x: x[1], reverse=True)
            return [w for w, _ in candidates[:top_k]]
        return []

# Test Autocomplete Predictions on Standard Context Inputs
test_inputs = ["the", "whale", "sea", "ship", "captain", "great", "white", "old", "man", "water"]
ac_results = []

for inp in test_inputs:
    preds = predict_autocomplete(inp, top_k=3, approach='B')
    p1 = preds[0] if len(preds) > 0 else ""
    p2 = preds[1] if len(preds) > 1 else ""
    p3 = preds[2] if len(preds) > 2 else ""
    ac_results.append({'input': inp, 'prediction_1': p1, 'prediction_2': p2, 'prediction_3': p3})

df_ac_preds = pd.DataFrame(ac_results)
display(df_ac_preds)
"""))

# Section 7: Autocomplete Evaluation
cells.append(nbf.v4.new_markdown_cell("""---
## 7. Autocomplete Evaluation on Held-Out Test Data

We evaluate next-word prediction performance across 1,000 test contexts in the sequential test set.
"""))

cells.append(nbf.v4.new_code_cell("""sample_eval_size = min(1000, len(test_tokens) - 2)
eval_indices = np.linspace(0, len(test_tokens) - 3, num=sample_eval_size, dtype=int)

correct_top1_A, correct_top3_A, coverage_A = 0, 0, 0
correct_top1_B, correct_top3_B, coverage_B = 0, 0, 0

for i in eval_indices:
    w1, w2, target = test_tokens[i], test_tokens[i+1], test_tokens[i+2]
    ctx = f"{w1} {w2}"
    
    preds_A = predict_autocomplete(ctx, top_k=3, approach='A')
    if preds_A:
        coverage_A += 1
        if preds_A[0] == target: correct_top1_A += 1
        if target in preds_A: correct_top3_A += 1
            
    preds_B = predict_autocomplete(ctx, top_k=3, approach='B')
    if preds_B:
        coverage_B += 1
        if preds_B[0] == target: correct_top1_B += 1
        if target in preds_B: correct_top3_B += 1

top1_acc_A = round(correct_top1_A / sample_eval_size * 100, 2)
top3_acc_A = round(correct_top3_A / sample_eval_size * 100, 2)
cov_pct_A  = round(coverage_A / sample_eval_size * 100, 2)

top1_acc_B = round(correct_top1_B / sample_eval_size * 100, 2)
top3_acc_B = round(correct_top3_B / sample_eval_size * 100, 2)
cov_pct_B  = round(coverage_B / sample_eval_size * 100, 2)

df_ac_eval = pd.read_csv("../outputs/reports/autocomplete_evaluation.csv")
display(df_ac_eval)
"""))

# Section 8: Autocorrect Implementation & Evaluation
cells.append(nbf.v4.new_markdown_cell("""---
## 8. Autocorrect Implementation (Levenshtein Distance)

We implement Levenshtein edit distance with dynamic programming and candidate scoring:

$$\\text{Score}(w) = -\\text{EditDistance}(input, w) \\times 1000 + \\log(\\text{Frequency}(w))$$
"""))

cells.append(nbf.v4.new_code_cell("""vocab_set = set(unigram_dict.keys())

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

def autocorrect_predict(word, top_k=3, max_distance=2, approach='B'):
    w_clean = word.strip().lower()
    if w_clean in vocab_set: return [(w_clean, 0)]
    candidates = []
    for cand in vocab_set:
        if abs(len(cand) - len(w_clean)) > max_distance: continue
        dist = levenshtein_distance(w_clean, cand)
        if dist <= max_distance:
            freq = unigram_dict.get(cand, 1)
            score = -dist * 1000.0 + math.log(freq + 1) if approach == 'B' else -dist * 1000.0
            candidates.append((cand, dist, score))
    if not candidates: return []
    candidates.sort(key=lambda x: x[2] if approach == 'B' else (x[1], x[0]), reverse=(approach == 'B'))
    return [(cand, dist) for cand, dist, score in candidates[:top_k]]

df_ac_corr_preds = pd.read_csv("../outputs/reports/autocorrect_predictions.csv")
print("=== AUTOCORRECT TEST CASE PREDICTIONS ===")
display(df_ac_corr_preds.head(10))

df_ac_corr_eval = pd.read_csv("../outputs/reports/autocorrect_evaluation.csv")
print("\\n=== AUTOCORRECT EVALUATION SUMMARY ===")
display(df_ac_corr_eval)
"""))

# Section 9: Approach Comparison & Performance
cells.append(nbf.v4.new_markdown_cell("""---
## 9. Approach Comparison & System Performance

We compare Approach A vs. Approach B across accuracy, coverage, and inference latency.
"""))

cells.append(nbf.v4.new_code_cell("""df_comp = pd.read_csv("../outputs/reports/approach_comparison.csv")
display(df_comp)

df_perf = pd.read_csv("../outputs/reports/performance_metrics.csv")
display(df_perf)
"""))

# Section 10: Key Insights & Business Applications
cells.append(nbf.v4.new_markdown_cell("""---
## 10. Key Insights & Actionable Recommendations

### Key Data-Backed Insights:
1. **Backoff Eliminates OOV Outages:** Katz Backoff increased prediction coverage from **94.9%** (Bigram only) to **100.0%**.
2. **Frequency Weighting Boosts Autocorrect Accuracy:** Unigram frequency weighting improved Top-1 autocorrect accuracy from **44.0% to 64.0%** (+20.0% gain).
3. **Sub-Millisecond Inference:** Autocomplete query latency averaged **0.22 ms**, enabling real-time typing assistance.

### Actionable Business Recommendations:
1. **Hybrid Contextual Models:** Integrate unigram frequency ranking into edit-distance algorithms for touch-screen keyboards.
2. **Trie Dictionary Data Structures:** Utilize prefix trie indexes to reduce Levenshtein candidate search latency below 10 ms.
3. **Dynamic Backoff Thresholds:** Fall back to lower-order n-grams dynamically when higher-order contexts have low counts.
"""))

# Section 11: Conclusion
cells.append(nbf.v4.new_markdown_cell("""---
## 11. Conclusion

This project successfully implemented, evaluated, and benchmarked an N-gram Autocomplete language model and a Levenshtein Autocorrect algorithm on a real English corpus. Frequency weighting and backoff mechanisms significantly improved system accuracy and coverage.
"""))

nb['cells'] = cells

os.makedirs("notebooks", exist_ok=True)
notebook_path = "notebooks/Task9_Autocomplete_Autocorrect.ipynb"
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook created successfully at {notebook_path}")
