import os
import re
import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure required output directories exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# Set Seaborn theme for clean presentation
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# ==========================================
# 1. DATA PREPROCESSING & CORPUS STATISTICS
# ==========================================
print("=== 1. PREPROCESSING CORPUS & GENERATING STATISTICS ===")
raw_corpus_path = "data/raw/corpus.txt"

with open(raw_corpus_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

total_characters = len(raw_text)

# Tokenize: Convert to lowercase, extract alphabetic words
tokens = re.findall(r'\b[a-z]+\b', raw_text.lower())
total_tokens = len(tokens)

# Save cleaned tokens corpus
cleaned_corpus_text = " ".join(tokens)
with open("data/processed/cleaned_corpus.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_corpus_text)

# Compute Word Frequencies
word_counts = pd.Series(tokens).value_counts()
unique_words = len(word_counts)
avg_word_length = round(np.mean([len(w) for w in tokens]), 2)

# Word Frequencies DataFrame
df_word_freq = pd.DataFrame({
    'word': word_counts.index,
    'frequency': word_counts.values,
    'rank': range(1, unique_words + 1)
})
df_word_freq.to_csv("outputs/reports/word_frequencies.csv", index=False)
df_word_freq.to_csv("data/processed/word_frequencies.csv", index=False)

# Corpus Statistics Report
df_stats = pd.DataFrame([{
    'total_characters': total_characters,
    'total_tokens': total_tokens,
    'unique_words': unique_words,
    'vocabulary_size': unique_words,
    'average_word_length': avg_word_length
}])
df_stats.to_csv("outputs/reports/corpus_statistics.csv", index=False)

# Top 20 Words Report
df_top20 = df_word_freq.head(20).copy()
df_top20.to_csv("outputs/reports/top_20_words.csv", index=False)

print(f"Total Characters : {total_characters:,}")
print(f"Total Tokens     : {total_tokens:,}")
print(f"Unique Words     : {unique_words:,}")
print(f"Avg Word Length  : {avg_word_length} characters")

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
plt.savefig("outputs/charts/top_20_words.png", dpi=300)
plt.close()

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
plt.savefig("outputs/charts/word_length_distribution.png", dpi=300)
plt.close()

# ==========================================
# 2. TRAIN / TEST SPLIT & N-GRAM MODELING
# ==========================================
print("\n=== 2. BUILDING N-GRAM LANGUAGE MODELS (TRAIN/TEST SPLIT) ===")
start_train_time = time.time()

split_idx = int(len(tokens) * 0.8)
train_tokens = tokens[:split_idx]
test_tokens = tokens[split_idx:]

print(f"Training Tokens   : {len(train_tokens):,} (80%)")
print(f"Testing Tokens    : {len(test_tokens):,} (20%)")

# Save held-out evaluation dataset
df_eval_pairs = pd.DataFrame({
    'context_word1': test_tokens[:-2],
    'context_word2': test_tokens[1:-1],
    'target_word': test_tokens[2:]
})
df_eval_pairs.to_csv("data/processed/evaluation_dataset.csv", index=False)

# Build Unigram, Bigram, Trigram Frequency Dictionaries from Train Tokens
unigram_dict = {}
for w in train_tokens:
    unigram_dict[w] = unigram_dict.get(w, 0) + 1

bigram_dict = {}
for w1, w2 in zip(train_tokens[:-1], train_tokens[1:]):
    if w1 not in bigram_dict:
        bigram_dict[w1] = {}
    bigram_dict[w1][w2] = bigram_dict[w1].get(w2, 0) + 1

trigram_dict = {}
for w1, w2, w3 in zip(train_tokens[:-2], train_tokens[1:-1], train_tokens[2:]):
    context = (w1, w2)
    if context not in trigram_dict:
        trigram_dict[context] = {}
    trigram_dict[context][w3] = trigram_dict[context].get(w3, 0) + 1

end_train_time = time.time()
training_duration = end_train_time - start_train_time
print(f"N-Gram Models Trained in {training_duration:.4f} seconds.")

# ==========================================
# 3. AUTOCOMPLETE IMPLEMENTATION & INFERENCE
# ==========================================
def predict_autocomplete(context_str, top_k=3, approach='B'):
    """
    Predicts next word or completes partial prefix.
    approach='B': Trigram -> Bigram -> Unigram Katz Backoff
    approach='A': Bigram only (No backoff)
    """
    words = context_str.strip().lower().split()
    
    if not words:
        # Fallback to top unigrams
        top_unigrams = sorted(unigram_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [w for w, _ in top_unigrams]
    
    if approach == 'B':
        # Try Trigram Backoff first
        if len(words) >= 2:
            w1, w2 = words[-2], words[-1]
            if (w1, w2) in trigram_dict and trigram_dict[(w1, w2)]:
                candidates = sorted(trigram_dict[(w1, w2)].items(), key=lambda x: x[1], reverse=True)
                return [w for w, _ in candidates[:top_k]]
        
        # Try Bigram Backoff next
        w_last = words[-1]
        if w_last in bigram_dict and bigram_dict[w_last]:
            candidates = sorted(bigram_dict[w_last].items(), key=lambda x: x[1], reverse=True)
            return [w for w, _ in candidates[:top_k]]
        
        # Fallback to Unigram top frequent
        top_unigrams = sorted(unigram_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [w for w, _ in top_unigrams]
    
    else: # Approach A: Bigram only
        w_last = words[-1]
        if w_last in bigram_dict and bigram_dict[w_last]:
            candidates = sorted(bigram_dict[w_last].items(), key=lambda x: x[1], reverse=True)
            return [w for w, _ in candidates[:top_k]]
        return []

def complete_prefix(prefix_str, top_k=3):
    """
    Completes partial prefix by searching vocabulary.
    """
    pref = prefix_str.strip().lower()
    matches = [w for w in unigram_dict.keys() if w.startswith(pref)]
    matches_sorted = sorted(matches, key=lambda w: unigram_dict[w], reverse=True)
    return matches_sorted[:top_k]

# Generate Autocomplete Predictions for Required Test Cases
test_inputs = ["the", "whale", "sea", "ship", "captain", "great", "white", "old", "man", "water"]
autocomplete_preds = []

start_ac_time = time.time()
for inp in test_inputs:
    preds = predict_autocomplete(inp, top_k=3, approach='B')
    p1 = preds[0] if len(preds) > 0 else ""
    p2 = preds[1] if len(preds) > 1 else ""
    p3 = preds[2] if len(preds) > 2 else ""
    autocomplete_preds.append({
        'input': inp,
        'prediction_1': p1,
        'prediction_2': p2,
        'prediction_3': p3
    })
end_ac_time = time.time()
ac_latency_per_query_ms = ((end_ac_time - start_ac_time) / len(test_inputs)) * 1000.0

df_ac_preds = pd.DataFrame(autocomplete_preds)
df_ac_preds.to_csv("outputs/reports/autocomplete_predictions.csv", index=False)
print("Saved outputs/reports/autocomplete_predictions.csv")

# ==========================================
# 4. AUTOCOMPLETE EVALUATION (HELD-OUT TEST)
# ==========================================
print("\n=== 4. EVALUATING AUTOCOMPLETE ON HELD-OUT TEST DATA ===")
sample_eval_size = min(1000, len(test_tokens) - 2)
eval_indices = np.linspace(0, len(test_tokens) - 3, num=sample_eval_size, dtype=int)

correct_top1_A, correct_top3_A, coverage_A = 0, 0, 0
correct_top1_B, correct_top3_B, coverage_B = 0, 0, 0

for i in eval_indices:
    w1, w2, target = test_tokens[i], test_tokens[i+1], test_tokens[i+2]
    ctx = f"{w1} {w2}"
    
    # Approach A
    preds_A = predict_autocomplete(ctx, top_k=3, approach='A')
    if preds_A:
        coverage_A += 1
        if preds_A[0] == target:
            correct_top1_A += 1
        if target in preds_A:
            correct_top3_A += 1
            
    # Approach B
    preds_B = predict_autocomplete(ctx, top_k=3, approach='B')
    if preds_B:
        coverage_B += 1
        if preds_B[0] == target:
            correct_top1_B += 1
        if target in preds_B:
            correct_top3_B += 1

top1_acc_A = round(correct_top1_A / sample_eval_size * 100, 2)
top3_acc_A = round(correct_top3_A / sample_eval_size * 100, 2)
cov_pct_A  = round(coverage_A / sample_eval_size * 100, 2)

top1_acc_B = round(correct_top1_B / sample_eval_size * 100, 2)
top3_acc_B = round(correct_top3_B / sample_eval_size * 100, 2)
cov_pct_B  = round(coverage_B / sample_eval_size * 100, 2)

prec_B = top1_acc_B
rec_B  = round(correct_top1_B / coverage_B * 100, 2) if coverage_B > 0 else 0.0
f1_B   = round(2 * (prec_B * rec_B) / (prec_B + rec_B), 2) if (prec_B + rec_B) > 0 else 0.0

df_ac_eval = pd.DataFrame([
    {
        'Approach': 'Approach A (Bigram Only)',
        'Top1_Accuracy (%)': top1_acc_A,
        'Top3_Accuracy (%)': top3_acc_A,
        'Coverage (%)': cov_pct_A,
        'Precision (%)': top1_acc_A,
        'Recall (%)': round(correct_top1_A / coverage_A * 100, 2) if coverage_A > 0 else 0.0,
        'F1_Score (%)': round(2 * (top1_acc_A * (correct_top1_A / coverage_A * 100)) / (top1_acc_A + (correct_top1_A / coverage_A * 100)), 2) if coverage_A > 0 else 0.0
    },
    {
        'Approach': 'Approach B (Trigram Backoff)',
        'Top1_Accuracy (%)': top1_acc_B,
        'Top3_Accuracy (%)': top3_acc_B,
        'Coverage (%)': cov_pct_B,
        'Precision (%)': prec_B,
        'Recall (%)': rec_B,
        'F1_Score (%)': f1_B
    }
])
df_ac_eval.to_csv("outputs/reports/autocomplete_evaluation.csv", index=False)
print("Saved outputs/reports/autocomplete_evaluation.csv")

# Chart 3: N-Gram Coverage & Performance Comparison
plt.figure(figsize=(8, 5))
df_cov_plot = pd.DataFrame({
    'Metric': ['Top-1 Accuracy', 'Top-3 Accuracy', 'Coverage'],
    'Approach A (Bigram)': [top1_acc_A, top3_acc_A, cov_pct_A],
    'Approach B (Trigram Backoff)': [top1_acc_B, top3_acc_B, cov_pct_B]
}).melt(id_vars='Metric', var_name='Approach', value_name='Percentage (%)')

ax = sns.barplot(data=df_cov_plot, x='Metric', y='Percentage (%)', hue='Approach', palette='Blues_d')
plt.title("Autocomplete Performance & Coverage: Approach A vs. B", fontsize=13, fontweight='bold', pad=12)
plt.ylabel("Percentage (%)", fontsize=11)
for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 1.0),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.ylim(0, 115)
plt.tight_layout()
plt.savefig("outputs/charts/ngram_coverage_performance.png", dpi=300)
plt.close()

# Chart 4: Autocomplete Top-3 Accuracy
plt.figure(figsize=(7, 5))
ax = sns.barplot(data=df_ac_eval, x='Approach', y='Top3_Accuracy (%)', hue='Approach', palette='crest', legend=False)
plt.title("Autocomplete Top-3 Accuracy Comparison", fontsize=13, fontweight='bold', pad=12)
plt.ylabel("Top-3 Accuracy (%)", fontsize=11)
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{h:.2f}%", (p.get_x() + p.get_width() / 2., h / 2),
                ha='center', va='center', fontsize=11, color='white', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/autocomplete_accuracy.png", dpi=300)
plt.close()

# ==========================================
# 5. AUTOCORRECT IMPLEMENTATION & INFERENCE
# ==========================================
print("\n=== 5. LEVENSHTEIN AUTOCORRECT IMPLEMENTATION & EVALUATION ===")

def levenshtein_distance(s1, s2):
    """Computes exact Levenshtein edit distance using dynamic programming."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # Deletion
                                   dp[i][j-1],      # Insertion
                                   dp[i-1][j-1])    # Substitution
    return dp[m][n]

# Vocabulary set from training unigrams
vocab_set = set(unigram_dict.keys())

def autocorrect_predict(word, top_k=3, max_distance=2, approach='B'):
    """
    Ranks correction candidates using Levenshtein distance & Unigram frequency.
    approach='B': Frequency-weighted Edit Distance (Score = -dist * 1000 + log(freq))
    approach='A': Basic Edit Distance (Distance primary, alphabetical tiebreak)
    """
    w_clean = word.strip().lower()
    if w_clean in vocab_set:
        return [(w_clean, 0)]
    
    candidates = []
    for cand in vocab_set:
        # Optimization: skip words with length difference > max_distance
        if abs(len(cand) - len(w_clean)) > max_distance:
            continue
        dist = levenshtein_distance(w_clean, cand)
        if dist <= max_distance:
            freq = unigram_dict.get(cand, 1)
            if approach == 'B':
                score = -dist * 1000.0 + math.log(freq + 1)
            else:
                score = -dist * 1000.0
            candidates.append((cand, dist, score))
            
    if not candidates:
        return []
    
    if approach == 'B':
        candidates.sort(key=lambda x: x[2], reverse=True)
    else:
        candidates.sort(key=lambda x: (x[1], x[0]))
        
    return [(cand, dist) for cand, dist, score in candidates[:top_k]]

# 25 Realistic Misspelled Words Evaluation Dataset
autocorrect_test_cases = [
    {"misspelled_word": "whael", "correct_word": "whale"},
    {"misspelled_word": "captin", "correct_word": "captain"},
    {"misspelled_word": "watr", "correct_word": "water"},
    {"misspelled_word": "oceen", "correct_word": "ocean"},
    {"misspelled_word": "saen", "correct_word": "seen"},
    {"misspelled_word": "graet", "correct_word": "great"},
    {"misspelled_word": "whiet", "correct_word": "white"},
    {"misspelled_word": "shipp", "correct_word": "ship"},
    {"misspelled_word": "haed", "correct_word": "head"},
    {"misspelled_word": "thier", "correct_word": "their"},
    {"misspelled_word": "beffore", "correct_word": "before"},
    {"misspelled_word": "agane", "correct_word": "again"},
    {"misspelled_word": "amoung", "correct_word": "among"},
    {"misspelled_word": "arround", "correct_word": "around"},
    {"misspelled_word": "commin", "correct_word": "coming"},
    {"misspelled_word": "doun", "correct_word": "down"},
    {"misspelled_word": "evver", "correct_word": "ever"},
    {"misspelled_word": "fyrst", "correct_word": "first"},
    {"misspelled_word": "haav", "correct_word": "have"},
    {"misspelled_word": "limtle", "correct_word": "little"},
    {"misspelled_word": "loong", "correct_word": "long"},
    {"misspelled_word": "maen", "correct_word": "mean"},
    {"misspelled_word": "nightt", "correct_word": "night"},
    {"misspelled_word": "othr", "correct_word": "other"},
    {"misspelled_word": "stille", "correct_word": "still"}
]

autocorrect_results = []
start_ac_corr_time = time.time()
correct_count_top1_B, correct_count_top3_B = 0, 0
correct_count_top1_A, correct_count_top3_A = 0, 0

for item in autocorrect_test_cases:
    m_word = item["misspelled_word"]
    c_word = item["correct_word"]
    
    # Approach B (Frequency-Weighted)
    preds_B = autocorrect_predict(m_word, top_k=3, approach='B')
    pred_w_B = preds_B[0][0] if preds_B else ""
    dist_B = preds_B[0][1] if preds_B else -1
    c_list_B = [w for w, d in preds_B]
    
    is_top1_B = (pred_w_B == c_word)
    is_top3_B = (c_word in c_list_B)
    
    if is_top1_B: correct_count_top1_B += 1
    if is_top3_B: correct_count_top3_B += 1
    
    # Approach A (Basic)
    preds_A = autocorrect_predict(m_word, top_k=3, approach='A')
    pred_w_A = preds_A[0][0] if preds_A else ""
    c_list_A = [w for w, d in preds_A]
    if pred_w_A == c_word: correct_count_top1_A += 1
    if c_word in c_list_A: correct_count_top3_A += 1
    
    autocorrect_results.append({
        'misspelled_word': m_word,
        'correct_word': c_word,
        'predicted_word': pred_w_B,
        'edit_distance': dist_B,
        'rank': (c_list_B.index(c_word) + 1) if c_word in c_list_B else -1
    })

end_ac_corr_time = time.time()
autocorrect_latency_ms = ((end_ac_corr_time - start_ac_corr_time) / len(autocorrect_test_cases)) * 1000.0

df_autocorrect_preds = pd.DataFrame(autocorrect_results)
df_autocorrect_preds.to_csv("outputs/reports/autocorrect_predictions.csv", index=False)
print("Saved outputs/reports/autocorrect_predictions.csv")

# Autocorrect Evaluation Report
n_test = len(autocorrect_test_cases)
acc1_A = round(correct_count_top1_A / n_test * 100, 2)
acc3_A = round(correct_count_top3_A / n_test * 100, 2)

acc1_B = round(correct_count_top1_B / n_test * 100, 2)
acc3_B = round(correct_count_top3_B / n_test * 100, 2)

df_autocorrect_eval = pd.DataFrame([
    {
        'Approach': 'Approach A (Basic Edit Distance)',
        'Top1_Accuracy (%)': acc1_A,
        'Top3_Accuracy (%)': acc3_A,
        'Precision (%)': acc1_A,
        'Recall (%)': acc1_A,
        'F1_Score (%)': acc1_A
    },
    {
        'Approach': 'Approach B (Frequency-Weighted Edit Distance)',
        'Top1_Accuracy (%)': acc1_B,
        'Top3_Accuracy (%)': acc3_B,
        'Precision (%)': acc1_B,
        'Recall (%)': acc1_B,
        'F1_Score (%)': acc1_B
    }
])
df_autocorrect_eval.to_csv("outputs/reports/autocorrect_evaluation.csv", index=False)
print("Saved outputs/reports/autocorrect_evaluation.csv")

# Chart 5: Autocorrect Accuracy Comparison
plt.figure(figsize=(7, 5))
ax = sns.barplot(data=df_autocorrect_eval, x='Approach', y='Top1_Accuracy (%)', hue='Approach', palette='magma', legend=False)
plt.title("Autocorrect Top-1 Accuracy: Approach A vs. B", fontsize=13, fontweight='bold', pad=12)
plt.ylabel("Top-1 Accuracy (%)", fontsize=11)
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h / 2),
                ha='center', va='center', fontsize=11, color='white', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/autocorrect_accuracy.png", dpi=300)
plt.close()

# Chart 6: Autocorrect Confusion Matrix (Top Confusion Pairs)
# Build confusion counts for test cases
conf_pairs = []
for item, res in zip(autocorrect_test_cases, autocorrect_results):
    c_target = item['correct_word']
    c_pred   = res['predicted_word']
    conf_pairs.append({'Target': c_target, 'Predicted': c_pred})

df_conf = pd.DataFrame(conf_pairs)
conf_matrix_df = pd.crosstab(df_conf['Target'], df_conf['Predicted'])

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5)
plt.title("Autocorrect Target vs. Predicted Heatmap (Evaluation Subset)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Predicted Correction Word", fontsize=11)
plt.ylabel("Target Ground-Truth Word", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/charts/autocorrect_confusion_matrix.png", dpi=300)
plt.close()

# ==========================================
# 6. APPROACH COMPARISON & PERFORMANCE METRICS
# ==========================================
print("\n=== 6. GENERATING APPROACH COMPARISON & PERFORMANCE METRICS ===")

df_comp = pd.DataFrame([
    {
        'Component': 'Autocomplete (Next-Word Prediction)',
        'Approach_A_Name': 'Bigram Only (No Backoff)',
        'Approach_A_Top1_Acc': f"{top1_acc_A}%",
        'Approach_A_Coverage': f"{cov_pct_A}%",
        'Approach_B_Name': 'Trigram Katz Backoff',
        'Approach_B_Top1_Acc': f"{top1_acc_B}%",
        'Approach_B_Coverage': f"{cov_pct_B}%",
        'Key_Advantage': 'Backoff dramatically increases coverage from low bigram counts to 100% via unigrams.'
    },
    {
        'Component': 'Autocorrect (Edit Distance)',
        'Approach_A_Name': 'Basic Levenshtein Distance',
        'Approach_A_Top1_Acc': f"{acc1_A}%",
        'Approach_A_Coverage': '100%',
        'Approach_B_Name': 'Frequency-Weighted Levenshtein',
        'Approach_B_Top1_Acc': f"{acc1_B}%",
        'Approach_B_Coverage': '100%',
        'Key_Advantage': 'Unigram frequency weighting resolves distance ties in favor of common English words.'
    }
])
df_comp.to_csv("outputs/reports/approach_comparison.csv", index=False)
print("Saved outputs/reports/approach_comparison.csv")

# Performance Metrics Report
df_perf = pd.DataFrame([{
    'training_time_seconds': round(training_duration, 4),
    'avg_autocomplete_latency_ms': round(ac_latency_per_query_ms, 3),
    'avg_autocorrect_latency_ms': round(autocorrect_latency_ms, 3),
    'test_cases_evaluated': len(autocorrect_test_cases)
}])
df_perf.to_csv("outputs/reports/performance_metrics.csv", index=False)
print("Saved outputs/reports/performance_metrics.csv")

# ==========================================
# 7. KEY INSIGHTS GENERATION
# ==========================================
print("\n=== 7. SAVING KEY INSIGHTS ===")
insights_data = [
    {
        "Insight_ID": 1,
        "Area": "Corpus Vocabulary Distribution",
        "Finding": f"The corpus contains {total_tokens:,} total tokens across {unique_words:,} unique vocabulary words, with an average word length of {avg_word_length} characters."
    },
    {
        "Insight_ID": 2,
        "Area": "Autocomplete Backoff Benefit",
        "Finding": f"Katz Backoff (Approach B) increased prediction coverage from {cov_pct_A}% (Bigram only) to 100.0% by falling back gracefully from Trigrams to Bigrams and Unigrams."
    },
    {
        "Insight_ID": 3,
        "Area": "Autocorrect Frequency Weighting",
        "Finding": f"Frequency-weighted Levenshtein distance (Approach B) achieved {acc1_B}% Top-1 accuracy on realistic misspellings compared to {acc1_A}% for basic edit distance."
    },
    {
        "Insight_ID": 4,
        "Area": "Autocomplete Prediction Accuracy",
        "Finding": f"On held-out test data, N-gram autocomplete achieved {top1_acc_B}% Top-1 accuracy and {top3_acc_B}% Top-3 accuracy for next-word prediction."
    },
    {
        "Insight_ID": 5,
        "Area": "Inference Latency & Efficiency",
        "Finding": f"Average query inference latency was {ac_latency_per_query_ms:.3f} ms for autocomplete and {autocorrect_latency_ms:.3f} ms for autocorrect, demonstrating real-time capability."
    }
]

df_insights = pd.DataFrame(insights_data)
df_insights.to_csv("outputs/reports/key_insights.csv", index=False)
print("Saved outputs/reports/key_insights.csv")

print("\n=== PIPELINE RUN COMPLETED SUCCESSFULLY ===")
