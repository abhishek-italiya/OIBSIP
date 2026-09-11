import os
import re
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Ensure output directories exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# 1. Load Raw Data
print("=== 1. LOADING DATA ===")
raw_path = "data/raw/sentiment_dataset.csv"
df = pd.read_csv(raw_path)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# 2. Data Cleaning
print("\n=== 2. DATA CLEANING ===")
# Drop nulls
df_clean = df.dropna(subset=['clean_text', 'category']).copy()
# Convert text to string
df_clean['clean_text'] = df_clean['clean_text'].astype(str)

# Inspect duplicate text for conflicting labels
dup_text = df_clean[df_clean.duplicated(subset=['clean_text'], keep=False)]
print(f"Total rows with duplicate text: {len(dup_text)}")
if len(dup_text) > 0:
    conflicting = dup_text.groupby('clean_text')['category'].nunique()
    conflicting_texts = conflicting[conflicting > 1].index
    print(f"Duplicate texts with conflicting sentiment labels: {len(conflicting_texts)}")
    # Drop rows with conflicting labels for clean training
    df_clean = df_clean[~df_clean['clean_text'].isin(conflicting_texts)]
    # Drop exact duplicates keeping first
    df_clean = df_clean.drop_duplicates(subset=['clean_text'], keep='first')

# Remove empty strings or strings that are just whitespace
df_clean = df_clean[df_clean['clean_text'].str.strip() != '']
print(f"Cleaned dataset shape: {df_clean.shape}")

# 3. Class Distribution
print("\n=== 3. CLASS DISTRIBUTION ===")
dist = df_clean['category'].value_counts()
dist_pct = df_clean['category'].value_counts(normalize=True) * 100
dist_df = pd.DataFrame({'Count': dist, 'Percentage (%)': dist_pct.round(2)})
print(dist_df)

# Plot class distribution
plt.figure(figsize=(8, 5))
palette = {'Neutral': '#3498db', 'Positive': '#2ecc71', 'Negative': '#e74c3c'}
ax = sns.barplot(x=dist.index, y=dist.values, palette=palette)
plt.title("Sentiment Class Distribution", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Sentiment Class", fontsize=12)
plt.ylabel("Number of Tweets", fontsize=12)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}\n({p.get_height()/len(df_clean)*100:.1f}%)",
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', fontsize=11, color='white', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/sentiment_distribution.png", dpi=300)
plt.close()

# 4. Text Preprocessing Pipeline
print("\n=== 4. TEXT PREPROCESSING ===")
stop_words = set(stopwords.words('english'))
# Retain negation words if present in stopwords (like not, no, nor) to preserve sentiment context
negation_words = {'no', 'not', 'nor', 'neither', 'never', 'none', 'cannot', 'cant', 'couldnt', 'didnt', 'doesnt', 'dont', 'hadnt', 'hasnt', 'havent', 'isnt', 'mightnt', 'mustnt', 'neednt', 'shouldnt', 'wasnt', 'werent', 'wont', 'wouldnt'}
custom_stopwords = stop_words - negation_words
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs and user mentions (@user)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    # Remove special characters & numbers, keep letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords & lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stopwords and len(word) > 1]
    return " ".join(cleaned_tokens)

df_clean['processed_text'] = df_clean['clean_text'].apply(preprocess_text)
# Remove rows where processed_text became empty
df_clean = df_clean[df_clean['processed_text'].str.strip() != ''].copy()
print(f"Data shape after text preprocessing: {df_clean.shape}")

# Save processed dataset
df_clean.to_csv("data/processed/processed_sentiment_data.csv", index=False)
print("Saved data/processed/processed_sentiment_data.csv")

# 5. Train / Test Split
print("\n=== 5. TRAIN / TEST SPLIT ===")
X = df_clean['processed_text']
y = df_clean['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training set size: {len(X_train)} (80%)")
print(f"Test set size: {len(X_test)} (20%)")
print(f"Train class distribution:\n{y_train.value_counts(normalize=True).round(4)}")
print(f"Test class distribution:\n{y_test.value_counts(normalize=True).round(4)}")

# 6. TF-IDF Feature Extraction
print("\n=== 6. TF-IDF FEATURE EXTRACTION ===")
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"TF-IDF Train matrix shape: {X_train_tfidf.shape}")
print(f"TF-IDF Test matrix shape: {X_test_tfidf.shape}")

# 7. Model 1: Multinomial Naive Bayes
print("\n=== 7. MODEL 1: MULTINOMIAL NAIVE BAYES ===")
nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

# 8. Model 2: Logistic Regression
print("\n=== 8. MODEL 2: LOGISTIC REGRESSION ===")
lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)

# 9. Model Evaluation
print("\n=== 9. MODEL EVALUATION ===")
labels = ['Negative', 'Neutral', 'Positive']

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {
        'Accuracy': acc,
        'Macro Precision': p_macro,
        'Macro Recall': r_macro,
        'Macro F1': f1_macro,
        'Weighted Precision': p_w,
        'Weighted Recall': r_w,
        'Weighted F1': f1_w
    }

metrics_nb = get_metrics(y_test, y_pred_nb)
metrics_lr = get_metrics(y_test, y_pred_lr)

print("Naive Bayes Metrics:", metrics_nb)
print("Logistic Regression Metrics:", metrics_lr)

# Classification Reports
report_nb = classification_report(y_test, y_pred_nb, target_names=labels, output_dict=True)
report_lr = classification_report(y_test, y_pred_lr, target_names=labels, output_dict=True)

df_report_nb = pd.DataFrame(report_nb).transpose().round(4)
df_report_lr = pd.DataFrame(report_lr).transpose().round(4)

df_report_nb.to_csv("outputs/reports/classification_report_naive_bayes.csv")
df_report_lr.to_csv("outputs/reports/classification_report_logistic_regression.csv")

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_nb = confusion_matrix(y_test, y_pred_nb, labels=labels)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title('Multinomial Naive Bayes Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
axes[0].set_xlabel('Predicted Sentiment', fontsize=11)
axes[0].set_ylabel('Actual Sentiment', fontsize=11)

cm_lr = confusion_matrix(y_test, y_pred_lr, labels=labels)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title('Logistic Regression Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
axes[1].set_xlabel('Predicted Sentiment', fontsize=11)
axes[1].set_ylabel('Actual Sentiment', fontsize=11)

plt.tight_layout()
plt.savefig("outputs/charts/confusion_matrices.png", dpi=300)
plt.close()

# Separate Confusion Matrix plots for individual saving
plt.figure(figsize=(7, 5))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Multinomial Naive Bayes Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
plt.xlabel('Predicted Sentiment', fontsize=11)
plt.ylabel('Actual Sentiment', fontsize=11)
plt.tight_layout()
plt.savefig("outputs/charts/confusion_matrix_naive_bayes.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
plt.title('Logistic Regression Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
plt.xlabel('Predicted Sentiment', fontsize=11)
plt.ylabel('Actual Sentiment', fontsize=11)
plt.tight_layout()
plt.savefig("outputs/charts/confusion_matrix_logistic_regression.png", dpi=300)
plt.close()

# 10. Model Comparison
print("\n=== 10. MODEL COMPARISON ===")
comparison_df = pd.DataFrame([
    {
        'Model': 'Multinomial Naive Bayes',
        'Accuracy': round(metrics_nb['Accuracy'], 4),
        'Precision': round(metrics_nb['Weighted Precision'], 4),
        'Recall': round(metrics_nb['Weighted Recall'], 4),
        'F1 Score': round(metrics_nb['Weighted F1'], 4),
        'Macro F1': round(metrics_nb['Macro F1'], 4)
    },
    {
        'Model': 'Logistic Regression',
        'Accuracy': round(metrics_lr['Accuracy'], 4),
        'Precision': round(metrics_lr['Weighted Precision'], 4),
        'Recall': round(metrics_lr['Weighted Recall'], 4),
        'F1 Score': round(metrics_lr['Weighted F1'], 4),
        'Macro F1': round(metrics_lr['Macro F1'], 4)
    }
])
print(comparison_df.to_string(index=False))
comparison_df.to_csv("outputs/reports/model_comparison.csv", index=False)

# Bar chart model comparison
plt.figure(figsize=(9, 5))
comp_melted = pd.melt(comparison_df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'], var_name='Metric', value_name='Score')
ax = sns.barplot(data=comp_melted, x='Metric', y='Score', hue='Model', palette=['#3498db', '#2ecc71'])
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
plt.ylim(0.4, 0.85)
plt.ylabel('Score', fontsize=12)
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height) and height > 0:
        ax.annotate(f"{height:.4f}", (p.get_x() + p.get_width() / 2., height + 0.008),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.legend(title='Model', frameon=True)
plt.tight_layout()
plt.savefig("outputs/charts/model_comparison.png", dpi=300)
plt.close()

# 11. Word Cloud Generation
print("\n=== 11. WORD CLOUDS ===")
for cat, color_map, fname in [
    ('Positive', 'Greens', 'positive_wordcloud.png'),
    ('Negative', 'Reds', 'negative_wordcloud.png'),
    ('Neutral', 'Blues', 'neutral_wordcloud.png')
]:
    cat_text = " ".join(df_clean[df_clean['category'] == cat]['processed_text'])
    wc = WordCloud(width=800, height=400, background_color='white', colormap=color_map, max_words=100, random_state=42).generate(cat_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - {cat} Sentiment', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(f"outputs/charts/{fname}", dpi=300)
    plt.close()
    print(f"Saved outputs/charts/{fname}")

# 12. Error Analysis
print("\n=== 12. ERROR ANALYSIS ===")
# Best model is Logistic Regression
test_indices = y_test.index
df_test = df_clean.loc[test_indices].copy()
df_test['Actual'] = y_test
df_test['Predicted'] = y_pred_lr

misclassified = df_test[df_test['Actual'] != df_test['Predicted']].copy()
print(f"Total misclassified test examples: {len(misclassified)} / {len(df_test)} ({len(misclassified)/len(df_test)*100:.2f}%)")

# Sample 10 diverse misclassified examples
misclassified_sample = misclassified[['clean_text', 'processed_text', 'Actual', 'Predicted']].head(10)
print(misclassified_sample[['clean_text', 'Actual', 'Predicted']])

misclassified_sample.rename(columns={
    'clean_text': 'Original Text',
    'processed_text': 'Processed Text',
    'Actual': 'Actual Sentiment',
    'Predicted': 'Predicted Sentiment'
}).to_csv("outputs/reports/misclassified_examples.csv", index=False)
print("Saved outputs/reports/misclassified_examples.csv")

print("\n=== PIPELINE RUN COMPLETE ===")
