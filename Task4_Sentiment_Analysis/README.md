# OIBSIP Task 4 – Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green.svg)](https://www.nltk.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) sentiment classification project developed for the **Oasis InfoByte Data Analytics Internship Program (OIBSIP)** – Task 4.

---

## 1. Project Overview
Sentiment Analysis (opinion mining) is an essential Natural Language Processing (NLP) technique used to automatically identify, extract, and quantify emotional tone behind digital text. This project delivers a machine learning pipeline that classifies social media text into **3 distinct sentiment classes**: **Positive**, **Negative**, and **Neutral**.

---

## 2. Project Objective
- **Data Acquisition & Validation:** Sourced and cleaned an authentic 3-class Twitter sentiment benchmark dataset (45,615 records).
- **Text Preprocessing Pipeline:** Built an NLTK preprocessing workflow incorporating lowercasing, regex cleaning (URLs, `@mentions`, noise), stopword filtering with **negation retention** (`not`, `no`, `never`, `cannot`), tokenization, and `WordNetLemmatizer`.
- **Feature Extraction:** Extracted numerical feature representations using `TfidfVectorizer` (unigrams + bigrams, 10,000 max features) fitted strictly on training data to prevent data leakage.
- **Model Benchmark:** Trained and compared **Multinomial Naive Bayes** and **Logistic Regression** classifiers.
- **Evaluation & Diagnostics:** Evaluated models using Accuracy, Weighted/Macro Precision, Recall, F1-Scores, Confusion Matrix Heatmaps, WordCloud visualizations, and Error Analysis.

---

## 3. Dataset Description & Attribution
- **Dataset Name:** Cardiff NLP TweetEval Standard Twitter Sentiment Analysis Dataset (SemEval Benchmark).
- **Total Records:** 45,615 raw tweets (45,584 post-cleaning).
- **Source Attribution:** [Cardiff NLP TweetEval Benchmark Dataset](https://github.com/cardiffnlp/tweeteval).
- **Licensing:** Open Academic Research License.

### Class Distribution Table:
| Sentiment Class | Sample Count | Percentage (%) | Representation |
| :--- | :---: | :---: | :--- |
| **Neutral** | 20,654 | 45.31% | Dominant majority class |
| **Positive** | 17,839 | 39.13% | Moderate secondary class |
| **Negative** | 7,091 | 15.56% | Underrepresented minority class |
| **Total Cleaned** | **45,584** | **100.00%** | **3-Class Sentiment Dataset** |

---

## 4. Technologies Used
- **Programming Language:** Python 3.10+
- **Data Processing:** `pandas`, `numpy`
- **Natural Language Processing:** `nltk` (`stopwords`, `WordNetLemmatizer`)
- **Machine Learning:** `scikit-learn` (`TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, `train_test_split`, `metrics`)
- **Visualization:** `matplotlib`, `seaborn`, `wordcloud`
- **Development Environment:** Jupyter Notebook, VS Code

---

## 5. Project Structure

```text
Task4_Sentiment_Analysis/
│
├── data/
│   ├── raw/
│   │   └── sentiment_dataset.csv             # Downloaded 3-class raw dataset
│   └── processed/
│       └── processed_sentiment_data.csv       # Cleaned & preprocessed CSV
│
├── notebooks/
│   └── Task4_Sentiment_Analysis.ipynb         # Primary executed Jupyter Notebook
│
├── outputs/
│   ├── charts/                                # High-resolution generated plots & wordclouds
│   │   ├── confusion_matrices.png
│   │   ├── confusion_matrix_logistic_regression.png
│   │   ├── confusion_matrix_naive_bayes.png
│   │   ├── model_comparison.png
│   │   ├── negative_wordcloud.png
│   │   ├── neutral_wordcloud.png
│   │   ├── positive_wordcloud.png
│   │   └── sentiment_distribution.png
│   └── reports/                               # Generated CSV reports
│       ├── classification_report_logistic_regression.csv
│       ├── classification_report_naive_bayes.csv
│       ├── misclassified_examples.csv
│       └── model_comparison.csv
│
├── build_task4_notebook.py                    # Programmatic notebook builder
├── download_sentiment_data.py                 # Automated dataset downloader
├── run_sentiment_pipeline.py                  # End-to-end Python execution pipeline
├── .gitignore                                 # Git ignore rules
├── LICENSE                                    # MIT License
├── README.md                                  # Project documentation
└── requirements.txt                           # Dependency specifications
```

---

## 6. Text Preprocessing & Feature Extraction
1. **Lowercasing:** Standardized raw text casing across all records.
2. **Regex Noise Removal:** Stripped URLs (`http...`), user handles (`@user`), numbers, and special characters.
3. **Smart Stopword Removal:** Removed general English stopwords while **preserving negation words** (`not`, `no`, `never`, `cannot`, `dont`, `wont`) to maintain critical polarity.
4. **Lemmatization:** Reduced words to base dictionary forms using NLTK `WordNetLemmatizer`.
5. **Stratified Split:** Split dataset into 80% Training (36,467 samples) and 20% Testing (9,117 samples) using `random_state=42` and `stratify=y`.
6. **TF-IDF Vectorization:** `TfidfVectorizer(max_features=10000, ngram_range=(1, 2))` fitted **strictly on `X_train`** and transformed both `X_train` and `X_test` to prevent data leakage.

---

## 7. Model Performance & Evaluation

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 60.98% | 0.6168 | 0.6098 | 0.5859 | 0.5147 |
| **Logistic Regression (Primary)** | **64.93%** | **0.6495** | **0.6493** | **0.6409** | **0.5979** |

* **Best Performing Model:** **Logistic Regression** achieved the highest accuracy (**64.93%**), weighted F1-score (**0.6409**), and macro F1-score (**0.5979**), outperforming Naive Bayes by +3.95% accuracy.

---

## 8. Visualizations Generated

All generated charts are saved in high resolution under `outputs/charts/`:

1. **Sentiment Class Distribution (`sentiment_distribution.png`):** Bar plot highlighting sample counts and percentages across Neutral (45.3%), Positive (39.1%), and Negative (15.6%).
2. **Confusion Matrices Comparison (`confusion_matrices.png`):** Side-by-side heatmaps illustrating class-wise true vs. predicted classifications.
3. **Logistic Regression Confusion Matrix (`confusion_matrix_logistic_regression.png`):** Detailed confusion matrix for the top-performing model.
4. **Naive Bayes Confusion Matrix (`confusion_matrix_naive_bayes.png`):** Confusion matrix for the baseline Naive Bayes model.
5. **Model Metric Comparison (`model_comparison.png`):** Grouped bar chart comparing Accuracy, Precision, Recall, and F1-Scores.
6. **Positive WordCloud (`positive_wordcloud.png`):** Word cloud highlighting dominant positive terms (*good*, *love*, *great*, *happy*, *thank*).
7. **Negative WordCloud (`negative_wordcloud.png`):** Word cloud highlighting dominant negative terms (*not*, *cant*, *dont*, *bad*, *worst*, *hate*).
8. **Neutral WordCloud (`neutral_wordcloud.png`):** Word cloud highlighting dominant neutral terms (*tomorrow*, *going*, *night*, *time*, *game*).

---

## 9. Error Analysis Summary
Error analysis on Logistic Regression revealed 3,197 misclassifications (35.07% test error):
- **Sarcasm & Irony:** Sarcastic tweets containing words like *"lol"* predicted as Positive despite negative intent.
- **Entity Mentions:** Neutral informational posts mentioning event names (*"Awards"*, *"Concert"*) misclassified as Positive due to word association.
- **Complex Negations:** Conditional negations (*"if you are not entertained..."*) require transformer context beyond bag-of-words/TF-IDF.

---

## 10. Key Insights & Business Applications
1. **Model Selection:** Logistic Regression's linear decision boundary optimization outperforms Naive Bayes' independence assumption on TF-IDF n-grams.
2. **Negation Impact:** Retaining negation tokens during stopword filtering significantly improved negative class recall.
3. **Real-World Utility:** Automated sentiment classification enables real-time brand reputation tracking, priority customer support routing, and product review monitoring.

---

## 11. How to Run the Project

### Setup Steps
1. Navigate to project root:
   ```bash
   cd Task4_Sentiment_Analysis
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the automated Python ML pipeline:
   ```bash
   python run_sentiment_pipeline.py
   ```
4. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/Task4_Sentiment_Analysis.ipynb
   ```

---

## 12. Requirements & Licensing
- Package dependencies specified in `requirements.txt`.
- Project source code and documentation licensed under the [MIT License](LICENSE).
- Dataset attribution belongs to Cardiff University NLP Group (TweetEval).

---

## 13. Conclusion
This project demonstrates an end-to-end NLP sentiment classification workflow. Using TF-IDF vectorization and Logistic Regression, we achieved 64.93% accuracy on a challenging 3-class social media dataset, complete with visual diagnostics, error analysis, and reproducible code.
