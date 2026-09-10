# OIBSIP Task 4 – Sentiment Analysis

**Author:** Abhishek Italiya  
**Repository:** [abhishek-italiya/OIBSIP](https://github.com/abhishek-italiya/OIBSIP)  
**Task:** Task 4 – Sentiment Analysis  
**Domain:** Data Analytics Internship (Oasis Infobyte)  

---

## 1. Project Overview
Sentiment Analysis (opinion mining) is an essential Natural Language Processing (NLP) technique used to automatically identify, extract, and quantify subjective information from text data. This project delivers an end-to-end Machine Learning pipeline that classifies social media text into **3 distinct sentiment classes**: **Positive**, **Negative**, and **Neutral**.

---

## 2. Objective
* Download and clean an authentic, publicly available 3-class text sentiment dataset.
* Implement a robust NLTK preprocessing pipeline (lowercasing, URL/user-mention removal, punctuation handling, stopword filtering with negation retention, tokenization, and lemmatization).
* Convert raw text features into numerical vectors using `TfidfVectorizer` without data leakage.
* Train and evaluate two distinct machine learning models: **Multinomial Naive Bayes** and **Logistic Regression**.
* Generate confusion matrices, metric reports, WordCloud visualizations, error analysis, and actionable business insights.

---

## 3. Dataset
* **Dataset Name:** Cardiff NLP TweetEval Standard Twitter Sentiment Analysis Dataset
* **Total Rows:** 45,615 samples (45,584 post-cleaning)
* **Features:** `clean_text` (Tweet content), `category` (Sentiment Label)
* **Classes:** 3 classes (**Neutral**, **Positive**, **Negative**)

---

## 4. Dataset Source
* **Source:** [Cardiff NLP TweetEval GitHub Repository](https://github.com/cardiffnlp/tweeteval) / SemEval Sentiment Analysis Benchmark.
* **Direct Access:** Sourced directly from open research NLP repositories.

---

## 5. Dataset License / Attribution
* **License:** Open Research Dataset License (Cardiff University NLP Group).
* **Attribution:** TweetEval benchmark dataset compiled for academic and open-source sentiment analysis research.

---

## 6. Technologies Used
* **Programming Language:** Python 3.11+
* **Data Processing:** pandas, numpy
* **Machine Learning:** scikit-learn (`TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, `train_test_split`, `metrics`)
* **Natural Language Processing:** NLTK (`stopwords`, `WordNetLemmatizer`)
* **Visualization:** matplotlib, seaborn, WordCloud
* **Interactive Notebook:** Jupyter Notebook (`ipynb`)

---

## 7. Data Cleaning
* **Missing Value Handling:** Verified zero null values in text and label columns.
* **Duplicate Inspection:** Detected 53 total duplicate tweet texts. Identified 3 duplicate texts with conflicting sentiment labels and purged them to prevent training confusion.
* **Whitespace Cleansing:** Removed empty strings and whitespace-only records.
* **Final Cleaned Rows:** 45,584 samples.

---

## 8. Sentiment Class Distribution
| Sentiment Class | Sample Count | Percentage (%) |
| :--- | :--- | :--- |
| **Neutral** | 20,654 | 45.31% |
| **Positive** | 17,839 | 39.13% |
| **Negative** | 7,091 | 15.56% |

> **Imbalance Note:** Moderate class imbalance exists as Negative tweets represent 15.56% of samples. Stratified sampling during train/test split preserved exact class ratios across subsets.

---

## 9. Text Preprocessing
1. **Lowercasing:** Standardized text string casing.
2. **Noise Reduction:** Stripped URLs (`http...`), user handles (`@user`), and non-alphabetic special characters using regex.
3. **Smart Stopword Removal:** Filtered standard English NLTK stopwords while **preserving negation words** (`not`, `no`, `never`, `cannot`, `dont`) to retain vital sentiment polarity.
4. **Lemmatization:** Reduced words to base root lemmas using NLTK `WordNetLemmatizer`.

---

## 10. Train / Test Split
* **Split Ratio:** 80% Training (36,467 samples) / 20% Testing (9,117 samples).
* **Random State:** `random_state=42`
* **Stratification:** Stratified by sentiment class label (`stratify=y`) to maintain identical class distributions in both training and test sets.

---

## 11. TF-IDF Feature Extraction
* **Vectorizer:** `TfidfVectorizer(max_features=10000, ngram_range=(1, 2))`
* **Data Leakage Prevention:** Fitted strictly on `X_train`, then transformed `X_train` and `X_test`.
* **Output Matrix Shape:** `(36467, 10000)` for train set, `(9117, 10000)` for test set.

---

## 12. Model Training
1. **Model 1 — Multinomial Naive Bayes (`MultinomialNB`):** Probabilistic classifier fitted on TF-IDF features (`alpha=1.0`).
2. **Model 2 — Logistic Regression (`LogisticRegression`):** Linear classifier fitted on TF-IDF features with `max_iter=1000`, `random_state=42`.

---

## 13. Model Evaluation & Comparison

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Macro F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Multinomial Naive Bayes** | 60.98% | 0.6168 | 0.6098 | 0.5859 | 0.5147 |
| **Logistic Regression** | **64.93%** | **0.6495** | **0.6493** | **0.6409** | **0.5979** |

* **Best Performing Model:** **Logistic Regression** outperformed Naive Bayes across all metrics (+3.95% Accuracy boost, +0.055 Weighted F1 improvement).

---

## 14. Confusion Matrices
Seaborn heatmaps were generated for both models and saved to `outputs/charts/`:
* `outputs/charts/confusion_matrices.png`
* `outputs/charts/confusion_matrix_naive_bayes.png`
* `outputs/charts/confusion_matrix_logistic_regression.png`

---

## 15. WordCloud Analysis
High-resolution word clouds were created for each sentiment class:
* **Positive WordCloud:** `outputs/charts/positive_wordcloud.png` (Key terms: *good*, *love*, *day*, *thank*, *great*, *happy*)
* **Negative WordCloud:** `outputs/charts/negative_wordcloud.png` (Key terms: *not*, *cant*, *dont*, *bad*, *hate*, *shit*, *worst*)
* **Neutral WordCloud:** `outputs/charts/neutral_wordcloud.png` (Key terms: *tomorrow*, *going*, *night*, *time*, *game*, *see*)

---

## 16. Error Analysis
* **Evaluated Model:** Logistic Regression (Best Model).
* **Total Misclassifications:** 3,197 out of 9,117 test samples (35.07%).
* **Saved Report:** `outputs/reports/misclassified_examples.csv`
* **Key Misclassification Causes Identified:**
  1. **Sarcasm & Irony:** Sarcastic tweets containing words like *"lol"* or *"great"* predicted as Positive despite negative intent.
  2. **Domain-Specific Entity Mentions:** Words like *"Awards"*, *"Concert"*, or *"Tournament"* pulling neutral tweets towards Positive.
  3. **Complex Sentence Structure & Subtlety:** Conditional negations requiring deep contextual language understanding beyond bag-of-words/TF-IDF.

---

## 17. Key Insights
* **Logistic Regression Superiority:** Logistic Regression effective linear boundary optimization and L2 regularization make it superior for high-dimensional TF-IDF vectors compared to Naive Bayes independence assumptions.
* **Negation Retention:** Keeping negation tokens in stopword removal directly improved negative class recall.
* **Business Value:** Enables automated monitoring of customer sentiment across social media platforms.

---

## 18. Real-World Applications
* **Social Media Reputation Tracking:** Instant alerting on negative brand sentiment spikes.
* **Customer Support Ticket Prioritization:** Automatic escalation of angry or dissatisfied customer tweets.
* **Product Launch Feedback Analysis:** Real-time sentiment tracking during marketing campaigns.

---

## 19. Project Structure
```text
Task4_Sentiment_Analysis/
│
├── data/
│   ├── raw/
│   │   └── sentiment_dataset.csv
│   └── processed/
│       └── processed_sentiment_data.csv
│
├── notebooks/
│   └── Task4_Sentiment_Analysis.ipynb
│
├── outputs/
│   ├── charts/
│   │   ├── confusion_matrices.png
│   │   ├── confusion_matrix_logistic_regression.png
│   │   ├── confusion_matrix_naive_bayes.png
│   │   ├── model_comparison.png
│   │   ├── negative_wordcloud.png
│   │   ├── neutral_wordcloud.png
│   │   ├── positive_wordcloud.png
│   │   └── sentiment_distribution.png
│   ├── models/
│   └── reports/
│       ├── classification_report_logistic_regression.csv
│       ├── classification_report_naive_bayes.csv
│       ├── misclassified_examples.csv
│       └── model_comparison.csv
│
├── build_task4_notebook.py
├── run_sentiment_pipeline.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 20. How to Run
1. Navigate to project root:
   ```bash
   cd Task4_Sentiment_Analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline script:
   ```bash
   python run_sentiment_pipeline.py
   ```
4. Open and execute the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/Task4_Sentiment_Analysis.ipynb
   ```

---

## 21. Conclusion
This project successfully demonstrates an end-to-end sentiment classification workflow. With TF-IDF vectorization and Logistic Regression, we achieved 64.93% accuracy on a challenging 3-class Twitter dataset, complete with visual reports, error analysis, and reproducible notebook execution.

---

## License
Project code is licensed under the [MIT License](LICENSE).
