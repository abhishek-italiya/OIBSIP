# OIBSIP – Data Analytics Internship

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Welcome to the official repository for the **Oasis InfoByte Data Analytics Internship Program (OIBSIP)**. This repository contains the complete, reproducible, and end-to-end analytical solutions for all **9 Internship Tasks** covering Exploratory Data Analysis, Machine Learning, Natural Language Processing, Customer Segmentation, Imbalanced Classification, and Predictive Text Modeling.

---

## 📋 About the Internship

The **Oasis InfoByte Data Analytics Internship (OIBSIP)** is an intensive program designed to build industry-ready expertise across data manipulation, statistical modeling, machine learning algorithms, natural language processing, and business analytics. 

This repository consolidates all completed internship deliverables across Levels 1, 2, and 3 into a single, structured, professional codebase.

---

## 🛠️ Internship Tasks Overview

| Task | Project Title | Key Techniques & Methodologies |
| :--- | :--- | :--- |
| **Task 1** | **Retail Sales EDA** | Python, Pandas, Matplotlib, Seaborn, Exploratory Data Analysis, Sales Trend Mining |
| **Task 2** | **Customer Segmentation** | RFM (Recency, Frequency, Monetary) Analysis, StandardScaler, K-Means Clustering |
| **Task 3** | **Data Cleaning** | Data Hygiene, Pandas, NumPy, Missing Value Imputation, Outlier Winsorization |
| **Task 4** | **Sentiment Analysis** | NLP Preprocessing, TF-IDF Vectorization, Multinomial Naive Bayes, Logistic Regression |
| **Task 5** | **House Price Prediction** | Supervised Regression, Feature Engineering, Linear Regression, Ridge, Lasso |
| **Task 6** | **Wine Quality Prediction** | Multiclass Classification, Random Forest Classifier, Support Vector Classifier (SVC), SGD |
| **Task 7** | **Fraud Detection** | Imbalanced Learning, SMOTE Oversampling, Stratified K-Fold CV, Random Forest, PR-AUC |
| **Task 8** | **Google Play Store Analysis** | Market Catalog Profiling, Revenue Proxy Estimation, TextBlob Sentiment, Business Strategy |
| **Task 9** | **Autocomplete & Autocorrect** | N-Gram Language Modeling, Katz Backoff, Levenshtein Edit Distance, Frequency Weighting |

---

## 📁 Repository Structure

```text
OIBSIP/
├── README.md                           # Main Internship Repository Overview (This File)
├── LICENSE                             # MIT License
├── .gitignore                          # Global Git Ignore Rules
│
├── Task1_Retail_Sales_EDA/             # Task 1: Retail Sales Exploratory Data Analysis
├── Task2_Customer_Segmentation/        # Task 2: RFM Customer Segmentation & Clustering
├── Task3_Data_Cleaning/                # Task 3: Data Cleaning & Preprocessing Pipeline
├── Task4_Sentiment_Analysis/           # Task 4: Text Sentiment Classification (NLP)
├── Task5_House_Price_Prediction/       # Task 5: House Price Predictive Modeling
├── Task6_Wine_Quality_Prediction/      # Task 6: Wine Quality Multiclass Classification
├── Task7_Fraud_Detection/              # Task 7: Credit Card Fraud Detection (Imbalanced)
├── Task8_Google_Play_Store_Analysis/   # Task 8: App Store Ecosystem & Sentiment Analysis
└── Task9_Autocomplete_Autocorrect/     # Task 9: N-Gram Autocomplete & Edit-Distance Autocorrect
```

> **Note:** Each task directory is structured as a self-contained project containing its own executable Jupyter Notebook (`notebooks/`), automated execution pipeline (`run_*.py`), comprehensive documentation (`README.md`), dependency manifest (`requirements.txt`), data folders (`data/raw/`, `data/processed/`), high-resolution plots (`outputs/charts/`), and summary reports (`outputs/reports/`).

---

## 🔬 Technologies Used

- **Programming Language:** Python 3.10+
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Machine Learning & Modeling:** `scikit-learn` (`LinearRegression`, `Ridge`, `Lasso`, `RandomForestClassifier`, `SVC`, `SGDClassifier`, `KMeans`, `StandardScaler`, `TfidfVectorizer`)
- **Natural Language Processing (NLP):** `nltk`, `textblob`, N-gram Language Modeling, Katz Backoff, Levenshtein Edit Distance
- **Development Environment:** Jupyter Notebook, VS Code

---

## 🚀 How to Run the Projects

### Prerequisites
Ensure Python 3.10+ and Git are installed on your machine.

### Installation & Execution Steps (Windows / PowerShell)

1. **Clone the Repository:**
   ```powershell
   git clone https://github.com/abhishek-italiya/OIBSIP.git
   cd OIBSIP
   ```

2. **Navigate to Desired Task Folder:**
   ```powershell
   cd Task7_Fraud_Detection
   ```

3. **Install Task Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Option A: Run Automated Python Pipeline:**
   ```powershell
   python run_fraud_detection_pipeline.py
   ```

5. **Option B: Launch Jupyter Notebook:**
   ```powershell
   jupyter notebook notebooks/Task7_Fraud_Detection.ipynb
   ```

---

## 📊 Dataset & Large File Policy

- **Public Datasets:** All projects utilize legitimate, publicly accessible datasets sourced from Kaggle, UCI Machine Learning Repository, and Project Gutenberg.
- **Large Dataset Protection:** Datasets exceeding GitHub's file size limits (such as the 143 MB Kaggle `creditcard.csv` dataset in Task 7) are kept local and safely ignored via `.gitignore` to prevent repository bloat.
- **Reproducibility:** Automated download scripts (e.g., `download_creditcard_data.py`, `download_google_play_data.py`, `download_corpus_data.py`) and detailed dataset setup instructions are provided within each task's `README.md`.

---

## 📈 Summary of Demonstrated Competencies

- **Exploratory Data Analysis:** Data profiling, statistical summary, trend analysis, distribution skewness evaluation, and correlation heatmaps.
- **Data Preprocessing & Hygiene:** Handling missing values, winsorizing extreme outliers, string cleaning, datetime parsing, and categorical encoding.
- **Unsupervised Learning:** Feature scaling, elbow method optimization, silhouette scoring, and RFM customer segmentation.
- **Supervised Learning & Regression:** Multi-variate linear regression, L1/L2 regularization (Lasso/Ridge), model cross-validation, and RMSE evaluation.
- **Classification & Model Tuning:** Hyperparameter optimization across Random Forest, Support Vector Classifiers, Naive Bayes, and SGD classifiers.
- **Imbalanced Learning:** Strict data-leakage prevention (fitting scalers exclusively on train splits), SMOTE resampling, Precision-Recall curve analysis, and PR-AUC scoring.
- **Natural Language Processing:** Text normalization, TF-IDF vectorization, sentiment polarity scoring, N-gram probability modeling, Katz backoff fallback, and Levenshtein edit-distance spell correction.

---

## ✅ Internship Completion Checklist

- [x] **Task 1 – Retail Sales EDA**
- [x] **Task 2 – Customer Segmentation**
- [x] **Task 3 – Data Cleaning**
- [x] **Task 4 – Sentiment Analysis**
- [x] **Task 5 – House Price Prediction**
- [x] **Task 6 – Wine Quality Prediction**
- [x] **Task 7 – Fraud Detection**
- [x] **Task 8 – Google Play Store Analysis**
- [x] **Task 9 – Autocomplete & Autocorrect**

---

## 👤 Author

**Abhishek Italiya**  
*Data Analytics Intern – Oasis Infobyte (OIBSIP)*  
- GitHub: [abhishek-italiya](https://github.com/abhishek-italiya)  
- Repository: [OIBSIP Repository](https://github.com/abhishek-italiya/OIBSIP)

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
