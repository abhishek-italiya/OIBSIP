# OIBSIP Task 7 – Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-0.11%2B-008080.svg)](https://imbalanced-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning classification project for credit card fraud detection under severe class imbalance developed for the **Oasis InfoByte Data Analytics Internship Program (OIBSIP)** – Task 7.

---

## 1. Project Overview
Credit card fraud detection is a critical application of machine learning in financial risk management. Detecting fraudulent transactions in real-time protects cardholders, reduces financial loss, and preserves consumer trust. This project delivers a machine learning solution designed to classify transactions as **Legitimate (0)** or **Fraudulent (1)** under extreme class imbalance without data leakage.

---

## 2. Project Objective
- **Dataset Acquisition & Inspection:** Sourced and inspected the public European Credit Card Fraud Detection dataset (284,807 transactions).
- **Exploratory Data Analysis:** Analyzed transaction amounts, time patterns, and feature correlations.
- **Accuracy Fallacy Analysis:** Demonstrated why standard Accuracy is misleading in fraud detection and emphasized Precision, Recall, F1-Score, ROC-AUC, and Precision-Recall AUC (PR-AUC).
- **Non-Leaking Preprocessing Pipeline:** Implemented feature scaling using `StandardScaler` fitted **STRICTLY on training data** (`X_train`) after an 80/20 stratified split.
- **Model Training & Benchmarking:** Trained and compared **Logistic Regression** and **Random Forest Classifier** models utilizing cost-sensitive class weighting (`class_weight='balanced'`).
- **Feature Importance & Scalability:** Extracted top feature importances, analyzed decision threshold trade-offs, and outlined scalability requirements for processing 1,000,000 transactions/hour in production.

---

## 3. Dataset Description & Attribution
- **Dataset Name:** Credit Card Fraud Detection Dataset (European Cardholder Transactions).
- **Total Transactions:** 284,807 transaction rows.
- **Features:** 31 columns (`Time`, `V1`–`V28`, `Amount`, `Class`).
- **Anonymized PCA Features:** `V1` through `V28` are principal components derived from PCA to maintain privacy.
- **Unscaled Features:** `Time` (seconds elapsed since first transaction) and `Amount` (transaction dollar value).
- **Target Feature:** `Class` ($0 = \\text{Legitimate}$, $1 = \\text{Fraudulent}$).
- **Source Attribution:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### Class Distribution Table:
| Transaction Class | Count | Percentage (%) | Representation |
| :--- | :---: | :---: | :--- |
| **Legitimate (0)** | 284,315 | 99.8273% | Dominant majority class |
| **Fraudulent (1)** | 492 | 0.1727% | Extreme minority class (~1 in 578) |
| **Total** | **284,807** | **100.0000%** | **Highly Imbalanced Dataset** |

> **Dataset Storage Note:** The raw dataset (`creditcard.csv` ~150 MB) and full processed dataset (`fraud_detection_processed.csv` ~126 MB) are intentionally **NOT stored on GitHub** to comply with GitHub's 100 MB per-file upload limit. They remain stored locally inside `data/raw/creditcard.csv` and `data/processed/fraud_detection_processed.csv`. Anyone reproducing the project can automatically download the authentic public dataset using `python download_creditcard_data.py` or manually place `creditcard.csv` into `data/raw/creditcard.csv` before running `run_fraud_detection_pipeline.py`.

---

## 4. Technologies Used
- **Programming Language:** Python 3.10+
- **Data Processing:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (`LogisticRegression`, `RandomForestClassifier`, `StandardScaler`, `train_test_split`, `metrics`)
- **Imbalance Handling:** `imbalanced-learn`
- **Visualization:** `matplotlib`, `seaborn`
- **Development Environment:** Jupyter Notebook, VS Code

---

## 5. Project Structure

```text
Task7_Fraud_Detection/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv                 # Raw dataset (gitignored due to 150MB limit)
│   └── processed/
│       └── fraud_detection_processed.csv   # Cleaned modeling sample export
│
├── notebooks/
│   └── Task7_Fraud_Detection.ipynb        # Primary executed Jupyter Notebook
│
├── outputs/
│   ├── charts/                            # High-resolution generated plots
│   │   ├── class_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_importance.png
│   │   ├── fraud_amount_comparison.png
│   │   ├── logistic_regression_confusion_matrix.png
│   │   ├── model_comparison.png
│   │   ├── precision_recall_curve.png
│   │   ├── random_forest_confusion_matrix.png
│   │   ├── roc_auc_comparison.png
│   │   ├── transaction_amount_distribution.png
│   │   ├── transaction_time_distribution.png
│   │   └── tree_confusion_matrix.png
│   └── reports/                           # Evaluation reports
│       ├── class_distribution_report.csv
│       ├── classification_reports.csv
│       ├── feature_importance.csv
│       └── model_metrics.csv
│
├── build_task7_notebook.py                # Programmatic notebook builder
├── download_creditcard_data.py            # Automated dataset downloader
├── run_fraud_detection_pipeline.py        # End-to-end Python execution pipeline
├── .gitignore                             # Git ignore rules
├── LICENSE                                # MIT License
├── README.md                              # Project documentation
└── requirements.txt                       # Dependency specifications
```

---

## 6. Strict Non-Leaking Data Preprocessing
- **Data Leakage Prevention:** Feature matrix $X$ and target vector $y$ are separated **FIRST**, followed immediately by an 80/20 stratified train/test split.
- **Feature Scaling:** `StandardScaler` is fitted **STRICTLY on `X_train[['Amount', 'Time']]`**, and then used to transform `X_test[['Amount', 'Time']]`. Test data is never used to calculate scaling statistics.
- **Stratified Split:** 80% Training (227,845 samples, 394 frauds) / 20% Testing (56,962 samples, 98 frauds) with `random_state=42`.
- **Cost-Sensitive Weighting:** Utilized `class_weight='balanced'` in loss functions to penalize fraud misclassifications inversely proportional to class frequencies.

---

## 7. Model Performance & Evaluation

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (Avg Precision) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 97.5528% | 0.0610 | **0.9184** | 0.1144 | 0.9722 | 0.7159 |
| **Random Forest (Primary)** | **99.9140%** | **0.7168** | 0.8265 | **0.7678** | **0.9790** | **0.8236** |

* **Fraud Class Performance (98 Total Fraud Test Cases):**
  * **Logistic Regression:** Caught 90 of 98 fraud cases (91.84% Recall) but generated 1,387 false positives (6.10% Precision).
  * **Random Forest:** Caught 81 of 98 fraud cases (82.65% Recall) with 71.68% Precision and 0.8236 PR-AUC.
* **Best Model Selection:** **Random Forest Classifier** is selected as the top model due to its high precision (71.68%), strong F1-Score (0.7678), and superior PR-AUC (0.8236).

---

## 8. Visualizations Generated

All generated visualizations are saved in high resolution under `outputs/charts/`:

1. **Class Distribution (`class_distribution.png`):** Logarithmic scale bar chart illustrating extreme class imbalance.
2. **Transaction Time Distribution (`transaction_time_distribution.png`):** Histogram showing 48-hour payment volume cycles.
3. **Transaction Amount Distribution (`transaction_amount_distribution.png`):** Log-scaled histogram of order values.
4. **Fraud Amount Comparison (`fraud_amount_comparison.png`):** Boxplot comparing transaction amounts for legitimate vs fraudulent cases.
5. **Correlation Heatmap (`correlation_heatmap.png`):** Correlation matrix heatmap for top features correlated with `Class`.
6. **Logistic Regression Confusion Matrix (`logistic_regression_confusion_matrix.png`):** Heatmap confusion matrix for Logistic Regression.
7. **Random Forest Confusion Matrix (`random_forest_confusion_matrix.png`):** Heatmap confusion matrix for Random Forest Classifier.
8. **ROC-AUC Comparison Curve (`roc_auc_comparison.png`):** Plot comparing ROC curves and AUC values.
9. **Precision-Recall Curve (`precision_recall_curve.png`):** Plot comparing PR curves and Average Precision scores.
10. **Model Metrics Comparison (`model_comparison.png`):** Grouped bar chart comparing metrics across models.
11. **Feature Importance (`feature_importance.png`):** Bar chart displaying top 15 Random Forest feature importances.

---

## 9. Feature Importance Analysis
Top 10 features extracted from Random Forest (`outputs/reports/feature_importance.csv`):

| Rank | Feature | Importance | Cumulative Contribution |
| :---: | :--- | :---: | :---: |
| 1 | **`V14`** | 0.1963 | 19.63% |
| 2 | **`V10`** | 0.1105 | 30.68% |
| 3 | **`V4`** | 0.1079 | 41.47% |
| 4 | **`V12`** | 0.0985 | 51.32% |
| 5 | **`V17`** | 0.0896 | 60.28% |
| 6 | **`V3`** | 0.0634 | 66.62% |
| 7 | **`V11`** | 0.0499 | 71.61% |
| 8 | **`V16`** | 0.0461 | 76.22% |
| 9 | **`V2`** | 0.0373 | 79.95% |
| 10 | **`V9`** | 0.0249 | 82.44% |

---

## 10. Key Insights & Production Scalability
1. **Accuracy Fallacy:** 99.9% accuracy is trivial; PR-AUC (0.8236) and Recall (82.65%) measure true business effectiveness.
2. **Dominant Features:** Features `V14`, `V10`, `V4`, `V12`, and `V17` drive over 60% of model decision logic.
3. **Scalability to 1M Transactions/Hour:** Requires streaming ingestion via **Apache Kafka**, low-latency feature stores (**Feast** / Redis $< 5\text{ms}$), and microservice serving (FastAPI / C++ ONNX runtime $< 20\text{ms}$).

---

## 11. How to Run the Project

### Setup Steps
1. Navigate to project root:
   ```bash
   cd Task7_Fraud_Detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the Credit Card Fraud Detection dataset:
   ```bash
   python download_creditcard_data.py
   ```
4. Run the automated Python ML pipeline:
   ```bash
   python run_fraud_detection_pipeline.py
   ```
5. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/Task7_Fraud_Detection.ipynb
   ```

---

## 12. Requirements & Licensing
- Dependencies specified in `requirements.txt`.
- Code source and documentation released under the [MIT License](LICENSE).
- Dataset attribution belongs to Machine Learning Group (MLG) - ULB / Kaggle.

---

## 13. Conclusion
This project demonstrates an end-to-end fraud detection system under severe class imbalance. **Random Forest Classifier** achieved **0.8236 PR-AUC** and **82.65% Recall**, establishing a robust foundation for scalable production risk management systems.
