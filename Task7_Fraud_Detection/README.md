# OIBSIP Task 7 – Credit Card Fraud Detection

**Author:** Abhishek Italiya  
**Repository:** [abhishek-italiya/OIBSIP](https://github.com/abhishek-italiya/OIBSIP)  
**Task:** Task 7 – Fraud Detection  
**Domain:** Data Analytics Internship (Oasis Infobyte)  

---

## Overview
Credit card fraud detection is a critical application of machine learning in financial risk management. Detecting fraudulent transactions in real-time protects cardholders, reduces financial loss, and preserves consumer trust. This project delivers an end-to-end Machine Learning solution designed to classify transactions as **Legitimate (0)** or **Fraudulent (1)** under extreme class imbalance.

---

## Objective
* Download and inspect the public European Credit Card Fraud Detection dataset (284,807 transactions).
* Perform exploratory data analysis on transaction amounts, time patterns, and feature correlations.
* Demonstrate why standard Accuracy is misleading in fraud detection and emphasize Precision, Recall, F1-Score, ROC-AUC, and Precision-Recall AUC (PR-AUC).
* Implement feature scaling using `StandardScaler` and stratified 80/20 train/test splitting.
* Train and compare **Logistic Regression** and **Random Forest Classifier** models utilizing cost-sensitive class weighting (`class_weight='balanced'`).
* Extract feature importances, analyze decision threshold trade-offs, and outline scalability requirements for processing 1,000,000 transactions/hour in production.

---

## Dataset
* **Dataset Name:** Credit Card Fraud Detection Dataset (European Cardholder Transactions)
* **Total Transactions:** 284,807
* **Features:** 31 columns (`Time`, `V1`–`V28`, `Amount`, `Class`)
* **Anonymized PCA Features:** `V1` through `V28` are principal components derived from PCA to maintain privacy.
* **Unscaled Features:** `Time` (seconds elapsed since first transaction) and `Amount` (transaction dollar value).
* **Target Feature:** `Class` ($0 = \\text{Legitimate}$, $1 = \\text{Fraudulent}$).

---

## Dataset Statistics & Class Imbalance
| Transaction Class | Count | Percentage (%) |
| :--- | :--- | :--- |
| **Legitimate (0)** | 284,315 | 99.8273% |
| **Fraudulent (1)** | 492 | 0.1727% |
| **Total** | **284,807** | **100.0000%** |

> **Critical Imbalance Note:** Fraud represents only **0.1727%** of transactions (~1 in 578 transactions). A naive model predicting every transaction as legitimate achieves 99.83% accuracy while missing 100% of fraud.

---

## Exploratory Data Analysis
1. **Time Distribution:** Analyzed transaction volume across seconds elapsed, revealing cyclic 24-hour day/night payment activity.
2. **Amount Distribution:** Heavily right-skewed; most transactions are low value (< $100), while fraudulent transactions concentrate in moderate amounts ($10–$500).
3. **Correlation Heatmap:** Identified strong negative correlations with `Class` for `V14`, `V17`, `V12`, `V10`, and `V3`, alongside positive correlations for `V4`, `V11`, and `V2`.

---

## Data Preprocessing & Imbalance Handling
* **Feature Scaling:** Applied `StandardScaler` to `Time` and `Amount` fitted **strictly on training data** (`X_train`) to prevent data leakage.
* **Train / Test Split:** Stratified 80% Training (227,845 samples, 394 frauds) / 20% Testing (56,962 samples, 98 frauds) with `random_state=42`.
* **Cost-Sensitive Weighting:** Utilized `class_weight='balanced'` in loss functions to penalize fraud misclassifications inversely proportional to class frequencies.

---

## Models Trained
1. **Model 1 — Logistic Regression:** Linear classifier with `class_weight='balanced'`, `max_iter=1000`, `random_state=42`.
2. **Model 2 — Random Forest Classifier:** Ensemble decision tree classifier with `n_estimators=100`, `max_depth=12`, `class_weight='balanced'`, `n_jobs=-1`, `random_state=42`.

---

## Evaluation Metrics & Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (Avg Precision) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 97.5475% | 0.0609 | **0.9184** | 0.1141 | 0.9722 | 0.7189 |
| **Random Forest** | **99.9175%** | **0.7217** | 0.8469 | **0.7793** | **0.9859** | **0.8303** |

* **Fraud Class Performance (98 Total Fraud Test Cases):**
  * **Logistic Regression:** Caught 90 of 98 fraud cases (91.84% Recall) but generated 1,387 false positives (6.09% Precision).
  * **Random Forest:** Caught 83 of 98 fraud cases (84.69% Recall) with only 32 false positives (72.17% Precision).
* **Best Model Selection:** **Random Forest Classifier** is selected as the top model due to its high precision (72.17%), strong F1-Score (0.7793), and superior PR-AUC (0.8303).

---

## Feature Importance
Top 10 features extracted from Random Forest (`outputs/reports/feature_importance.csv`):

| Rank | Feature | Importance | Cumulative Contribution |
| :--- | :--- | :--- | :--- |
| 1 | **V14** | 0.2160 | 21.60% |
| 2 | **V10** | 0.1127 | 32.87% |
| 3 | **V4** | 0.1085 | 43.72% |
| 4 | **V17** | 0.0906 | 52.78% |
| 5 | **V12** | 0.0797 | 60.75% |
| 6 | **V3** | 0.0750 | 68.25% |
| 7 | **V11** | 0.0648 | 74.73% |
| 8 | **V16** | 0.0363 | 78.36% |
| 9 | **V7** | 0.0280 | 81.16% |
| 10 | **V2** | 0.0263 | 83.79% |

---

## Threshold Trade-off Analysis
* Default threshold $p = 0.5$ balances precision and recall.
* **Lowering Threshold ($p = 0.2$):** Increases Recall (catches more fraud) at the cost of higher customer false alarms.
* **Raising Threshold ($p = 0.7$):** Maximizes Precision (minimizes card declines) at the risk of letting sophisticated fraud pass through.

---

## Scalability to 1 Million Transactions / Hour
To process ~278 transactions per second peak throughput:
* **Event Ingestion:** Streaming via **Apache Kafka** / AWS Kinesis queues.
* **Feature Store:** Low-latency online aggregations via **Feast** or Redis ($< 5\\text{ms}$).
* **Model Serving:** Microservice API (FastAPI / C++ ONNX runtime) returning predictions in $< 20\\text{ms}$.
* **Queue Prioritization:** High-confidence alerts trigger immediate card blocks; medium-confidence alerts route to manual risk analyst queues.

---

## Key Insights
1. **Class Imbalance:** Extreme 99.83% vs 0.17% ratio demands cost-sensitive loss functions and PR-AUC evaluation.
2. **Accuracy Fallacy:** 99.9% accuracy is trivial; PR-AUC (0.8303) and Recall (84.69%) measure true business effectiveness.
3. **Random Forest Dominance:** Ensemble trees outperform linear models by capturing non-linear interactions between PCA components.
4. **Key Signals:** Features `V14`, `V10`, `V4`, `V17`, and `V12` drive over 60% of model decision logic.

---

## Real-World Applications
* Real-time payment screening during credit card swipe / checkout.
* Automated escalation rules for banking risk operations.
* Payment gateway screening for e-commerce fraud prevention.

---

## Limitations
* PCA anonymization hides underlying transaction context (e.g., merchant ID, device IP).
* Historical static data cannot measure real-time concept drift or evolving fraud tactics.

---

## Project Structure
```text
Task7_Fraud_Detection/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv (gitignored due to 150MB GitHub limit)
│   └── processed/
│       └── fraud_detection_processed.csv
│
├── notebooks/
│   └── Task7_Fraud_Detection.ipynb
│
├── outputs/
│   ├── charts/
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
│   └── reports/
│       ├── class_distribution_report.csv
│       ├── classification_reports.csv
│       ├── feature_importance.csv
│       └── model_metrics.csv
│
├── build_task7_notebook.py
├── download_creditcard_data.py
├── run_fraud_detection_pipeline.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## How to Run

1. Clone the repository and navigate to Task 7:
   ```bash
   cd Task7_Fraud_Detection
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the Credit Card Fraud Detection dataset:
   ```bash
   python download_creditcard_data.py
   ```
4. Execute the pipeline to refresh metrics and charts:
   ```bash
   python run_fraud_detection_pipeline.py
   ```
5. Launch and run the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/Task7_Fraud_Detection.ipynb
   ```

---

## License
Project source code is licensed under the [MIT License](LICENSE).
