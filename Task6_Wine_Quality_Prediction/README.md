# OIBSIP Task 6 – Wine Quality Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning classification project predicting wine quality tiers from physicochemical properties developed for the **Oasis InfoByte Data Analytics Internship Program (OIBSIP)** – Task 6.

---

## 1. Project Overview
Predicting product quality from objective chemical measurements is a key application of machine learning in food and beverage manufacturing. Using the benchmark **UCI Wine Quality Dataset** (combining Red and White wine datasets), this project classifies wine quality into **3 operational quality tiers** (**`Low`**, **`Medium`**, **`High`**) based on measurable physicochemical attributes.

---

## 2. Project Objective
- **Dataset Integration & Cleaning:** Combined UCI Red (1,599 rows) and White (4,898 rows) datasets, purged 1,177 duplicate physical records yielding 5,320 unique observations.
- **Quality Tier Mapping:** Transformed discrete 3–9 quality scores into 3 operational tiers: `Low` ($\le 5$), `Medium` ($= 6$), and `High` ($\ge 7$).
- **Pipeline Preprocessing:** Implemented Scikit-Learn `Pipeline` with `StandardScaler` for scale-sensitive algorithms without data leakage.
- **Model Benchmarking:** Evaluated **Random Forest Classifier**, **Support Vector Classifier (SVC)**, and **SGD Classifier** using `class_weight='balanced'`.
- **Metric Evaluation:** Analyzed Accuracy, Macro F1-Score, Weighted F1-Score, Confusion Matrices, and Feature Importance.

---

## 3. Dataset Description & Attribution
- **Dataset Name:** UCI Wine Quality Dataset.
- **Total Records:** 6,497 raw observations (5,320 unique post-deduplication).
- **Source Attribution:** [UCI Machine Learning Repository - Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality).

### Physicochemical Attribute Descriptions:
| Attribute Name | Units | Description |
| :--- | :--- | :--- |
| `fixed acidity` | g/dm³ | Non-volatile tartaric acid content |
| `volatile acidity` | g/dm³ | Acetic acid content; excessive levels lead to sour vinegar taste |
| `citric acid` | g/dm³ | Adds freshness and flavor structure to wine |
| `residual sugar` | g/dm³ | Natural sugar remaining post-fermentation |
| `chlorides` | g/dm³ | Salt content in wine |
| `free sulfur dioxide` | mg/dm³ | Free SO₂ preventing microbial growth and oxidation |
| `total sulfur dioxide` | mg/dm³ | Combined free and bound forms of SO₂ |
| `density` | g/cm³ | Density depending on alcohol and sugar content |
| `pH` | log scale | Acidity scale (0 = highly acidic, 14 = basic) |
| `sulphates` | g/dm³ | SO₂ additive acting as antimicrobial/antioxidant |
| `alcohol` | % vol. | Percent alcohol content by volume |
| `wine_type` | binary | Wine variant (`0` = Red, `1` = White) |

---

## 4. Technologies Used
- **Programming Language:** Python 3.10+
- **Data Processing:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (`RandomForestClassifier`, `SVC`, `SGDClassifier`, `StandardScaler`, `Pipeline`, `train_test_split`, `metrics`)
- **Visualization:** `matplotlib`, `seaborn`
- **Development Environment:** Jupyter Notebook, VS Code

---

## 5. Project Structure

```text
Task6_Wine_Quality_Prediction/
│
├── data/
│   ├── raw/
│   │   ├── winequality-red.csv             # Raw UCI Red Wine dataset
│   │   └── winequality-white.csv           # Raw UCI White Wine dataset
│   └── processed/
│       └── wine_quality_processed.csv       # Merged & cleaned export
│
├── notebooks/
│   └── Task6_Wine_Quality_Prediction.ipynb  # Primary executed Jupyter Notebook
│
├── outputs/
│   ├── charts/                             # High-resolution generated plots
│   │   ├── class_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_distributions.png
│   │   ├── feature_importance.png
│   │   ├── model_comparison.png
│   │   ├── quality_distribution.png
│   │   ├── random_forest_confusion_matrix.png
│   │   ├── sgd_confusion_matrix.png
│   │   └── svc_confusion_matrix.png
│   └── reports/                            # Evaluation reports
│       ├── classification_reports.csv
│       ├── feature_importance.csv
│       └── model_comparison.csv
│
├── build_task6_notebook.py                 # Programmatic notebook builder
├── run_wine_pipeline.py                    # End-to-end Python execution pipeline
├── .gitignore                              # Git ignore rules
├── LICENSE                                 # MIT License
├── README.md                               # Project documentation
└── requirements.txt                        # Dependency specifications
```

---

## 6. Preprocessing & Stratified Train/Test Split
1. **Deduplication:** Purged 1,177 duplicate physical measurement records, leaving **5,320 unique observations**.
2. **Class Mapping:**
   - `Low` ($\le 5$): 1,985 samples (37.3%)
   - `Medium` ($= 6$): 2,325 samples (43.7%)
   - `High` ($\ge 7$): 1,010 samples (19.0%)
3. **Stratified Split:** 80% Training (4,256 samples) / 20% Testing (1,064 samples) using `random_state=42` and `stratify=y`.
4. **Feature Scaling:** Encapsulated `StandardScaler` inside Scikit-Learn pipelines for SGD and SVC to prevent data leakage.

---

## 7. Model Performance & Evaluation

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Primary)** | **0.6335** | **0.6198** | 0.6276 | **0.6233** | **0.6332** |
| **Support Vector Classifier (SVC)** | 0.6062 | 0.5988 | **0.6380** | 0.5949 | 0.5964 |
| **SGD Classifier** | 0.5893 | 0.5741 | 0.5896 | 0.5782 | 0.5923 |

* **Best Performing Model:** **Random Forest Classifier** achieved the highest overall predictive accuracy (**63.35%**) and Macro F1-score (**0.6233**).

---

## 8. Visualizations Generated

All generated charts are saved in high resolution under `outputs/charts/`:

1. **Quality Score Distribution (`quality_distribution.png`):** Histogram of original 3–9 quality ratings.
2. **Operational Class Distribution (`class_distribution.png`):** Bar chart of 3 mapped quality tiers (`Low`, `Medium`, `High`).
3. **Physicochemical Feature Distributions (`feature_distributions.png`):** Multi-panel distribution plots for all 11 chemical attributes.
4. **Correlation Heatmap (`correlation_heatmap.png`):** Pearson correlation matrix between chemical features and quality tier.
5. **Random Forest Confusion Matrix (`random_forest_confusion_matrix.png`):** Confusion matrix for the top Random Forest model.
6. **SVC Confusion Matrix (`svc_confusion_matrix.png`):** Confusion matrix for Support Vector Classifier.
7. **SGD Confusion Matrix (`sgd_confusion_matrix.png`):** Confusion matrix for SGD Classifier.
8. **Model Performance Comparison (`model_comparison.png`):** Grouped bar chart comparing Accuracy, Macro F1, and Weighted F1 across models.
9. **Feature Importance (`feature_importance.png`):** Horizontal bar chart of Random Forest feature importances.

---

## 9. Feature Importance Analysis
Top chemical predictors extracted from Random Forest:

| Rank | Feature | Importance | Chemical Influence |
| :---: | :--- | :---: | :--- |
| 1 | **`alcohol`** | **0.1549** | Primary determinant; higher alcohol correlates with premium quality |
| 2 | **`volatile acidity`** | **0.1264** | Negative driver; high acetic acid creates undesirable vinegar flavor |
| 3 | **`density`** | **0.1083** | Sugar/alcohol balance indicator |
| 4 | **`sulphates`** | **0.0914** | Antioxidant/preservative flavor protector |
| 5 | **`free sulfur dioxide`** | **0.0827** | Antimicrobial protection level |

---

## 10. Key Insights & Real-World Applications
1. **Alcohol Content Dominance:** Alcohol (% vol.) is the single strongest predictor of wine quality (~15.5% importance).
2. **Acidity Control:** Minimizing volatile acidity while maintaining balanced citric acid is essential for premium ratings.
3. **Automated Winery QA:** Enables wineries to perform rapid automated batch classification before expensive sensory tasting panels.

---

## 11. How to Run the Project

### Setup Steps
1. Navigate to project root:
   ```bash
   cd Task6_Wine_Quality_Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the automated Python ML pipeline:
   ```bash
   python run_wine_pipeline.py
   ```
4. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/Task6_Wine_Quality_Prediction.ipynb
   ```

---

## 12. Requirements & Licensing
- Dependencies specified in `requirements.txt`.
- Code source and documentation released under the [MIT License](LICENSE).
- Dataset attribution belongs to UCI Machine Learning Repository.

---

## 13. Conclusion
This project demonstrates an end-to-end classification system for wine quality prediction. **Random Forest Classifier** achieved the highest accuracy (**63.35%**) and Macro F1-score (**0.6233**), proving that objective chemical parameters effectively forecast commercial wine quality tiers.
