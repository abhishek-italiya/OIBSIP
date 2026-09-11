# OIBSIP Task 6 - Wine Quality Prediction

**Author:** Abhishek Italiya  
**Domain:** Data Analytics Internship (Oasis Infobyte)  
**Project:** Task 6 – Wine Quality Prediction Classification Project  
**Repository:** [abhishek-italiya/OIBSIP](https://github.com/abhishek-italiya/OIBSIP)  

---

## Executive Summary

This repository contains the complete implementation of **Task 6: Wine Quality Prediction** for the Oasis Infobyte Data Analytics Internship. The goal of this project is to build an end-to-end machine learning classification system that predicts wine quality categories from measurable physicochemical properties.

Using the benchmark **UCI Wine Quality Dataset** (combining both Red and White wine datasets), we analyze physical parameters such as acidity, sugar levels, chlorides, sulfur dioxide, density, pH, sulphates, and alcohol content. We evaluate three distinct machine learning classifiers—**Random Forest Classifier**, **Stochastic Gradient Descent (SGD) Classifier**, and **Support Vector Classifier (SVC)**—to identify the optimal model and key physicochemical quality drivers.

---

## Project Structure

```
Task6_Wine_Quality_Prediction/
│
├── data/
│   ├── raw/
│   │   ├── winequality-red.csv
│   │   └── winequality-white.csv
│   │
│   └── processed/
│       └── wine_quality_processed.csv
│
├── notebooks/
│   └── Task6_Wine_Quality_Prediction.ipynb
│
├── outputs/
│   ├── charts/
│   │   ├── quality_distribution.png
│   │   ├── class_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_distributions.png
│   │   ├── random_forest_confusion_matrix.png
│   │   ├── sgd_confusion_matrix.png
│   │   ├── svc_confusion_matrix.png
│   │   ├── model_comparison.png
│   │   └── feature_importance.png
│   │
│   └── reports/
│       ├── classification_reports.csv
│       ├── model_comparison.csv
│       └── feature_importance.csv
│
├── build_task6_notebook.py
├── run_wine_pipeline.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Dataset Overview

The project uses the public **UCI Wine Quality Dataset**:
- **Red Wine Dataset**: 1,599 observations (`winequality-red.csv`)
- **White Wine Dataset**: 4,898 observations (`winequality-white.csv`)
- **Combined Raw Dataset**: 6,497 total observations
- **Target Variable**: Original integer `quality` rating (3 to 9)

### Feature Descriptions
1. `fixed acidity`: Most acids involved with wine (tartaric acid, g/dm³)
2. `volatile acidity`: Amount of acetic acid in wine (g/dm³); excessive levels lead to vinegar flavor
3. `citric acid`: Found in small quantities, adds 'freshness' and flavor (g/dm³)
4. `residual sugar`: Amount of sugar remaining after fermentation stops (g/dm³)
5. `chlorides`: Amount of salt in the wine (g/dm³)
6. `free sulfur dioxide`: Free form of SO₂ preventing microbial growth and oxidation (mg/dm³)
7. `total sulfur dioxide`: Amount of free and bound forms of S0₂ (mg/dm³)
8. `density`: Density of water depending on percent alcohol and sugar content (g/cm³)
9. `pH`: Describes acidity on a logarithmic scale (0 = very acidic, 14 = basic)
10. `sulphates`: Wine additive contributing to SO₂ levels, acting as an antimicrobial/antioxidant (g/dm³)
11. `alcohol`: Percentage alcohol content by volume (% vol.)
12. `wine_type`: Encoded binary feature (`red` = 0, `white` = 1)

---

## Data Cleaning & Preprocessing

1. **Missing Values**: Verified zero missing values across both raw datasets.
2. **Duplicate Handling**: Identified 1,177 duplicate rows across physical measurements. Removed duplicate records to prevent data leakage between train/test splits, yielding **5,320 unique observations**.
3. **Target Transformation**: Mapped discrete quality scores into three operational quality classes:
   - **`Low`**: Quality score $\le 5$ (1,985 samples, 37.3%)
   - **`Medium`**: Quality score $= 6$ (2,325 samples, 43.7%)
   - **`High`**: Quality score $\ge 7$ (1,010 samples, 19.0%)
4. **Stratified Train/Test Split**: 80% training (4,256 samples) and 20% testing (1,064 samples) split with `stratify=y` and `random_state=42`.
5. **Feature Scaling**: Implemented `StandardScaler` inside Scikit-Learn `Pipeline` for scale-sensitive models (SGD and SVC) to prevent data leakage.

---

## Machine Learning Models & Evaluation

Three models were trained with `class_weight='balanced'` to address class imbalance:
1. **Random Forest Classifier** (`n_estimators=200`, `random_state=42`)
2. **SGD Classifier** (`loss='log_loss'`, `StandardScaler` pipeline)
3. **Support Vector Classifier (SVC)** (`kernel='rbf'`, `StandardScaler` pipeline)

### Empirical Performance Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.6335** | **0.6198** | 0.6276 | **0.6233** | **0.6336** | **0.6335** | **0.6332** |
| **Support Vector Classifier (SVC)** | 0.6062 | 0.5988 | **0.6380** | 0.5949 | 0.6310 | 0.6062 | 0.5964 |
| **SGD Classifier** | 0.5893 | 0.5741 | 0.5896 | 0.5782 | 0.6003 | 0.5893 | 0.5923 |

### Metric Selection Rationale
Because the dataset exhibits class imbalance (`High` class is 19.0% vs `Medium` 43.7%), **Macro F1-Score** was selected as the primary decision metric. Macro F1 evaluates unweighted mean performance across all classes, ensuring minority class performance is not masked by majority class accuracy.

**Random Forest** achieved the best performance across all metrics (Macro F1 = **0.6233**, Accuracy = **0.6335**).

---

## Feature Importance Analysis

Relative feature importances extracted from the **Random Forest** model:

| Rank | Feature | Relative Importance |
| :---: | :--- | :---: |
| 1 | `alcohol` | **0.1549** |
| 2 | `volatile acidity` | **0.1264** |
| 3 | `density` | **0.1083** |
| 4 | `sulphates` | **0.0914** |
| 5 | `free sulfur dioxide` | **0.0827** |
| 6 | `chlorides` | **0.0823** |
| 7 | `residual sugar` | **0.0784** |
| 8 | `total sulfur dioxide` | **0.0772** |
| 9 | `pH` | **0.0747** |
| 10 | `citric acid` | **0.0655** |
| 11 | `fixed acidity` | **0.0532** |
| 12 | `wine_type_code` | **0.0050** |

*Note: Feature importance indicates mathematical contribution to decision tree splits, not direct biological/chemical causation.*

---

## Key Data Insights

1. **Alcohol is the Leading Determinant**: `alcohol` content is the single most important predictor (~15.5% importance). Higher alcohol percentage strongly correlates with `High` quality wines.
2. **Volatile Acidity Negatively Impacts Quality**: High `volatile acidity` (acetic acid) is strongly associated with `Low` quality ratings due to off-putting vinegar sensory notes.
3. **Density & Sweetness Balance**: `density` (~10.8% importance) acts as a proxy for alcohol and residual sugar content, providing vital secondary class discrimination.
4. **Impact of Class Imbalance**: Incorporating `class_weight='balanced'` enabled SVC and Random Forest to achieve strong recall (>62%) on the underrepresented `High` quality tier.
5. **Universal Chemical Drivers**: Combining Red and White wines demonstrated that core chemical drivers (alcohol, acidity, sulphates) transcend specific wine styles.

---

## Real-World Applications

- **Automated Quality Screening**: Implementing real-time sensor monitoring on production lines to flag sub-standard batches early in fermentation.
- **Winery Quality Assurance**: Providing objective batch categorization before submitting wines to expensive professional sensory evaluation panels.
- **Blend Formulation**: Assisting enologists in adjusting acidity, sulphates, and alcohol parameters to optimize target flavor profiles.
- **Consumer Grading**: Standardizing quality benchmarks for retail pricing and international export compliance.

---

## Limitations

1. **Sensory Subjectivity**: Human taste preference introduces noise into ground-truth quality ratings.
2. **Regional Specificity**: Dataset covers Portuguese *Vinho Verde* wines and may require recalibration for different wine regions or grape varieties.
3. **Uncaptured Features**: Factors such as vintage year, storage temperature, oak aging time, and micro-oxygenation were not in the dataset.

---

## How to Run

### Prerequisites
Ensure Python 3.10+ is installed.

### Setup & Execution
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/abhishek-italiya/OIBSIP.git
   cd OIBSIP/Task6_Wine_Quality_Prediction
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

## Conclusion

This project successfully demonstrates an end-to-end Machine Learning solution for wine quality classification. The **Random Forest Classifier** achieved the highest predictive accuracy (**63.35%**) and Macro F1-score (**0.6233**), proving that objective chemical parameters can reliably forecast wine quality tiers.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
