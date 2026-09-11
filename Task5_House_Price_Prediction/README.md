# OIBSIP Task 5 – House Prices with Linear Regression

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning regression project for residential real estate valuation developed for the **Oasis InfoByte Data Analytics Internship Program (OIBSIP)** – Task 5.

---

## 1. Project Overview
Predicting residential real estate prices is a cornerstone application of predictive analytics and machine learning. This project delivers an end-to-end regression pipeline that estimates residential property prices (`SalePrice`) using structural, physical, quality, and location features.

---

## 2. Project Objective
- **Dataset Acquisition & Inspection:** Sourced and cleaned the benchmark Kaggle House Prices dataset (Ames Housing Dataset).
- **Exploratory Data Analysis:** Identified distribution right-skewness, missing values, and high-correlation feature pairs.
- **Leakage-Free Preprocessing Pipeline:** Constructed a Scikit-Learn `ColumnTransformer` pipeline performing numerical median imputation + standardization and categorical mode imputation + One-Hot Encoding.
- **Regression Modeling & Benchmarking:** Evaluated **Linear Regression** alongside regularized variants (**Ridge Regression** & **Lasso Regression**).
- **Model Evaluation:** Computed Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and $R^2$ Score.
- **Diagnostic Residual & Feature Coefficient Analysis:** Identified dominant structural and quality valuation drivers.

---

## 3. Dataset Description & Attribution
- **Dataset Name:** Kaggle House Prices – Advanced Regression Techniques Dataset (Ames Housing Dataset).
- **Total Records:** 1,460 property rows.
- **Total Attributes:** 81 features (79 explanatory variables, `Id`, and target `SalePrice`).
- **Target Variable:** `SalePrice` (Continuous monetary valuation in USD).
- **Source Attribution:** [Kaggle - House Prices Advanced Regression Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

### Selected Feature Descriptions:
| Feature Name | Feature Type | Description |
| :--- | :--- | :--- |
| `OverallQual` | Numerical | Overall material and finish quality rating (1–10) |
| `GrLivArea` | Numerical | Above grade (ground) living area square feet |
| `GarageCars` | Numerical | Size of garage in car capacity |
| `GarageArea` | Numerical | Size of garage in square feet |
| `TotalBsmtSF` | Numerical | Total square feet of basement area |
| `1stFlrSF` | Numerical | First floor area in square feet |
| `FullBath` | Numerical | Full bathrooms above grade |
| `TotRmsAbvGrd` | Numerical | Total rooms above grade (excluding bathrooms) |
| `YearBuilt` | Numerical | Original construction date |
| `YearRemodAdd` | Numerical | Remodel date |
| `Neighborhood` | Categorical | Physical location within Ames city limits |
| `ExterQual` | Categorical | Exterior material quality evaluation |
| `KitchenQual` | Categorical | Kitchen quality evaluation |

---

## 4. Technologies Used
- **Programming Language:** Python 3.10+
- **Data Processing:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (`LinearRegression`, `Ridge`, `Lasso`, `ColumnTransformer`, `Pipeline`, `StandardScaler`, `OneHotEncoder`, `train_test_split`, `metrics`)
- **Visualization:** `matplotlib`, `seaborn`
- **Development Environment:** Jupyter Notebook, VS Code

---

## 5. Project Structure

```text
Task5_House_Price_Prediction/
│
├── data/
│   ├── raw/
│   │   └── train.csv                       # Downloaded raw Ames Housing dataset
│   └── processed/
│       └── house_prices_processed.csv       # Cleaned modeling export
│
├── notebooks/
│   └── Task5_House_Price_Prediction.ipynb  # Primary executed Jupyter Notebook
│
├── outputs/
│   ├── charts/                             # High-resolution generated plots
│   │   ├── actual_vs_predicted.png
│   │   ├── coefficient_importance.png
│   │   ├── correlation_heatmap.png
│   │   ├── residual_plot.png
│   │   ├── saleprice_distribution.png
│   │   └── top_features_correlation.png
│   └── reports/                            # CSV evaluation reports
│       ├── feature_coefficients.csv
│       ├── missing_values_summary.csv
│       └── model_metrics.csv
│
├── build_task5_notebook.py                 # Programmatic notebook builder
├── run_house_price_pipeline.py             # End-to-end Python execution pipeline
├── .gitignore                              # Git ignore rules
├── LICENSE                                 # MIT License
├── README.md                               # Project documentation
└── requirements.txt                        # Dependency specifications
```

---

## 6. Preprocessing & Modeling Pipeline
To prevent **data leakage**, all preprocessing steps are contained inside a Scikit-Learn `ColumnTransformer` pipeline fitted strictly on the training set (`X_train`):
1. **Numerical Features (13):** Median imputation (`SimpleImputer`) followed by `StandardScaler()`.
2. **Categorical Features (5):** Mode imputation (`SimpleImputer`) followed by `OneHotEncoder(handle_unknown='ignore')`.
3. **Train / Test Split:** 80% Training (1,168 samples) / 20% Testing (292 samples) using `random_state=42`.

---

## 7. Model Performance & Evaluation

| Model | MSE | RMSE ($) | $R^2$ Score | Variance Explained |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression (Primary)** | **1,072,759,628** | **$32,753.01** | **0.8601** | **86.01%** |
| **Lasso Regression (Alpha=100)** | 1,091,482,279 | $33,037.59 | 0.8577 | 85.77% |
| **Ridge Regression (Alpha=10)** | 1,106,747,208 | $33,267.81 | 0.8557 | 85.57% |

* **Best Performing Model:** **Linear Regression** achieved an **$R^2$ Score of 0.8601** (explaining **86.01%** of price variance) with a **Root Mean Squared Error (RMSE) of $32,753.01**.

---

## 8. Visualizations Generated

All generated visualizations are saved in high resolution under `outputs/charts/`:

1. **SalePrice Distribution (`saleprice_distribution.png`):** Histogram and KDE curve illustrating property price right-skewness.
2. **Correlation Heatmap (`correlation_heatmap.png`):** Pearson correlation heatmap for top numerical variables.
3. **Top Features Correlation (`top_features_correlation.png`):** Bar plot highlighting variables most strongly correlated with `SalePrice`.
4. **Actual vs. Predicted Prices (`actual_vs_predicted.png`):** Scatter plot comparing actual house prices vs model predictions with $y = x$ reference line.
5. **Residual Plot (`residual_plot.png`):** Residual diagnostics scatter plot centered around zero verifying model error behavior.
6. **Coefficient Importance (`coefficient_importance.png`):** Horizontal bar chart displaying top positive and negative standardized regression coefficients.

---

## 9. Key Insights & Feature Analysis
1. **Overall Quality (`OverallQual`):** Strongest single predictor ($r = +0.791$). A 1-standard-deviation increase in quality rating corresponds to ~$21,900 increase in predicted sale price.
2. **Living Area Size (`GrLivArea`):** Above-ground living space ($r = +0.709$) is the second most dominant valuation factor.
3. **Garage & Basement Capacity:** Total basement area (`TotalBsmtSF`) and garage car capacity (`GarageCars`) add substantial collateral value.
4. **Location Premium:** Properties in premium neighborhoods (*Northridge Heights*, *Stone Brook*) command high location premiums.

---

## 10. Real-World Applications
- **Automated Valuation Models (AVMs):** Real estate portals (Zillow, Redfin) utilize linear regression baselines for algorithmic property pricing.
- **Mortgage Underwriting:** Financial institutions assess property collateral values during loan approvals.
- **Renovation ROI Analytics:** Homeowners evaluate expected return on investment for structural expansions.

---

## 11. Model Limitations
- **Linearity Assumption:** Linear models assume linear feature-target relationships, slightly underestimating extreme high-end luxury estates.
- **Heteroscedasticity:** Error variance increases for properties valued above $400,000.

---

## 12. How to Run the Project

### Setup Steps
1. Navigate to project root:
   ```bash
   cd Task5_House_Price_Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the automated Python ML pipeline:
   ```bash
   python run_house_price_pipeline.py
   ```
4. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/Task5_House_Price_Prediction.ipynb
   ```

---

## 13. Requirements & Licensing
- Package dependencies specified in `requirements.txt`.
- Code source and documentation released under the [MIT License](LICENSE).
- Dataset attribution belongs to Kaggle / Ames Housing Dataset.

---

## 14. Conclusion
This project delivers a complete regression pipeline for house price estimation. **Linear Regression** achieved an $R^2$ score of **0.8601** and RMSE of **$32,753.01**, validating that key structural and quality features effectively predict property valuations.
