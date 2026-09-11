# OIBSIP Task 5 – House Prices with Linear Regression

**Author:** Abhishek Italiya  
**Repository:** [abhishek-italiya/OIBSIP](https://github.com/abhishek-italiya/OIBSIP)  
**Task:** Task 5 – House Price Prediction with Linear Regression  
**Domain:** Data Analytics Internship (Oasis Infobyte)  

---

## 1. Project Overview
Predicting residential real estate prices is a cornerstone application of predictive analytics and machine learning. This project delivers an end-to-end Machine Learning regression pipeline that estimates residential property prices (`SalePrice`) using structural, physical, quality, and location features.

---

## 2. Objective
* Download, inspect, and clean the benchmark **Kaggle House Prices – Advanced Regression Techniques dataset (`train.csv`)**.
* Conduct extensive Exploratory Data Analysis (EDA), missing value handling, and correlation analysis.
* Build a leakage-free Scikit-Learn preprocessing and modeling pipeline utilizing **Linear Regression**.
* Evaluate and benchmark **Linear Regression** against regularized models (**Ridge Regression** & **Lasso Regression**).
* Calculate key metrics: **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **$R^2$ Score**.
* Perform **Residual Analysis** and **Coefficient Interpretation** to highlight critical real estate valuation drivers.

---

## 3. Dataset Information
* **Dataset Name:** Kaggle House Prices – Advanced Regression Techniques Dataset (Ames Housing Dataset).
* **Dataset Source:** OpenML Benchmark Repository / Kaggle Competition (`train.csv`).
* **Total Samples (Rows):** 1,460 properties.
* **Total Attributes (Columns):** 81 features (79 explanatory features, `Id`, and target `SalePrice`).
* **Target Variable:** `SalePrice` (Continuous monetary valuation in USD).

---

## 4. Technologies Used
* **Programming Language:** Python 3.11+
* **Data Processing & Analysis:** pandas, numpy
* **Machine Learning:** scikit-learn (`LinearRegression`, `Ridge`, `Lasso`, `ColumnTransformer`, `Pipeline`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `train_test_split`, `metrics`)
* **Visualization:** matplotlib, seaborn
* **Interactive Notebook:** Jupyter Notebook (`ipynb`)

---

## 5. Feature Selection
The modeling pipeline uses 18 carefully selected predictor variables based on correlation strength and real estate domain significance:

### Numerical Features (13)
* `OverallQual`: Overall material and finish quality rating (1–10).
* `GrLivArea`: Above grade (ground) living area square feet.
* `GarageCars`: Size of garage in car capacity.
* `GarageArea`: Size of garage in square feet.
* `TotalBsmtSF`: Total square feet of basement area.
* `1stFlrSF`: First Floor square feet.
* `FullBath`: Full bathrooms above grade.
* `TotRmsAbvGrd`: Total rooms above grade (excluding bathrooms).
* `YearBuilt`: Original construction date.
* `YearRemodAdd`: Remodel date.
* `Fireplaces`: Number of fireplaces.
* `LotArea`: Lot size in square feet.
* `LotFrontage`: Linear feet of street connected to property.

### Categorical Features (5)
* `Neighborhood`: Physical location within Ames city limits.
* `ExterQual`: Exterior material quality evaluation.
* `KitchenQual`: Kitchen quality evaluation.
* `BsmtQual`: Basement height and structural quality evaluation.
* `MSZoning`: General zoning classification.

---

## 6. Preprocessing & Modeling Pipeline
To prevent **data leakage**, all transformation steps are encapsulated in a Scikit-Learn `ColumnTransformer` pipeline fitted strictly on the training set (`X_train`):
1. **Numerical Pipeline:** `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
2. **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
3. **Train/Test Split:** 80% Training (1,168 samples) / 20% Testing (292 samples) with `random_state=42`.

---

## 7. Model Performance & Evaluation

| Model | MSE | RMSE ($) | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **Linear Regression (Primary)** | **1,072,759,628** | **$32,753.01** | **0.8601** |
| **Lasso Regression (Alpha=100)** | 1,091,482,279 | $33,037.59 | 0.8577 |
| **Ridge Regression (Alpha=10)** | 1,106,747,208 | $33,267.81 | 0.8557 |

* **Best Performing Model:** **Linear Regression** achieved an **$R^2$ Score of 0.8601** (explaining **86.01%** of price variance) with a **Root Mean Squared Error (RMSE) of $32,753.01**.

---

## 8. Output Charts & Visualizations
All charts are saved in high resolution under `outputs/charts/`:
* `outputs/charts/saleprice_distribution.png`: Histogram & KDE of house sale prices showing right skewness.
* `outputs/charts/correlation_heatmap.png`: Pearson correlation matrix heatmap for top numerical variables.
* `outputs/charts/top_features_correlation.png`: Bar plot of top numerical predictors correlated with `SalePrice`.
* `outputs/charts/actual_vs_predicted.png`: Scatter plot comparing actual prices vs model predictions with $y = x$ reference line.
* `outputs/charts/residual_plot.png`: Residual diagnostics scatter plot centered around zero.
* `outputs/charts/coefficient_importance.png`: Bar plot illustrating top positive and negative regression coefficients.

---

## 9. Key Insights
1. **Overall Quality (`OverallQual`) is Paramount:** `OverallQual` exhibits the strongest single correlation (+0.791) with sale price, adding >$20,000 per quality grade increment.
2. **Living Area Size (`GrLivArea`):** Above-ground living area (+0.709 correlation) is the second most dominant pricing factor.
3. **Garage Size & Capacity:** Houses with 2+ car garages command substantial premiums over non-garage properties.
4. **Neighborhood Valuation Premium:** Top-tier neighborhoods (such as *Northridge Heights*) add substantial baseline value.
5. **Basement & Age Impact:** Total basement square footage (`TotalBsmtSF`) and newer construction date (`YearBuilt`) protect property resale values.

---

## 10. Real-World Applications
* **Automated Valuation Models (AVMs):** Real estate portals (Zillow, Redfin) use regression baselines to estimate property values automatically.
* **Mortgage Underwriting:** Financial institutions utilize algorithmic valuations to assess property collateral risk.
* **Property Renovation ROI:** Homeowners and real estate investors can prioritize high-ROI structural improvements.

---

## 11. Model Limitations
* **Linear Relationship Assumption:** Linear regression assumes linear feature-target relationships, which may slightly underestimate extreme high-end luxury prices.
* **Heteroscedasticity at High Prices:** Residual variance increases for properties valued over $400,000.
* **Multicollinearity:** Structural features like `GarageCars` and `GarageArea` share high inter-correlation.

---

## 12. Project Structure
```text
Task5_House_Price_Prediction/
│
├── data/
│   ├── raw/
│   │   └── train.csv
│   └── processed/
│       └── house_prices_processed.csv
│
├── notebooks/
│   └── Task5_House_Price_Prediction.ipynb
│
├── outputs/
│   ├── charts/
│   │   ├── saleprice_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── top_features_correlation.png
│   │   ├── actual_vs_predicted.png
│   │   ├── residual_plot.png
│   │   └── coefficient_importance.png
│   │
│   └── reports/
│       ├── model_metrics.csv
│       ├── feature_coefficients.csv
│       └── missing_values_summary.csv
│
├── run_house_price_pipeline.py
├── build_task5_notebook.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 13. How to Run
1. Navigate to the project root:
   ```bash
   cd Task5_House_Price_Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the complete pipeline script:
   ```bash
   python run_house_price_pipeline.py
   ```
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/Task5_House_Price_Prediction.ipynb
   ```
