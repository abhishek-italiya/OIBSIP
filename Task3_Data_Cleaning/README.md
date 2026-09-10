# OIBSIP Task 3 – Data Cleaning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Data Engineering and Quality Assurance project developed for the **Oasis InfoByte Data Analytics Internship Program (OIBSIP)** - Task 3.

---

## 1. Project Overview
Raw real-world datasets collected from operational e-commerce systems, user forms, or legacy databases are inherently messy. Uncleaned data containing missing values, duplicate entries, formatting inconsistencies, incorrect data types, and extreme numeric outliers leads to flawed analytics and model failures (**Garbage In, Garbage Out**).

This project demonstrates an industry-standard, reproducible data cleaning pipeline on a deliberately messy customer transaction dataset containing **1,035 raw records** across 9 attributes.

---

## 2. Project Objective
- **Audit Data Quality:** Perform an initial data quality audit and document all flaws in `outputs/reports/data_quality_before.csv`.
- **Duplicate Purging:** Identify and remove exact duplicate transaction records.
- **Categorical & Text Standardization:** Standardize inconsistent casing, whitespace, and category typos across string columns.
- **Data Type Parsing:** Convert monetary string columns to `float64` and parse mixed text date formats to `datetime64[ns]`.
- **Statistical Imputation:** Treat missing values using domain-appropriate strategies (Mode for categories, Median for numerical values).
- **Outlier Analysis & Dual-Column Retention:** Implement Interquartile Range (**IQR**) outlier detection and Winsorization (capping) into a dedicated column `Purchase_Amount_Capped`.
- **Validation & Export:** Export the 100% clean, analysis-ready dataset to `data/cleaned/cleaned_dataset.csv`.

---

## 3. Dataset Description
The dataset represents a customer transactions dataset designed to exhibit real-world data quality defects:

- **Raw Dataset Size:** 1,035 records, 9 columns
- **Cleaned Dataset Size:** 1,000 records, 10 columns
- **Total Duplicate Rows Purged:** 35 duplicate rows
- **Total Missing Values Resolved:** 568 missing values

### Attribute Definitions:
| Column Name | Raw Data Type | Cleaned Data Type | Description |
| :--- | :--- | :--- | :--- |
| `Transaction_ID` | `object` | `object` | Unique transaction identifier (`TXN-1000`) |
| `Customer_Name` | `object` | `object` | Customer name (standardized to Title Case) |
| `Gender` | `object` | `object` | Gender (`Male` / `Female`) |
| `Join_Date` | `object` | `datetime64[ns]` | Customer sign-up date (parsed to ISO datetime) |
| `Customer_Age` | `float64` | `int64` | Customer age in years (Range: 18 – 70) |
| `Income_USD` | `object` | `float64` | Annual customer income in USD ($) |
| `Purchase_Amount` | `float64` | `float64` | Original/imputed purchase order amount in USD ($) |
| `Product_Category` | `object` | `object` | Retail category (`Electronics`, `Clothing`, `Beauty`, `Home & Kitchen`) |
| `Country` | `object` | `object` | Country of residence |
| `Purchase_Amount_Capped` | N/A | `float64` | IQR-winsorized purchase amount for parametric model stability |

---

## 4. Dataset Source & Attribution
- **Dataset Origin:** Public domain deliberately messy dataset benchmark for data wrangling tutorials and data quality evaluation.
- **Raw File Location:** `data/raw/original_dataset.csv`
- **Cleaned File Location:** `data/cleaned/cleaned_dataset.csv`

---

## 5. Data Quality Problems Found & Solved

| Problem Category | Raw Flaw Description | Cleaning Strategy Applied |
| :--- | :--- | :--- |
| **Duplicates** | 35 exact duplicate rows | Purged duplicates using `.drop_duplicates()` |
| **Missing Values** | 568 missing cells across 8 columns | Categorical mode & numerical median imputation |
| **Text Casing & Whitespace** | Extra whitespace and mixed case (` sarah CONNOR `) | `.str.strip().str.title()` |
| **Categorical Variations** | 8 gender variations (`Male`, `male`, `M`, `m`, `F`...) | Mapped to 2 standardized values (`Male`, `Female`) |
| **Category Typos** | Inconsistent category spelling (`cloths`, `ELECTRONICS`) | Standardized to 4 core categories |
| **Country Inconsistencies** | Mixed country codes (`USA`, `US`, `U.S.A.`, `UK`, `CAN`) | Standardized to full country names |
| **Monetary String Parsing** | `Income_USD` formatted as string (`$45,000.00`, `UNKNOWN`) | Cleaned symbols and parsed to `float64` |
| **Mixed Date Formats** | Dates formatted as `YYYY-MM-DD`, `MM/DD/YYYY`, text | Parsed via `pd.to_datetime(format='mixed')` |
| **Numeric Age Anomalies** | Invalid ages (`-5`, `150`, `200`) | Filtered invalid ranges and imputed median (42 yrs) |
| **Purchase Amount Outliers** | Negative values (`-$50.00`) and high spikes (`$15,000.00`) | Filtered negatives and Winsorized into `Purchase_Amount_Capped` |

---

## 6. Missing Value Handling Table

| Column Name | Missing Count (Raw) | Imputation Strategy | Justification |
| :--- | :---: | :--- | :--- |
| `Customer_Name` | 110 | String Imputation (`"Unknown Customer"`) | Preserves transaction record without inferring names |
| `Gender` | 51 | Mode Imputation (`"Male"`) | Replaces nulls with most frequent category |
| `Join_Date` | 89 | Forward & Backward Fill (`ffill` / `bfill`) | Maintains temporal continuity |
| `Customer_Age` | 46 | Median Imputation (**42 years**) | Median is robust to age outliers |
| `Income_USD` | 116 | Median Imputation (**$85,000.00**) | Median avoids distortion from income skewness |
| `Purchase_Amount` | 31 | Median Imputation (**$249.77**) | Median preserves central order value |
| `Product_Category` | 114 | Mode Imputation (`"Electronics"`) | Assigns most frequent retail product category |
| `Country` | 117 | Mode Imputation (`"United States"`) | Assigns primary customer market location |

---

## 7. Outlier Detection & Dual-Column Retention Rationale

For `Purchase_Amount`, the Interquartile Range (**IQR**) parameters are:
- **First Quartile ($Q_1$):** $185.87
- **Third Quartile ($Q_3$):** $379.41
- **Interquartile Range ($\text{IQR}$):** $193.54
- **Upper Bound ($Q_3 + 1.5 \times \text{IQR}$):** $670.36

### Dual-Column Strategy (`Purchase_Amount` vs `Purchase_Amount_Capped`):
1. **`Purchase_Amount` (Original / Imputed):** Preserves authentic historical customer spend values without destroying extreme transaction records.
2. **`Purchase_Amount_Capped` (Winsorized):** Contains purchase amounts capped at the upper IQR threshold (**$670.36**). Extreme transaction spikes (e.g. **$15,000.00**) are capped at **$670.36**.
3. **Purpose of Dual Retention:** Keeping both columns ensures complete transparency and data auditability. Downstream modeling algorithms (e.g. distance-based clustering or linear regression) benefit from `Purchase_Amount_Capped` to prevent variance distortion, while `Purchase_Amount` maintains true accounting totals.

---

## 8. Before vs After Summary Table

| Metric | Before Cleaning (Raw) | After Cleaning (Cleaned) |
| :--- | :---: | :---: |
| **Total Rows** | 1,035 | **1,000** |
| **Total Columns** | 9 | **10** |
| **Duplicate Rows** | 35 | **0** |
| **Total Missing Values** | 568 | **0** |
| **Gender Variations** | 8 variations | **2 categories ('Male', 'Female')** |
| **Product Category Variations** | 9 variations | **4 core categories** |
| **Country Name Variations** | 9 variations | **4 country names** |
| **Join_Date Data Type** | Object (mixed text strings) | **datetime64[ns]** |
| **Income_USD Data Type** | Object (string with `$`, `,`) | **Numeric float64** |
| **Age Anomalies** | Invalid values (-5, 150, 200) | **Imputed with Median Age (42 yrs)** |
| **Purchase Amount Outliers** | Extreme spikes ($15,000.00) | **Winsorized to IQR Upper Bound ($670.36)** |

---

## 9. Visualizations Generated

All generated plots are saved in `outputs/charts/`:

1. **`missing_values_before_after.png`:** Bar chart illustrating the complete elimination of null values across all columns.
2. **`category_standardization.png`:** Bar chart showing clean distribution across standardized product categories.
3. **`outliers_boxplots_before_after.png`:** Side-by-side boxplots comparing **"Purchase Amount (Before Outlier Treatment)"** (original purchase amounts) versus **"Purchase Amount (After Outlier Capping)"** (capped values at $670.36).

---

## 10. Project Structure

```
Task3_Data_Cleaning/
│
├── data/
│   ├── raw/
│   │   └── original_dataset.csv       # Original raw messy dataset (1,035 rows)
│   └── cleaned/
│       └── cleaned_dataset.csv        # 100% Cleaned, analysis-ready dataset (1,000 rows)
│
├── notebooks/
│   └── Task3_Data_Cleaning.ipynb      # Primary executed Jupyter Notebook
│
├── outputs/
│   ├── charts/                         # High-resolution generated charts
│   │   ├── category_standardization.png
│   │   ├── missing_values_before_after.png
│   │   └── outliers_boxplots_before_after.png
│   └── reports/                        # Audit & comparison report CSVs
│       ├── before_after_summary.csv
│       └── data_quality_before.csv
│
├── .gitignore                          # Git ignore rules for Python/Jupyter
├── LICENSE                             # MIT License
├── README.md                           # Comprehensive project documentation
└── requirements.txt                    # Package dependencies
```

---

## 11. How to Run the Project

### Setup & Execution
1. Open terminal and navigate to the project directory:
   ```bash
   cd Task3_Data_Cleaning
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/Task3_Data_Cleaning.ipynb
   ```

---

## 12. Key Learnings
- **Domain-Specific Imputation:** Mean imputation can skew right-skewed numerical data; median imputation provides robust central tendency.
- **Data Preservation via Dual Columns:** Retaining both uncapped and winsorized columns preserves historical accuracy while offering variance stability for machine learning models.
- **Regex & String Cleansing:** Standardizing casing and removing hidden leading/trailing spaces prevents duplicated category keys in grouping operations.

---

## 13. Conclusion
This project successfully executed a professional data cleaning pipeline. The raw dataset was systematically audited, cleaned, standardized, imputed, and validated. The final dataset `data/cleaned/cleaned_dataset.csv` contains 1,000 high-quality records ready for statistical analysis and machine learning model training.

---

## 14. License
- **Project Source Code & Documentation:** Released under the [MIT License](LICENSE).
- **Dataset License:** Public domain benchmark dataset for data wrangling tutorials.
