# OIBSIP Task 1 – Retail Sales EDA

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Exploratory Data Analysis (EDA) project on retail sales transactions developed for the **Oasis InfoByte Data Analytics Internship Program (OIBSIP)** - Task 1.

---

## 1. Project Overview
In retail analytics, understanding transactional patterns, customer demographics, and product category performance is crucial for strategic decision-making. This project presents a thorough, reproducible exploratory data analysis of 1,000 retail transactions recorded between **January 2023 and January 2024**.

---

## 2. Project Objective
- **Data Integrity & Validation:** Inspect, clean, and validate transactional records for consistent formatting and data types.
- **Statistical Profiling:** Calculate fundamental measures of central tendency and variability (Mean, Median, Mode, Std Dev, Min, Max).
- **Time Series Analysis:** Trace monthly and quarterly sales revenue trends to highlight seasonality and peak revenue periods.
- **Demographic Segmentation:** Analyze customer purchasing behavior across age brackets (18–25, 26–35, 36–45, 46–55, 56+) and gender distributions.
- **Product Category Performance:** Compare sales volume (units sold) and gross revenue across product categories (`Electronics`, `Clothing`, `Beauty`).
- **Correlation Discovery:** Identify relationships between unit price, order quantity, customer age, and total purchase value.
- **Strategic Business Guidance:** Formulate actionable, data-backed recommendations to optimize retail operations.

---

## 3. Dataset Description & Attribution

The dataset used in this project is the authentic, publicly available **Retail Sales Dataset** sourced from Kaggle.

- **Total Records:** 1,000 transaction rows
- **Total Fields:** 9 attributes
- **Date Range:** January 1, 2023 – January 1, 2024
- **Source Attribution:** [Kaggle - Retail Sales Dataset by Mohammad Mohammad](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset).

### Field Descriptions:
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Transaction ID` | `int64` | Unique numeric identifier for each transaction |
| `Date` | `datetime64` | Date when the purchase occurred (YYYY-MM-DD) |
| `Customer ID` | `object` | Unique customer alphanumeric code (e.g. CUST001) |
| `Gender` | `object` | Customer gender (`Male` / `Female`) |
| `Age` | `int64` | Customer age in years (Range: 18 – 64) |
| `Product Category` | `object` | Retail category (`Clothing`, `Electronics`, `Beauty`) |
| `Quantity` | `int64` | Number of items purchased in transaction (1 – 4) |
| `Price per Unit` | `int64` | Price per item in USD ($25 – $500) |
| `Total Amount` | `int64` | Gross revenue per order ($25 – $2,000) |

---

## 4. Technologies Used
- **Programming Language:** Python 3.10+
- **Data Processing:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Development Environment:** Jupyter Notebook, VS Code
- **Package Management:** `pip`

---

## 5. Project Structure

```
Task1_Retail_Sales_EDA/
│
├── data/
│   └── retail_sales.csv            # Original downloaded transaction dataset
│
├── notebooks/
│   └── Task1_Retail_Sales_EDA.ipynb  # Primary executed Jupyter Notebook
│
├── outputs/
│   ├── charts/                     # High-resolution generated plot images
│   │   ├── age_distribution.png
│   │   ├── age_group_analysis.png
│   │   ├── category_sales_by_gender.png
│   │   ├── correlation_heatmap.png
│   │   ├── gender_distribution.png
│   │   ├── monthly_sales_trend.png
│   │   ├── quarterly_sales_trend.png
│   │   ├── revenue_by_category.png
│   │   ├── sales_by_dayofweek.png
│   │   └── top_products_revenue.png
│   └── cleaned_data/
│       └── retail_sales_cleaned.csv # Standardized cleaned CSV export
│
├── .gitignore                      # Git ignore file for Python/Jupyter
├── LICENSE                         # MIT License for project source code
├── README.md                       # Complete project documentation
└── requirements.txt                # Python package dependency specifications
```

---

## 6. Data Cleaning & Preparation
- **Datetime Parsing:** Transformed string-formatted `Date` column into pandas `datetime64[ns]` format.
- **Temporal Feature Engineering:** Derived `Year`, `Month`, `YearMonth` (monthly period), `Quarter` (quarterly period), and `DayOfWeek`.
- **Demographic Binning:** Constructed 5 age categories (`18–25`, `26–35`, `36–45`, `46–55`, `56+`).
- **Validation:** Confirmed 0 null values, 0 duplicate records, and verified mathematical integrity (`Total Amount = Quantity * Price per Unit`).
- **Export:** Saved clean dataset to `outputs/cleaned_data/retail_sales_cleaned.csv`.

---

## 7. Exploratory Data Analysis & Descriptive Statistics

### Summary Statistics Table:
| Metric | Customer Age | Order Quantity | Unit Price ($) | Total Amount ($) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean** | 41.39 | 2.25 | $179.89 | $456.00 |
| **Median** | 42.00 | 2.00 | $50.00 | $135.00 |
| **Mode** | 43.00 | 1.00 | $50.00 | $50.00 |
| **Std Dev** | 13.68 | 1.12 | $189.68 | $559.99 |
| **Minimum** | 18.00 | 1.00 | $25.00 | $25.00 |
| **Maximum** | 64.00 | 4.00 | $500.00 | $2,000.00 |

---

## 8. Visualizations Generated

All generated visualizations are saved in high-resolution DPI inside `outputs/charts/`:

1. **Monthly Sales Revenue Trend (`monthly_sales_trend.png`):** Shows revenue spikes in May ($53,150) and October ($46,580).
2. **Quarterly Revenue Breakdown (`quarterly_sales_trend.png`):** Demonstrates Q2 2023 leadership ($131,340) followed by Q4 ($124,580).
3. **Customer Age Distribution (`age_distribution.png`):** Histogram overlay with KDE density curve showing uniform customer age distribution across 18–64.
4. **Revenue by Age Group (`age_group_analysis.png`):** Identifies 46–55 ($99,945) and 26–35 ($99,575) as top revenue-generating age brackets.
5. **Gender Distribution (`gender_distribution.png`):** Donut chart showing 51.0% female and 49.0% male buyer participation.
6. **Total Revenue by Product Category (`revenue_by_category.png`):** Bar chart comparing Electronics ($156,905), Clothing ($155,580), and Beauty ($143,515).
7. **Category Volume vs Revenue (`top_products_revenue.png`):** Dual-axis chart comparing total units sold vs revenue across categories.
8. **Correlation Matrix Heatmap (`correlation_heatmap.png`):** Visualizes feature inter-correlations (Unit Price vs Total Amount: **r = 0.852**).
9. **Category Sales by Gender (`category_sales_by_gender.png`):** Grouped bar chart highlighting male preference for Electronics and female preference for Beauty & Clothing.
10. **Sales Revenue by Day of Week (`sales_by_dayofweek.png`):** Highlights mid-week purchasing spikes on Thursdays ($67,150) and Tuesdays ($66,415).

---

## 9. Key Insights
1. **Strong Price-Revenue Correlation:** `Price per Unit` is the main driver of transaction revenue ($r = 0.852$). Higher unit prices (e.g. $300 - $500) drive order size more than purchase volume.
2. **Category Balance:** Electronics leads gross revenue ($156.9k), but Clothing leads in total units sold (894 units).
3. **Targeted Gender Dynamics:** Females lead spending in Clothing ($85.5k) and Beauty ($74.8k), whereas Males lead in Electronics ($84.4k).
4. **Age Independence:** Customer age has zero meaningful correlation with order value ($r = -0.060$), indicating that marketing should focus on product preferences rather than age-based restrictions.

---

## 10. Business Recommendations

1. **Gender-Targeted Marketing & Cross-Category Bundling:**
   - Run targeted advertisements showcasing **Electronics** to male demographics and **Clothing + Beauty** bundles to female demographics.
2. **Premium Pricing Tier & Checkout Add-Ons:**
   - Capitalize on the $r = 0.852$ unit-price correlation by offering premium product tiers and post-purchase add-on bundles to raise Average Order Value (AOV).
3. **Mid-Week Promotional Scheduling:**
   - Schedule email promotional campaigns and inventory replenishment for **Tuesdays and Thursdays** to align with peak mid-week purchasing days.

---

## 11. How to Run the Project

### Setup Steps
1. Clone or open the project folder:
   ```bash
   cd Task1_Retail_Sales_EDA
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
   jupyter notebook notebooks/Task1_Retail_Sales_EDA.ipynb
   ```

---

## 12. Requirements & Licensing
- Dependencies are specified in `requirements.txt`.
- The project source code and notebook documentation are released under the [MIT License](LICENSE).
- The raw dataset is sourced from Kaggle. Users should refer to the original dataset source link for specific dataset licensing terms.

---

## 13. Conclusion
This project provides a clean, automated, and comprehensive exploratory data analysis of retail sales data. All visualizations, numerical calculations, and insights are derived directly from real transaction data and saved cleanly within the project directory structure, making it fully ready for internship submission and recruiter evaluation.
