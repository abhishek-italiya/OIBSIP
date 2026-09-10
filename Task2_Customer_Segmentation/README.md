# OIBSIP Task 2 – Customer Segmentation Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and Behavioral Customer Analytics project developed for the **Oasis InfoByte Data Analytics Internship Program (OIBSIP)** - Task 2.

---

## 1. Project Overview
Customer Segmentation is an essential strategy in retail analytics that partitions a heterogeneous customer base into distinct behavioral groups. By categorizing customers according to purchasing patterns, business teams can deploy targeted marketing strategies, increase customer retention, and maximize lifetime customer value.

This project performs unsupervised machine learning (**K-Means Clustering**) based on the **RFM (Recency, Frequency, Monetary)** framework using a real-world dataset of **541,909 transnational e-commerce transactions**.

---

## 2. Project Objective
- **Data Preprocessing & Quality Assurance:** Clean transactional records by filtering null customer identifiers, returns, and invalid prices.
- **RFM Feature Engineering:** Compute customer-level Recency ($R$), Frequency ($F$), and Monetary ($M$) behavioral metrics.
- **Feature Transformation & Scaling:** Normalize right-skewed RFM distributions using log transformation (`log1p`) and `StandardScaler`.
- **Optimal Cluster Determination:** Implement the **Elbow Method** and **Silhouette Score Analysis** to objectively determine the optimal cluster count ($K = 4$).
- **K-Means Clustering:** Train unsupervised K-Means model to segment **4,338 unique customers** into 4 distinct behavioral personas.
- **Strategic Marketing Alignment:** Formulate tailored marketing campaigns and loyalty programs for each customer segment.

---

## 3. Dataset Description
The analysis utilizes the official **Online Retail Dataset**, a real-world transnational dataset containing all transactions occurring between December 1, 2010, and December 9, 2011, for a UK-based online retail store specializing in giftware.

- **Raw Records:** 541,909 transactions
- **Cleaned Dataset:** 392,692 valid transactions across **4,338 unique customers**
- **Date Range:** December 1, 2010 – December 9, 2011
- **Total Portfolio Revenue:** $8,887,208.89

### Dataset Attributes:
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `InvoiceNo` | `object` | 6-digit integral transaction code (Prefix 'C' indicates cancellation) |
| `StockCode` | `object` | Distinct product item code |
| `Description` | `object` | Product item name |
| `Quantity` | `int64` | Number of units purchased per order |
| `InvoiceDate` | `datetime64` | Transaction date and time |
| `UnitPrice` | `float64` | Price per unit in Sterling (£ / $) |
| `CustomerID` | `float64 / int` | Unique 5-digit numeric customer identifier |
| `Country` | `object` | Customer residence country |

---

## 4. Dataset Source & Attribution
- **Source Repository:** [UCI Machine Learning Repository - Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)
- **Dataset Citation:** Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository.
- **Original File Format:** `.xlsx` converted directly to `data/online_retail.csv`.

---

## 5. Technologies Used
- **Programming Language:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning & Preprocessing:** `scikit-learn` (`KMeans`, `StandardScaler`, `silhouette_score`)
- **Data Visualization:** `matplotlib`, `seaborn`
- **Interactive Environment:** Jupyter Notebook

---

## 6. RFM Analysis Methodology

RFM values are derived per customer using a reference snapshot date of **December 10, 2011** (`max(InvoiceDate) + 1 day`):

1. **Recency ($R$):** $Days = \text{Snapshot Date} - \text{Max(InvoiceDate)}_{\text{customer}}$
2. **Frequency ($F$):** $Orders = \text{Count of Unique InvoiceNo}_{\text{customer}}$
3. **Monetary ($M$):** $Spend = \sum (\text{Quantity} \times \text{UnitPrice})_{\text{customer}}$

---

## 7. Data Cleaning Pipeline
- **Missing Customer IDs:** Dropped 135,080 unassigned transactions (guest checkouts).
- **Cancellations & Negative Quantities:** Filtered out 10,624 cancelled orders (`InvoiceNo` starting with 'C' or `Quantity <= 0`).
- **Invalid Prices:** Removed 2 zero/negative price items.
- **Deduplication:** Dropped 5,268 exact duplicate transaction rows.

---

## 8. K-Means Clustering & Standardization
1. **Log Transformation:** Applied `np.log1p()` to Recency, Frequency, and Monetary to resolve heavy right-skewness.
2. **Feature Scaling:** Standardized log-transformed features using `StandardScaler()` to achieve mean $\mu = 0$ and standard deviation $\sigma = 1$.
3. **Clustering:** Trained `KMeans(n_clusters=4, random_state=42)`.

---

## 9. Elbow Method & Silhouette Evaluation

| Number of Clusters ($K$) | Inertia (WCSS) | Silhouette Score | Evaluation Result |
| :---: | :---: | :---: | :--- |
| **K = 2** | 5,422.38 | 0.3952 | Broad separation, under-segmented |
| **K = 3** | 4,463.12 | 0.3060 | Sub-optimal cluster cohesion |
| **K = 4** | **3,939.05** | **0.3375** | **Optimal Elbow & Strong Silhouette Separation** |
| **K = 5** | 3,506.74 | 0.2905 | Diminishing returns in inertia |
| **K = 6** | 3,172.91 | 0.2764 | Over-segmentation |

---

## 10. Cluster Profiling Table

| Cluster | Segment Name | Customer Count | Customer % | Avg Recency (Days) | Avg Frequency (Orders) | Avg Spend ($) | Total Revenue ($) | Revenue % |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **Champions / High-Value** | 713 | 16.44% | 12.17 | 13.75 | $8,088.02 | $5,766,757.07 | **64.89%** |
| **1** | **At-Risk / Hibernating** | 1,622 | 37.39% | 181.51 | 1.32 | $340.99 | $553,099.77 | **6.22%** |
| **2** | **Recent / New Buyers** | 837 | 19.29% | 17.70 | 2.19 | $557.32 | $466,479.03 | **5.25%** |
| **3** | **Loyal / Regular Customers** | 1,166 | 26.88% | 71.64 | 4.08 | $1,801.78 | $2,100,873.02 | **23.64%** |

---

## 11. Visualizations Generated

All generated visualizations are saved in `outputs/charts/`:

1. **`rfm_distributions.png`:** Histograms showing raw Recency, Frequency, and Monetary feature distributions.
2. **`elbow_method.png`:** Dual-axis line chart illustrating Inertia vs Silhouette Score across $K=2..10$.
3. **`recency_vs_monetary_clusters.png`:** Log-scale scatter plot demonstrating clear separation between Champions and At-Risk groups.
4. **`frequency_vs_monetary_clusters.png`:** Log-scale scatter plot illustrating positive linear trajectory of order frequency vs spend.
5. **`customers_per_cluster.png`:** Bar chart showing customer volume across segments.
6. **`revenue_by_cluster.png`:** Bar chart displaying gross revenue contribution per segment.

---

## 12. Key Insights
- **Pareto Principle Validated:** **16.4%** of customers (Champions) account for **64.9% ($5.77M)** of gross company revenue.
- **High At-Risk Volume:** **37.4%** of customers (1,622 buyers) haven't purchased in over 6 months, representing dormant accounts.
- **Loyal Backbone:** **26.9%** of customers are consistent repeat buyers generating **$2.10M** in stable revenue.

---

## 13. Marketing Recommendations

### 1. Champions / High-Value Segment
- **Action:** Launch an exclusive VIP Concierge tier with dedicated support, early access to new releases, and personalized thank-you gifts.
- **Goal:** Maintain 100% retention and encourage brand ambassadorship.

### 2. Loyal / Regular Segment
- **Action:** Implement personalized cross-selling, threshold-based reward coupons, and automated product replenishment reminders.
- **Goal:** Elevate average order monetary spend into the Champion tier.

### 3. Recent / New Buyers Segment
- **Action:** Deploy a 3-step automated welcome email onboarding flow with a 14-day 10% discount incentive for their 2nd purchase.
- **Goal:** Convert single-purchase buyers into repeat customers.

### 4. At-Risk / Hibernating Segment
- **Action:** Deploy win-back email campaigns offering a steep incentive (e.g. 20% off or free gift) and exit surveys.
- **Goal:** Reactivate dormant buyers before permanent churn.

---

## 14. Project Structure

```
Task2_Customer_Segmentation/
│
├── data/
│   └── online_retail.csv               # Downloaded UCI Online Retail dataset
│
├── notebooks/
│   └── Task2_Customer_Segmentation.ipynb # Executed Jupyter Notebook
│
├── outputs/
│   ├── charts/                          # High-resolution generated charts
│   │   ├── customers_per_cluster.png
│   │   ├── elbow_method.png
│   │   ├── frequency_vs_monetary_clusters.png
│   │   ├── recency_vs_monetary_clusters.png
│   │   ├── revenue_by_cluster.png
│   │   └── rfm_distributions.png
│   └── processed_data/
│       └── customer_segments.csv       # Exported RFM dataset with cluster labels
│
├── .gitignore                           # Git ignore specification
├── LICENSE                              # MIT License
├── README.md                            # Comprehensive project documentation
└── requirements.txt                     # Package dependencies
```

---

## 15. How to Run the Project

### Prerequisites
- Python 3.10+ installed

### Setup & Execution
1. Open terminal and navigate to the project directory:
   ```bash
   cd Task2_Customer_Segmentation
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
   jupyter notebook notebooks/Task2_Customer_Segmentation.ipynb
   ```

---

## 16. Requirements

Dependencies are specified in `requirements.txt`:
```text
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
openpyxl>=3.1.0
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.20.0
```

---

## 17. Conclusion
This project provides a robust, machine-learning-driven customer segmentation framework. By applying RFM feature engineering, StandardScaler, and K-Means clustering ($K=4$), the analysis uncovered critical revenue concentration patterns and actionable customer personas, offering business leaders clear strategic guidance for customer retention and revenue growth.

---

## 18. License
- **Project Source Code & Documentation:** Released under the [MIT License](LICENSE).
- **Dataset License:** Sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail). Users should refer to original UCI terms for dataset usage.
