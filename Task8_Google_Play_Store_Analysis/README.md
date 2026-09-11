# OIBSIP Task 8 – Google Play Store Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![TextBlob](https://img.shields.io/badge/TextBlob-0.17%2B-008080.svg)](https://textblob.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Exploratory Data Analysis (EDA) and Sentiment Analysis project on the mobile app market developed for the **Oasis InfoByte Data Analytics Internship Program (OIBSIP)** – Task 8 (Level 2 Task 4).

---

## 1. Project Overview
The Google Play Store represents one of the world's largest digital software marketplaces. Understanding app store dynamics—such as category popularity, user ratings, install volume, monetization models, and customer sentiment—is essential for app developers, product managers, and digital marketers. This project provides a comprehensive data analysis of 10,000+ Android applications and 64,000+ user reviews.

---

## 2. Project Objective
- **Data Quality & Integrity:** Inspect, clean, and standardize raw app metadata and user review datasets.
- **Categorical & Market Profiling:** Identify top app categories by catalog volume, average user rating, review volume, and total downloads.
- **Monetization Analysis:** Compare Free vs. Paid business models across adoption rates, user satisfaction, and proxy revenue estimations.
- **User Sentiment Analysis:** Analyze user review sentiment polarity and subjectivity using both provided labels and independent `TextBlob` NLP sentiment scoring.
- **Correlation Discovery:** Uncover statistical relationships between app metrics (Rating, Reviews, Installs, Price, Size).
- **Strategic Recommendations:** Formulate actionable, data-driven recommendations to guide app launch, pricing, and category placement strategies.

---

## 3. Dataset Description & Attribution
- **Dataset Name:** Kaggle Google Play Store Apps & User Reviews Dataset.
- **Apps Dataset (`googleplaystore.csv`):** 10,841 raw rows (9,659 cleaned unique app records).
- **User Reviews Dataset (`googleplaystore_user_reviews.csv`):** 64,295 raw rows (29,692 cleaned review records).
- **Source Attribution:** [Kaggle - Google Play Store Apps Dataset by Lava18](https://www.kaggle.com/datasets/lava18/google-play-store-apps).

### Dataset Summary Table:
| Dataset | Raw Record Count | Cleaned Record Count | Removed Count | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Apps Metadata** | 10,841 | **9,659** | 1,182 | App metadata (Rating, Installs, Price, Size, Category) |
| **User Reviews** | 64,295 | **29,692** | 34,603 | User review text with sentiment polarity & subjectivity |

---

## 4. Technologies Used
- **Programming Language:** Python 3.10+
- **Data Processing:** `pandas`, `numpy`
- **Natural Language Processing:** `textblob`
- **Machine Learning & Stats:** `scikit-learn`
- **Visualization:** `matplotlib`, `seaborn`
- **Development Environment:** Jupyter Notebook, VS Code

---

## 5. Project Structure

```text
Task8_Google_Play_Store_Analysis/
│
├── data/
│   ├── raw/
│   │   ├── googleplaystore.csv                # Raw Kaggle Apps metadata dataset
│   │   └── googleplaystore_user_reviews.csv   # Raw Kaggle User Reviews dataset
│   └── processed/
│       ├── googleplaystore_cleaned.csv        # Cleaned unique apps export
│       └── googleplaystore_reviews_cleaned.csv# Cleaned user reviews export
│
├── notebooks/
│   └── Task8_Google_Play_Store_Analysis.ipynb# Primary executed Jupyter Notebook
│
├── outputs/
│   ├── charts/                                # 15 High-resolution generated plot images
│   │   ├── app_size_distribution.png
│   │   ├── apps_by_category.png
│   │   ├── avg_rating_by_category.png
│   │   ├── avg_sentiment_by_category.png
│   │   ├── content_rating_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── free_vs_paid_count.png
│   │   ├── installs_distribution.png
│   │   ├── paid_app_price_distribution.png
│   │   ├── rating_distribution.png
│   │   ├── reviews_vs_installs.png
│   │   ├── sentiment_distribution.png
│   │   ├── sentiment_polarity_distribution.png
│   │   ├── top_10_installed_apps.png
│   │   └── top_10_reviewed_apps.png
│   └── reports/                               # 7 Generated CSV reports
│       ├── category_summary.csv
│       ├── correlation_matrix.csv
│       ├── free_vs_paid_summary.csv
│       ├── key_insights.csv
│       ├── sentiment_summary.csv
│       ├── top_apps_by_installs.csv
│       └── top_apps_by_reviews.csv
│
├── build_task8_notebook.py                    # Programmatic notebook builder
├── download_google_play_data.py               # Automated dataset downloader
├── run_google_play_analysis_pipeline.py       # End-to-end Python execution pipeline
├── .gitignore                                 # Git ignore rules
├── LICENSE                                    # MIT License
├── README.md                                  # Project documentation
└── requirements.txt                           # Dependency specifications
```

---

## 6. Data Cleaning & Preparation
- **Malformed Row Removal:** Removed Row 10472 where a missing Category shifted columns.
- **Deduplication:** Retained the record with the maximum review count for duplicate app names.
- **Numeric Conversions:**
  - `Reviews`: Parsed string count to `int64`.
  - `Installs`: Removed `+` and `,`, cast to `int64`.
  - `Price`: Stripped `$` prefix and cast to `float64`.
  - `Size`: Parsed Megabyte (`M`) and Kilobyte (`k`) strings to float MB (`Size_MB`).
  - `Last Updated`: Converted string dates to `datetime64[ns]`.
- **Imputation:** Imputed missing `Rating` values using Category median ratings.
- **Reviews Cleaning:** Dropped empty reviews and duplicate entries; calculated independent `TextBlob` polarity and subjectivity scores.

---

## 7. Monetization & Revenue Proxy Estimation

| App Type | App Count | Percentage (%) | Average Rating | Average Installs | Average Price ($) | Total Estimated Revenue Proxy ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Free** | **8,905** | **92.19%** | 4.19 | 16,689,286 | $0.00 | $0.00 |
| **Paid** | **754** | **7.81%** | **4.27** | 75,227 | **$13.92** | **$377,258,409.73** |

### Real-World Revenue Proxy Limitations:
1. **Cumulative Tiered Installs:** Install counts represent bracketed lower bounds (e.g. $1,000,000+$).
2. **In-App Purchases (IAP):** Microtransactions and subscriptions in Free apps are unreflected in upfront price.
3. **Regional Pricing & Refunds:** Price discounts, local currency conversions, and platform fees are omitted.
4. **Proxy Note:** Estimated Revenue ($\text{Price} \times \text{Installs}$) is an exploratory proxy metric and does not represent actual company accounting revenue.

---

## 8. User Review Sentiment Analysis

| Sentiment Category | Review Count | Percentage (%) | Mean Polarity Score | Mean Subjectivity Score |
| :--- | :---: | :---: | :---: | :---: |
| **Positive** | **19,036** | **64.11%** | +0.3601 | 0.5471 |
| **Negative** | **8,241** | **27.75%** | -0.2604 | 0.5398 |
| **Neutral** | **2,415** | **8.13%** | 0.0000 | 0.1264 |

* **Overall Sentiment Polarity:** Mean polarity across all user reviews is **+0.1823**, indicating an overall positive user sentiment bias across Play Store reviews.

---

## 9. Visualizations Generated

All generated visualizations are saved in high resolution under `outputs/charts/`:

1. **Apps by Category (`apps_by_category.png`):** Horizontal bar chart of top 15 categories by app catalog count.
2. **Average Rating by Category (`avg_rating_by_category.png`):** Horizontal bar chart of top categories by average rating.
3. **Top 10 Installed Apps (`top_10_installed_apps.png`):** Bar chart of the 10 most installed apps.
4. **Top 10 Reviewed Apps (`top_10_reviewed_apps.png`):** Bar chart of the 10 most reviewed apps.
5. **Rating Distribution (`rating_distribution.png`):** Histogram & KDE plot showing left-skewed rating distribution (Mean: 4.19).
6. **Installs Distribution (`installs_distribution.png`):** Log-scaled bar chart of install count brackets.
7. **Free vs. Paid App Count (`free_vs_paid_count.png`):** Bar chart comparing Free (92.2%) vs. Paid (7.8%) apps.
8. **Paid App Price Distribution (`paid_app_price_distribution.png`):** Histogram of paid app prices under $50.
9. **Content Rating Distribution (`content_rating_distribution.png`):** Bar chart showing app counts across age content ratings.
10. **App Size Distribution (`app_size_distribution.png`):** Distribution of app file sizes in Megabytes.
11. **Sentiment Distribution (`sentiment_distribution.png`):** Bar chart of Positive (64.1%), Negative (27.8%), and Neutral (8.1%) reviews.
12. **Sentiment Polarity Distribution (`sentiment_polarity_distribution.png`):** Distribution of TextBlob polarity scores.
13. **Average Sentiment by Category (`avg_sentiment_by_category.png`):** Top categories sorted by average review sentiment polarity.
14. **Reviews vs. Installs Scatter Plot (`reviews_vs_installs.png`):** Log-log scatter plot illustrating strong co-movement ($r = 0.643$).
15. **Correlation Heatmap (`correlation_heatmap.png`):** Pearson correlation matrix heatmap for numeric features.

---

## 10. Key Insights
1. **Catalog Dominance:** `Family` (1,943 apps) and `Game` (1,121 apps) represent 31.7% of the Play Store catalog.
2. **Download Volume Leader:** `Game` category generated the highest total downloads (**35.08 Billion+ installs**).
3. **Free Model Market Share:** Free apps constitute **92.19% of apps** and drive **99.2% of total downloads**.
4. **Reviews Drive Downloads:** Reviews and Installs display a strong positive correlation (**$r = 0.6432$**).
5. **Positive Review Bias:** 64.11% of user reviews express Positive sentiment, with mean polarity at +0.1823.
6. **Rating Skewness:** Average app rating is **4.19 / 5.0**, with ratings displaying a distinct left-skewed distribution.

---

## 11. Business Recommendations
1. **Adopt Freemium Monetization:** Offer a Free download with In-App Purchases (IAP). Free apps achieve $100\times$ higher download volume than paid apps.
2. **In-App Review Milestones:** Trigger review prompts immediately after positive user achievements (e.g. level completion). Higher review volume directly correlates with install ranking ($r = 0.643$).
3. **App Size Optimization (< 50 MB):** Keep initial download sizes under 50 MB to minimize user drop-off on mobile data connections.

---

## 12. How to Run the Project

### Setup Steps
1. Navigate to project root:
   ```bash
   cd Task8_Google_Play_Store_Analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dataset downloader:
   ```bash
   python download_google_play_data.py
   ```
4. Run the automated Python ML pipeline:
   ```bash
   python run_google_play_analysis_pipeline.py
   ```
5. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/Task8_Google_Play_Store_Analysis.ipynb
   ```

---

## 13. Requirements & Licensing
- Package dependencies specified in `requirements.txt`.
- Code source and documentation released under the [MIT License](LICENSE).
- Dataset attribution belongs to Kaggle / Lava18.

---

## 14. Conclusion
This project demonstrates a thorough data analysis of the Google Play Store app market. By cleaning app metadata and user reviews, analyzing monetization models, evaluating sentiment polarity, and discovering correlation patterns, we provided actionable strategic insights for mobile app launch and growth.
