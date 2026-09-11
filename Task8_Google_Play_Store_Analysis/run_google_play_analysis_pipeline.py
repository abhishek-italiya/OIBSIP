import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Ensure required directories exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# Set visualization theme
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# 1. Load Raw Datasets
print("=== 1. LOADING RAW DATASETS ===")
apps_raw_path = "data/raw/googleplaystore.csv"
reviews_raw_path = "data/raw/googleplaystore_user_reviews.csv"

df_apps = pd.read_csv(apps_raw_path)
df_reviews = pd.read_csv(reviews_raw_path)

print(f"Apps Raw Shape    : {df_apps.shape[0]:,} rows x {df_apps.shape[1]} columns")
print(f"Reviews Raw Shape : {df_reviews.shape[0]:,} rows x {df_reviews.shape[1]} columns")

# 2. Data Cleaning: Apps Dataset
print("\n=== 2. CLEANING APPS DATASET ===")
# Remove malformed row (Row 10472 where Category is '1.9' due to missing Category value)
df_apps_clean = df_apps[df_apps['Category'] != '1.9'].copy()

# Remove duplicate apps, keeping the record with maximum Reviews count
df_apps_clean['Reviews_numeric'] = pd.to_numeric(df_apps_clean['Reviews'], errors='coerce')
df_apps_clean = df_apps_clean.sort_values(by='Reviews_numeric', ascending=False)
df_apps_clean = df_apps_clean.drop_duplicates(subset=['App'], keep='first')
df_apps_clean = df_apps_clean.drop(columns=['Reviews_numeric'])

# Clean Reviews: convert to integer
df_apps_clean['Reviews'] = pd.to_numeric(df_apps_clean['Reviews'], errors='coerce')

# Clean Installs: remove commas, '+' signs, convert to integer
df_apps_clean['Installs_clean'] = df_apps_clean['Installs'].astype(str).str.replace('+', '', regex=False).str.replace(',', '', regex=False)
df_apps_clean['Installs'] = pd.to_numeric(df_apps_clean['Installs_clean'], errors='coerce')
df_apps_clean = df_apps_clean.drop(columns=['Installs_clean'])

# Clean Price: remove '$', convert to float
df_apps_clean['Price_clean'] = df_apps_clean['Price'].astype(str).str.replace('$', '', regex=False)
df_apps_clean['Price'] = pd.to_numeric(df_apps_clean['Price_clean'], errors='coerce')
df_apps_clean = df_apps_clean.drop(columns=['Price_clean'])

# Clean Size: convert 'M' to MB, 'k' to MB (k/1000), 'Varies with device' to NaN
def clean_size(size_str):
    if pd.isna(size_str) or size_str == 'Varies with device':
        return np.nan
    size_str = str(size_str).strip()
    if size_str.endswith('M') or size_str.endswith('m'):
        try:
            return float(size_str[:-1])
        except:
            return np.nan
    elif size_str.endswith('k') or size_str.endswith('K'):
        try:
            return float(size_str[:-1]) / 1024.0
        except:
            return np.nan
    else:
        try:
            return float(size_str)
        except:
            return np.nan

df_apps_clean['Size_MB'] = df_apps_clean['Size'].apply(clean_size)

# Parse Last Updated to Datetime
df_apps_clean['Last Updated'] = pd.to_datetime(df_apps_clean['Last Updated'], errors='coerce')

# Impute missing Rating values with category median (or overall median if category median is NaN)
category_rating_median = df_apps_clean.groupby('Category')['Rating'].transform('median')
df_apps_clean['Rating'] = df_apps_clean['Rating'].fillna(category_rating_median).fillna(df_apps_clean['Rating'].median())

# Clean Category formatting (replace underscores with spaces, title case)
df_apps_clean['Category'] = df_apps_clean['Category'].astype(str).str.replace('_', ' ').str.title()

# Estimate Revenue (Price * Installs)
df_apps_clean['Estimated_Revenue'] = df_apps_clean['Price'] * df_apps_clean['Installs']

print(f"Apps Dataset Cleaned Shape: {df_apps_clean.shape[0]:,} rows x {df_apps_clean.shape[1]} columns")
print(f"Apps Duplicates Removed   : {len(df_apps) - len(df_apps_clean):,}")
df_apps_clean.to_csv("data/processed/googleplaystore_cleaned.csv", index=False)
print("Saved data/processed/googleplaystore_cleaned.csv")

# 3. Data Cleaning: User Reviews Dataset
print("\n=== 3. CLEANING REVIEWS DATASET ===")
# Drop missing review texts
df_reviews_clean = df_reviews.dropna(subset=['Translated_Review']).copy()

# Drop duplicate reviews
df_reviews_clean = df_reviews_clean.drop_duplicates(subset=['App', 'Translated_Review'], keep='first')

# Compute TextBlob sentiment polarity & subjectivity independently
def get_textblob_sentiment(text):
    try:
        blob = TextBlob(str(text))
        pol = blob.sentiment.polarity
        subj = blob.sentiment.subjectivity
        if pol > 0:
            label = 'Positive'
        elif pol < 0:
            label = 'Negative'
        else:
            label = 'Neutral'
        return pol, subj, label
    except:
        return 0.0, 0.0, 'Neutral'

tb_results = df_reviews_clean['Translated_Review'].apply(get_textblob_sentiment)
df_reviews_clean['TextBlob_Polarity'] = [r[0] for r in tb_results]
df_reviews_clean['TextBlob_Subjectivity'] = [r[1] for r in tb_results]
df_reviews_clean['TextBlob_Sentiment'] = [r[2] for r in tb_results]

print(f"User Reviews Cleaned Shape: {df_reviews_clean.shape[0]:,} rows x {df_reviews_clean.shape[1]} columns")
print(f"Reviews Rows Removed      : {len(df_reviews) - len(df_reviews_clean):,}")
df_reviews_clean.to_csv("data/processed/googleplaystore_reviews_cleaned.csv", index=False)
print("Saved data/processed/googleplaystore_reviews_cleaned.csv")

# 4. Analysis & CSV Reports Generation
print("\n=== 4. COMPUTING EDA STATS & CSV REPORTS ===")

# Report 1: Category Summary Report
cat_summary = df_apps_clean.groupby('Category').agg(
    App_Count=('App', 'count'),
    Average_Rating=('Rating', 'mean'),
    Total_Reviews=('Reviews', 'sum'),
    Average_Installs=('Installs', 'mean'),
    Total_Installs=('Installs', 'sum')
).reset_index()

cat_summary['Average_Rating'] = cat_summary['Average_Rating'].round(2)
cat_summary['Average_Installs'] = cat_summary['Average_Installs'].round(0)
cat_summary = cat_summary.sort_values(by='App_Count', ascending=False)
cat_summary.to_csv("outputs/reports/category_summary.csv", index=False)
print("Saved outputs/reports/category_summary.csv")

# Report 2: Top Apps by Installs
top_installs = df_apps_clean.sort_values(by=['Installs', 'Reviews'], ascending=[False, False])[['App', 'Category', 'Rating', 'Reviews', 'Installs', 'Type', 'Price']].head(10)
top_installs.to_csv("outputs/reports/top_apps_by_installs.csv", index=False)
print("Saved outputs/reports/top_apps_by_installs.csv")

# Report 3: Top Apps by Reviews
top_reviews = df_apps_clean.sort_values(by='Reviews', ascending=False)[['App', 'Category', 'Rating', 'Reviews', 'Installs', 'Type', 'Price']].head(10)
top_reviews.to_csv("outputs/reports/top_apps_by_reviews.csv", index=False)
print("Saved outputs/reports/top_apps_by_reviews.csv")

# Report 4: Free vs Paid Summary
free_vs_paid = df_apps_clean.groupby('Type').agg(
    App_Count=('App', 'count'),
    Average_Rating=('Rating', 'mean'),
    Total_Reviews=('Reviews', 'sum'),
    Average_Installs=('Installs', 'mean'),
    Average_Price=('Price', 'mean'),
    Total_Estimated_Revenue=('Estimated_Revenue', 'sum')
).reset_index()

free_vs_paid['Percentage (%)'] = (free_vs_paid['App_Count'] / len(df_apps_clean) * 100).round(2)
free_vs_paid['Average_Rating'] = free_vs_paid['Average_Rating'].round(2)
free_vs_paid['Average_Installs'] = free_vs_paid['Average_Installs'].round(0)
free_vs_paid['Average_Price'] = free_vs_paid['Average_Price'].round(2)
free_vs_paid.to_csv("outputs/reports/free_vs_paid_summary.csv", index=False)
print("Saved outputs/reports/free_vs_paid_summary.csv")

# Report 5: Sentiment Summary Report
sent_summary = df_reviews_clean['Sentiment'].value_counts().reset_index()
sent_summary.columns = ['Sentiment', 'Count']
sent_summary['Percentage (%)'] = (sent_summary['Count'] / len(df_reviews_clean) * 100).round(2)

# Merge reviews with apps to get category sentiment
df_merged_reviews = pd.merge(df_reviews_clean, df_apps_clean[['App', 'Category']], on='App', how='inner')
cat_sent = df_merged_reviews.groupby('Category').agg(
    Review_Count=('Translated_Review', 'count'),
    Average_Polarity=('Sentiment_Polarity', 'mean'),
    Average_Subjectivity=('Sentiment_Subjectivity', 'mean')
).reset_index().round(4)

sent_summary.to_csv("outputs/reports/sentiment_summary.csv", index=False)
print("Saved outputs/reports/sentiment_summary.csv")

# Report 6: Correlation Matrix Report
num_cols = ['Rating', 'Reviews', 'Installs', 'Price', 'Size_MB']
corr_matrix = df_apps_clean[num_cols].corr().round(4)
corr_matrix.to_csv("outputs/reports/correlation_matrix.csv")
print("Saved outputs/reports/correlation_matrix.csv")

# Report 7: Key Insights CSV
insights_list = [
    {"Insight_ID": 1, "Area": "Category Dominance", "Finding": f"The '{cat_summary.iloc[0]['Category']}' category leads in total apps ({cat_summary.iloc[0]['App_Count']:,} apps), representing {cat_summary.iloc[0]['App_Count']/len(df_apps_clean)*100:.2f}% of the Play Store store catalog."},
    {"Insight_ID": 2, "Area": "Category Installs", "Finding": f"The '{cat_summary.sort_values(by='Total_Installs', ascending=False).iloc[0]['Category']}' category generated the highest total installs ({cat_summary.sort_values(by='Total_Installs', ascending=False).iloc[0]['Total_Installs']:,} installs)."},
    {"Insight_ID": 3, "Area": "Free vs Paid Distribution", "Finding": f"Free apps represent {free_vs_paid[free_vs_paid['Type']=='Free']['Percentage (%)'].values[0]}% of apps ({free_vs_paid[free_vs_paid['Type']=='Free']['App_Count'].values[0]:,} apps), while Paid apps represent {free_vs_paid[free_vs_paid['Type']=='Paid']['Percentage (%)'].values[0]}% ({free_vs_paid[free_vs_paid['Type']=='Paid']['App_Count'].values[0]:,} apps)."},
    {"Insight_ID": 4, "Area": "Free vs Paid Adoption", "Finding": f"Free apps achieve substantially higher average installs ({free_vs_paid[free_vs_paid['Type']=='Free']['Average_Installs'].values[0]:,.0f}) compared to Paid apps ({free_vs_paid[free_vs_paid['Type']=='Paid']['Average_Installs'].values[0]:,.0f})."},
    {"Insight_ID": 5, "Area": "Correlation Insights", "Finding": f"Reviews and Installs exhibit a strong positive correlation (r = {corr_matrix.loc['Reviews', 'Installs']:.4f}), demonstrating that user reviews strongly drive app downloads."},
    {"Insight_ID": 6, "Area": "Sentiment Breakdown", "Finding": f"User reviews are overwhelmingly Positive ({sent_summary[sent_summary['Sentiment']=='Positive']['Percentage (%)'].values[0]}%), followed by Negative ({sent_summary[sent_summary['Sentiment']=='Negative']['Percentage (%)'].values[0]}%) and Neutral ({sent_summary[sent_summary['Sentiment']=='Neutral']['Percentage (%)'].values[0]}%)."},
    {"Insight_ID": 7, "Area": "Rating Distribution", "Finding": f"Overall average app rating across the Google Play Store is {df_apps_clean['Rating'].mean():.2f} / 5.0, with ratings displaying a distinct left-skewed distribution."},
    {"Insight_ID": 8, "Area": "Estimated Paid Revenue", "Finding": f"Total estimated gross revenue proxy across all paid apps is ${free_vs_paid[free_vs_paid['Type']=='Paid']['Total_Estimated_Revenue'].values[0]:,.2f}."}
]
insights_df = pd.DataFrame(insights_list)
insights_df.to_csv("outputs/reports/key_insights.csv", index=False)
print("Saved outputs/reports/key_insights.csv")

# 5. Visualizations Generation (15 High-Resolution Charts)
print("\n=== 5. GENERATING VISUALIZATIONS ===")

# Chart 1: Apps by Category
plt.figure(figsize=(12, 6))
top_cats = cat_summary.head(15)
ax = sns.barplot(data=top_cats, x='App_Count', y='Category', palette='viridis')
plt.title("Top 15 App Categories by Total App Count", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Number of Apps", fontsize=11)
plt.ylabel("Category", fontsize=11)
for p in ax.patches:
    width = p.get_width()
    ax.annotate(f"{int(width):,}", (width + 10, p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/apps_by_category.png", dpi=300)
plt.close()

# Chart 2: Average Rating by Category
plt.figure(figsize=(12, 6))
top_rating_cats = cat_summary.sort_values(by='Average_Rating', ascending=False).head(15)
ax = sns.barplot(data=top_rating_cats, x='Average_Rating', y='Category', palette='magma')
plt.title("Top 15 Categories by Average Rating", fontsize=13, fontweight='bold', pad=12)
plt.xlim(3.8, 5.0)
plt.xlabel("Average Rating (Out of 5.0)", fontsize=11)
plt.ylabel("Category", fontsize=11)
for p in ax.patches:
    width = p.get_width()
    ax.annotate(f"{width:.2f}", (width + 0.02, p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/avg_rating_by_category.png", dpi=300)
plt.close()

# Chart 3: Top 10 Apps by Installs
plt.figure(figsize=(12, 5))
ax = sns.barplot(data=top_installs, x='Installs', y='App', palette='crest')
plt.title("Top 10 Most Installed Google Play Store Apps", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Total Downloads (Installs)", fontsize=11)
plt.ylabel("App Name", fontsize=11)
for p in ax.patches:
    width = p.get_width()
    ax.annotate(f"{int(width):,}", (width * 0.5, p.get_y() + p.get_height() / 2.),
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/top_10_installed_apps.png", dpi=300)
plt.close()

# Chart 4: Top 10 Apps by Reviews
plt.figure(figsize=(12, 5))
ax = sns.barplot(data=top_reviews, x='Reviews', y='App', palette='flare')
plt.title("Top 10 Most Reviewed Google Play Store Apps", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Total User Reviews", fontsize=11)
plt.ylabel("App Name", fontsize=11)
for p in ax.patches:
    width = p.get_width()
    ax.annotate(f"{int(width):,}", (width * 0.5, p.get_y() + p.get_height() / 2.),
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/top_10_reviewed_apps.png", dpi=300)
plt.close()

# Chart 5: Rating Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df_apps_clean['Rating'], bins=30, kde=True, color='#3498db')
plt.axvline(df_apps_clean['Rating'].mean(), color='red', linestyle='--', label=f"Mean: {df_apps_clean['Rating'].mean():.2f}")
plt.axvline(df_apps_clean['Rating'].median(), color='green', linestyle='-', label=f"Median: {df_apps_clean['Rating'].median():.2f}")
plt.title("Google Play Store App Rating Distribution", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Rating (1.0 to 5.0)", fontsize=11)
plt.ylabel("App Count", fontsize=11)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("outputs/charts/rating_distribution.png", dpi=300)
plt.close()

# Chart 6: Installs Distribution (Log Scale)
plt.figure(figsize=(8, 5))
sns.countplot(data=df_apps_clean, x='Installs', palette='Blues_r', order=df_apps_clean['Installs'].value_counts().index[:10])
plt.yscale('log')
plt.xticks(rotation=45)
plt.title("Distribution of App Installs (Log Scale)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Install Bracket", fontsize=11)
plt.ylabel("App Count (Log Scale)", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/charts/installs_distribution.png", dpi=300)
plt.close()

# Chart 7: Free vs Paid App Count
plt.figure(figsize=(7, 5))
palette_fp = {'Free': '#2ecc71', 'Paid': '#e74c3c'}
ax = sns.barplot(data=free_vs_paid, x='Type', y='App_Count', palette=palette_fp)
plt.title("Free vs. Paid App Count Comparison", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("App Type", fontsize=11)
plt.ylabel("App Count", fontsize=11)
for p in ax.patches:
    height = p.get_height()
    pct = (height / len(df_apps_clean)) * 100
    ax.annotate(f"{int(height):,}\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2., height / 2),
                ha='center', va='center', fontsize=11, color='white', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/free_vs_paid_count.png", dpi=300)
plt.close()

# Chart 8: Price Distribution of Paid Apps
plt.figure(figsize=(8, 5))
df_paid = df_apps_clean[df_apps_clean['Price'] > 0]
sns.histplot(df_paid[df_paid['Price'] < 50]['Price'], bins=30, kde=True, color='#e74c3c')
plt.title("Price Distribution of Paid Apps (Price < $50)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Price ($)", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/charts/paid_app_price_distribution.png", dpi=300)
plt.close()

# Chart 9: Content Rating Distribution
plt.figure(figsize=(8, 5))
cr_counts = df_apps_clean['Content Rating'].value_counts()
ax = sns.barplot(x=cr_counts.index, y=cr_counts.values, palette='Spectral')
plt.xticks(rotation=15)
plt.title("App Distribution by Content Rating", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Content Rating Category", fontsize=11)
plt.ylabel("Number of Apps", fontsize=11)
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f"{int(height):,}", (p.get_x() + p.get_width() / 2., height + 50),
                ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/content_rating_distribution.png", dpi=300)
plt.close()

# Chart 10: App Size Distribution (MB)
plt.figure(figsize=(8, 5))
sns.histplot(df_apps_clean['Size_MB'].dropna(), bins=40, kde=True, color='#9b59b6')
plt.title("App Size Distribution (Megabytes)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Size (MB)", fontsize=11)
plt.ylabel("App Count", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/charts/app_size_distribution.png", dpi=300)
plt.close()

# Chart 11: Sentiment Distribution
plt.figure(figsize=(7, 5))
palette_sent = {'Positive': '#2ecc71', 'Neutral': '#3498db', 'Negative': '#e74c3c'}
ax = sns.barplot(data=sent_summary, x='Sentiment', y='Count', palette=palette_sent)
plt.title("User Review Sentiment Distribution", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Sentiment Classification", fontsize=11)
plt.ylabel("Review Count", fontsize=11)
for p in ax.patches:
    height = p.get_height()
    pct = (height / len(df_reviews_clean)) * 100
    ax.annotate(f"{int(height):,}\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2., height / 2),
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/sentiment_distribution.png", dpi=300)
plt.close()

# Chart 12: Sentiment Polarity Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df_reviews_clean['Sentiment_Polarity'], bins=40, kde=True, color='#1abc9c')
plt.axvline(0, color='black', linestyle='--', alpha=0.7)
plt.title("User Review Sentiment Polarity Distribution (-1.0 to +1.0)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Sentiment Polarity Score", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/charts/sentiment_polarity_distribution.png", dpi=300)
plt.close()

# Chart 13: Average Sentiment by Category
plt.figure(figsize=(12, 6))
top_sent_cats = cat_sent.sort_values(by='Average_Polarity', ascending=False).head(15)
ax = sns.barplot(data=top_sent_cats, x='Average_Polarity', y='Category', palette='Greens_r')
plt.title("Top 15 App Categories by Average Review Sentiment Polarity", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Average Sentiment Polarity", fontsize=11)
plt.ylabel("Category", fontsize=11)
for p in ax.patches:
    width = p.get_width()
    ax.annotate(f"{width:.3f}", (width + 0.005, p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/avg_sentiment_by_category.png", dpi=300)
plt.close()

# Chart 14: Reviews vs Installs Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_apps_clean, x='Reviews', y='Installs', alpha=0.5, color='#e67e22')
plt.xscale('log')
plt.yscale('log')
plt.title("Reviews vs. Installs (Log-Log Scale)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Total Reviews (Log Scale)", fontsize=11)
plt.ylabel("Total Installs (Log Scale)", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/charts/reviews_vs_installs.png", dpi=300)
plt.close()

# Chart 15: Correlation Heatmap
plt.figure(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title("Numeric Features Correlation Heatmap", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("outputs/charts/correlation_heatmap.png", dpi=300)
plt.close()

print("\n=== PIPELINE RUN COMPLETED SUCCESSFULLY ===")
