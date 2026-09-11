# pyrefly: ignore [missing-import]
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Set aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"

# Ensure output directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "house_prices_processed.csv")
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "charts")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def run_pipeline():
    print("=== 1. LOADING DATASET ===")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw Dataset Shape: {df.shape}")
    print(f"Columns count: {len(df.columns)}")

    # 2. MISSING VALUES REPORT
    print("\n=== 2. MISSING VALUES ANALYSIS ===")
    null_counts = df.isnull().sum()
    null_percent = (null_counts / len(df)) * 100
    missing_df = pd.DataFrame({
        "Column": null_counts.index,
        "Missing_Count": null_counts.values,
        "Missing_Percentage": np.round(null_percent.values, 2)
    })
    missing_summary = missing_df[missing_df["Missing_Count"] > 0].sort_values(by="Missing_Count", ascending=False)
    missing_summary_path = os.path.join(REPORTS_DIR, "missing_values_summary.csv")
    missing_summary.to_csv(missing_summary_path, index=False)
    print(f"Saved missing values summary to {missing_summary_path} ({len(missing_summary)} columns with missing values)")

    # 3. DATA CLEANING & PROCESSED EXPORT
    print("\n=== 3. DATA CLEANING & PROCESSED EXPORT ===")
    # Handle duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows detected: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates().copy()

    # Save processed dataframe version
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed dataset to {PROCESSED_DATA_PATH}")

    # 4. EDA CHARTS
    print("\n=== 4. GENERATING EDA CHARTS ===")
    # Chart 1: SalePrice Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["SalePrice"], kde=True, color="#2b5c8f", bins=40, edgecolor="white")
    mean_val = df["SalePrice"].mean()
    median_val = df["SalePrice"].median()
    plt.axvline(mean_val, color="#d9534f", linestyle="--", linewidth=2, label=f"Mean (${mean_val:,.0f})")
    plt.axvline(median_val, color="#5cb85c", linestyle="-", linewidth=2, label=f"Median (${median_val:,.0f})")
    plt.title("Distribution of House Sale Prices (SalePrice)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Sale Price ($)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    saleprice_chart_path = os.path.join(CHARTS_DIR, "saleprice_distribution.png")
    plt.savefig(saleprice_chart_path, dpi=300)
    plt.close()
    print(f"Saved {saleprice_chart_path}")

    # Chart 2: Correlation Heatmap for key numerical variables
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_all.remove("Id")
    corr_matrix = df[num_cols_all].corr()

    # Get top 12 numerical features correlated with SalePrice
    top_corr_features = corr_matrix["SalePrice"].abs().sort_values(ascending=False).head(13).index
    top_corr_matrix = df[top_corr_features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(top_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Top Numerical Features Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    corr_heatmap_path = os.path.join(CHARTS_DIR, "correlation_heatmap.png")
    plt.savefig(corr_heatmap_path, dpi=300)
    plt.close()
    print(f"Saved {corr_heatmap_path}")

    # Chart 3: Top Features Correlation Bar Chart
    top_corr_series = corr_matrix["SalePrice"].drop("SalePrice").sort_values(ascending=False).head(12)
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=top_corr_series.values, y=top_corr_series.index, hue=top_corr_series.index, palette="Blues_r", legend=False)
    plt.title("Top 12 Numerical Features Correlated with SalePrice", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Pearson Correlation Coefficient", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    for p in bars.patches:
        width = p.get_width()
        bars.annotate(f"{width:.3f}", (width + 0.01, p.get_y() + p.get_height() / 2.),
                      ha="left", va="center", fontsize=10)
    plt.xlim(0, 0.9)
    plt.tight_layout()
    top_corr_chart_path = os.path.join(CHARTS_DIR, "top_features_correlation.png")
    plt.savefig(top_corr_chart_path, dpi=300)
    plt.close()
    print(f"Saved {top_corr_chart_path}")

    # 5. FEATURE SELECTION & PREPROCESSING PIPELINE
    print("\n=== 5. FEATURE SELECTION & MODEL PIPELINE SETUP ===")
    selected_num_cols = [
        "OverallQual", "GrLivArea", "GarageCars", "GarageArea",
        "TotalBsmtSF", "1stFlrSF", "FullBath", "TotRmsAbvGrd",
        "YearBuilt", "YearRemodAdd", "Fireplaces", "LotArea", "LotFrontage"
    ]

    selected_cat_cols = [
        "Neighborhood", "ExterQual", "KitchenQual", "BsmtQual", "MSZoning"
    ]

    features = selected_num_cols + selected_cat_cols
    target = "SalePrice"

    X = df[features]
    y = df[target]

    print(f"Selected Features count: {len(features)} ({len(selected_num_cols)} numerical, {len(selected_cat_cols)} categorical)")

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

    # Define Column Transformer
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, selected_num_cols),
        ("cat", cat_transformer, selected_cat_cols)
    ])

    # 6. MODEL TRAINING & COMPARISON
    print("\n=== 6. TRAINING & EVALUATING MODELS ===")
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=42),
        "Lasso Regression": Lasso(alpha=100.0, random_state=42, max_iter=5000)
    }

    metrics_list = []
    trained_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metrics_list.append({
            "Model": name,
            "MSE": np.round(mse, 2),
            "RMSE": np.round(rmse, 2),
            "R2": np.round(r2, 4)
        })
        print(f"Model: {name:<20} | RMSE: ${rmse:,.2f} | R²: {r2:.4f}")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_csv_path = os.path.join(REPORTS_DIR, "model_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved model metrics report to {metrics_csv_path}")

    # 7. PRIMARY MODEL DIAGNOSTICS & CHARTS (Linear Regression)
    lr_pipeline = trained_pipelines["Linear Regression"]
    lr_preds = lr_pipeline.predict(X_test)
    residuals = y_test - lr_preds

    # Chart 4: Actual vs Predicted Plot
    plt.figure(figsize=(9, 7))
    plt.scatter(y_test, lr_preds, alpha=0.6, color="#2b5c8f", edgecolors="w", s=50)
    min_val = min(y_test.min(), lr_preds.min())
    max_val = max(y_test.max(), lr_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#d9534f", linestyle="--", linewidth=2, label="Perfect Prediction (y = x)")
    plt.title("Actual vs Predicted Sale Price (Linear Regression)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Actual SalePrice ($)", fontsize=12)
    plt.ylabel("Predicted SalePrice ($)", fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    actual_vs_pred_path = os.path.join(CHARTS_DIR, "actual_vs_predicted.png")
    plt.savefig(actual_vs_pred_path, dpi=300)
    plt.close()
    print(f"Saved {actual_vs_pred_path}")

    # Chart 5: Residual Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(lr_preds, residuals, alpha=0.6, color="#e67e22", edgecolors="w", s=50)
    plt.axhline(0, color="#d9534f", linestyle="--", linewidth=2)
    plt.title("Residuals vs Predicted Values (Linear Regression)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted SalePrice ($)", fontsize=12)
    plt.ylabel("Residuals (Actual - Predicted) ($)", fontsize=12)
    plt.tight_layout()
    residual_plot_path = os.path.join(CHARTS_DIR, "residual_plot.png")
    plt.savefig(residual_plot_path, dpi=300)
    plt.close()
    print(f"Saved {residual_plot_path}")

    # 8. COEFFICIENT ANALYSIS
    print("\n=== 8. FEATURE COEFFICIENTS ANALYSIS ===")
    preprocessor_fitted = lr_pipeline.named_steps["preprocessor"]
    cat_feature_names = preprocessor_fitted.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(selected_cat_cols)
    all_feature_names = selected_num_cols + list(cat_feature_names)

    lr_model = lr_pipeline.named_steps["regressor"]
    coefficients = lr_model.coef_

    coef_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Coefficient": coefficients,
        "Absolute_Coefficient": np.abs(coefficients)
    }).sort_values(by="Absolute_Coefficient", ascending=False)

    coef_csv_path = os.path.join(REPORTS_DIR, "feature_coefficients.csv")
    coef_df.to_csv(coef_csv_path, index=False)
    print(f"Saved feature coefficients report to {coef_csv_path}")

    # Chart 6: Top Coefficients Bar Plot
    top_pos_coef = coef_df.sort_values(by="Coefficient", ascending=False).head(8)
    top_neg_coef = coef_df.sort_values(by="Coefficient", ascending=True).head(8)
    top_coef_combined = pd.concat([top_pos_coef, top_neg_coef]).sort_values(by="Coefficient", ascending=True)

    plt.figure(figsize=(12, 8))
    colors = ["#d9534f" if c < 0 else "#2b5c8f" for c in top_coef_combined["Coefficient"]]
    bars = plt.barh(top_coef_combined["Feature"], top_coef_combined["Coefficient"], color=colors, edgecolor="black", alpha=0.85)
    plt.axvline(0, color="black", linestyle="-", linewidth=0.8)
    plt.title("Top Most Influential Feature Coefficients (Linear Regression)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Coefficient Value ($ change per std dev / category)", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    for bar in bars:
        width = bar.get_width()
        offset = 500 if width >= 0 else -500
        ha = "left" if width >= 0 else "right"
        plt.annotate(f"${width:,.0f}", (width + offset, bar.get_y() + bar.get_height() / 2.),
                     ha=ha, va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    coef_chart_path = os.path.join(CHARTS_DIR, "coefficient_importance.png")
    plt.savefig(coef_chart_path, dpi=300)
    plt.close()
    print(f"Saved {coef_chart_path}")

    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_pipeline()
