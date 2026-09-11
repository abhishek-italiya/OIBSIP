import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

def run_pipeline():
    # 1. Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_red_path = os.path.join(base_dir, "data", "raw", "winequality-red.csv")
    raw_white_path = os.path.join(base_dir, "data", "raw", "winequality-white.csv")
    processed_data_path = os.path.join(base_dir, "data", "processed", "wine_quality_processed.csv")
    charts_dir = os.path.join(base_dir, "outputs", "charts")
    reports_dir = os.path.join(base_dir, "outputs", "reports")

    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 2. Load Raw Datasets
    print("--- Loading Raw Datasets ---")
    df_red = pd.read_csv(raw_red_path, sep=";")
    df_white = pd.read_csv(raw_white_path, sep=";")

    df_red["wine_type"] = "red"
    df_white["wine_type"] = "white"
    df_red["wine_type_code"] = 0
    df_white["wine_type_code"] = 1

    df = pd.concat([df_red, df_white], ignore_index=True)
    print(f"Raw Red shape: {df_red.shape}")
    print(f"Raw White shape: {df_white.shape}")
    print(f"Combined shape: {df.shape}")

    # 3. Data Cleaning & Inspection
    print("\n--- Data Cleaning & Inspection ---")
    print(f"Missing values sum:\n{df.isnull().sum()}")
    
    num_duplicates = df.duplicated(subset=[col for col in df.columns if col not in ["wine_type"]]).sum()
    print(f"Duplicate rows detected across physicochemical features: {num_duplicates}")
    
    # Drop duplicate rows across physical features
    df_cleaned = df.drop_duplicates(subset=[col for col in df.columns if col not in ["wine_type"]]).copy()
    print(f"Shape after removing duplicate rows: {df_cleaned.shape}")

    # Map quality target into Low (<=5), Medium (6), High (>=7)
    def map_quality(q):
        if q <= 5:
            return "Low"
        elif q == 6:
            return "Medium"
        else:
            return "High"

    df_cleaned["quality_class"] = df_cleaned["quality"].apply(map_quality)
    
    # Save processed dataset
    df_cleaned.to_csv(processed_data_path, index=False)
    print(f"Saved processed dataset to: {processed_data_path}")

    # 4. Visualization 1: Quality Distribution
    plt.figure(figsize=(10, 6))
    palette = {"red": "#b30000", "white": "#d9c27c"}
    ax = sns.countplot(data=df_cleaned, x="quality", hue="wine_type", palette=palette, edgecolor="black", alpha=0.85)
    plt.title("Wine Quality Score Distribution (UCI Dataset)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Original Quality Score (3 to 9)", fontsize=12)
    plt.ylabel("Number of Wines", fontsize=12)
    plt.legend(title="Wine Type", frameon=True)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height / 2),
                        ha='center', va='center', fontsize=9, color='white' if p.get_facecolor()[0] < 0.5 else 'black', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "quality_distribution.png"), dpi=300)
    plt.close()
    print("Generated quality_distribution.png")

    # 5. Visualization 2: Class Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    class_order = ["Low", "Medium", "High"]
    class_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    
    counts = df_cleaned["quality_class"].value_counts().reindex(class_order)
    percentages = (counts / len(df_cleaned)) * 100

    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=class_colors, ax=ax1, edgecolor="black", legend=False)
    ax1.set_title("Wine Quality Classification Counts", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Quality Class", fontsize=11)
    ax1.set_ylabel("Sample Count", fontsize=11)
    for i, count in enumerate(counts.values):
        ax1.text(i, count + 30, f"{count} ({percentages.iloc[i]:.1f}%)", ha="center", fontsize=10, fontweight="bold")

    ax2.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=class_colors, startangle=140,
            explode=(0.03, 0.03, 0.03), textprops={"fontsize": 11, "fontweight": "bold"}, wedgeprops={"edgecolor": "black"})
    ax2.set_title("Wine Quality Class Percentage Breakdown", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "class_distribution.png"), dpi=300)
    plt.close()
    print("Generated class_distribution.png")

    # 6. Visualization 3: Correlation Heatmap
    plt.figure(figsize=(12, 9))
    num_cols = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol", "wine_type_code", "quality"
    ]
    corr = df_cleaned[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Physicochemical Features Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()
    print("Generated correlation_heatmap.png")

    # 7. Visualization 4: Feature Distributions
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    key_features = ["alcohol", "volatile acidity", "sulphates", "citric acid", "density", "pH"]
    
    for idx, feature in enumerate(key_features):
        ax = axes[idx // 3, idx % 3]
        sns.boxplot(data=df_cleaned, x="quality_class", y=feature, order=class_order, hue="quality_class", palette=class_colors, ax=ax, width=0.5, legend=False)
        ax.set_title(f"{feature.title()} by Quality Class", fontsize=12, fontweight="bold")
        ax.set_xlabel("Quality Class", fontsize=10)
        ax.set_ylabel(feature, fontsize=10)

    plt.suptitle("Physicochemical Feature Distributions Across Wine Quality Classes", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(charts_dir, "feature_distributions.png"), dpi=300)
    plt.close()
    print("Generated feature_distributions.png")

    # 8. Machine Learning Data Preparation
    feature_cols = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol", "wine_type_code"
    ]
    X = df_cleaned[feature_cols]
    y = df_cleaned["quality_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\n--- Machine Learning Data Split ---")
    print(f"Training Samples: {len(X_train)} ({len(X_train)/len(df_cleaned)*100:.1f}%)")
    print(f"Testing Samples:  {len(X_test)} ({len(X_test)/len(df_cleaned)*100:.1f}%)")

    # 9. Model Training & Evaluation
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "SGD Classifier": Pipeline([
            ("scaler", StandardScaler()),
            ("sgd", SGDClassifier(loss="log_loss", random_state=42, class_weight="balanced", max_iter=1000))
        ]),
        "SVC": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", random_state=42, class_weight="balanced"))
        ])
    }

    comparison_results = []
    classification_reports_list = []
    confusion_filenames = {
        "Random Forest": "random_forest_confusion_matrix.png",
        "SGD Classifier": "sgd_confusion_matrix.png",
        "SVC": "svc_confusion_matrix.png"
    }

    for name, model in models.items():
        print(f"\nTraining Model: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_prec = precision_score(y_test, y_pred, average="macro")
        macro_rec = recall_score(y_test, y_pred, average="macro")
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_prec = precision_score(y_test, y_pred, average="weighted")
        weighted_rec = recall_score(y_test, y_pred, average="weighted")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")

        comparison_results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Macro_Precision": round(macro_prec, 4),
            "Macro_Recall": round(macro_rec, 4),
            "Macro_F1": round(macro_f1, 4),
            "Weighted_Precision": round(weighted_prec, 4),
            "Weighted_Recall": round(weighted_rec, 4),
            "Weighted_F1": round(weighted_f1, 4)
        })

        # Classification report dict
        report_dict = classification_report(y_test, y_pred, labels=class_order, output_dict=True)
        for cls in class_order:
            classification_reports_list.append({
                "Model": name,
                "Class": cls,
                "Precision": round(report_dict[cls]["precision"], 4),
                "Recall": round(report_dict[cls]["recall"], 4),
                "F1-Score": round(report_dict[cls]["f1-score"], 4),
                "Support": int(report_dict[cls]["support"])
            })

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=class_order)

        # Save individual confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_order)
        disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)
        ax.set_title(f"{name} Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Quality Class", fontsize=11)
        ax.set_ylabel("True Quality Class", fontsize=11)
        plt.tight_layout()
        filename = confusion_filenames[name]
        plt.savefig(os.path.join(charts_dir, filename), dpi=300)
        plt.close()
        print(f"Generated {filename}")

    # Remove old sgd_classifier_confusion_matrix.png if exists
    old_sgd_file = os.path.join(charts_dir, "sgd_classifier_confusion_matrix.png")
    if os.path.exists(old_sgd_file):
        os.remove(old_sgd_file)

    # 10. Save Reports CSVs
    df_comparison = pd.DataFrame(comparison_results)
    df_comparison.to_csv(os.path.join(reports_dir, "model_comparison.csv"), index=False)
    print("\nSaved model_comparison.csv")

    df_reports = pd.DataFrame(classification_reports_list)
    df_reports.to_csv(os.path.join(reports_dir, "classification_reports.csv"), index=False)
    print("Saved classification_reports.csv")

    # 11. Feature Importances (Random Forest)
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    df_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    df_importance.to_csv(os.path.join(reports_dir, "feature_importance.csv"), index=False)
    print("Saved feature_importance.csv")

    # 12. Visualization 8: Model Comparison Chart
    plt.figure(figsize=(12, 6))
    df_melted = df_comparison.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Macro_F1", "Weighted_F1"],
        var_name="Metric",
        value_name="Score"
    )
    ax = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="viridis", edgecolor="black")
    plt.title("Model Performance Comparison (Accuracy, Macro F1, Weighted F1)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Machine Learning Model", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(title="Metric", loc="lower right", frameon=True)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h + 0.01),
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "model_comparison.png"), dpi=300)
    plt.close()
    print("Generated model_comparison.png")

    # 13. Visualization 9: Feature Importance Chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_importance, x="Importance", y="Feature", hue="Feature", palette="rocket", edgecolor="black", legend=False)
    plt.title("Random Forest Feature Importance", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Relative Importance Score", fontsize=12)
    plt.ylabel("Physicochemical Feature", fontsize=12)
    for p in ax.patches:
        w = p.get_width()
        ax.annotate(f"{w:.4f}", (w + 0.002, p.get_y() + p.get_height() / 2.),
                    ha="left", va="center", fontsize=9, fontweight="bold")
    plt.xlim(0, max(importances) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "feature_importance.png"), dpi=300)
    plt.close()
    print("Generated feature_importance.png")

    print("\n--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    run_pipeline()
