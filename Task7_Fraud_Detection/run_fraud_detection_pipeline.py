# pyrefly: ignore [missing-import]
import os
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)

# Ensure output directories exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# 1. Load Dataset
print("=== 1. LOADING DATASET ===")
raw_path = "data/raw/creditcard.csv"
df = pd.read_csv(raw_path)
print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Missing values count: {df.isnull().sum().sum()}")
print(f"Duplicate rows count: {df.duplicated().sum():,}")

# 2. Class Imbalance Analysis
print("\n=== 2. CLASS IMBALANCE ANALYSIS ===")
class_counts = df['Class'].value_counts()
class_pct = df['Class'].value_counts(normalize=True) * 100

class_report_df = pd.DataFrame({
    'Class': ['Legitimate (0)', 'Fraudulent (1)'],
    'Count': [class_counts[0], class_counts[1]],
    'Percentage': [round(class_pct[0], 4), round(class_pct[1], 4)]
})
print(class_report_df.to_string(index=False))
class_report_df.to_csv("outputs/reports/class_distribution_report.csv", index=False)

# Chart 1: Class Distribution
plt.figure(figsize=(7, 5))
palette = ['#2ecc71', '#e74c3c']
ax = sns.barplot(x=['Legitimate (0)', 'Fraudulent (1)'], y=[class_counts[0], class_counts[1]], hue=['Legitimate (0)', 'Fraudulent (1)'], palette=palette, legend=False)
plt.yscale('log')
plt.title("Class Distribution (Log Scale)", fontsize=13, fontweight='bold', pad=12)
plt.ylabel("Transaction Count (Log Scale)", fontsize=11)
plt.xlabel("Transaction Type", fontsize=11)

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f"{int(height):,}\n({height/len(df)*100:.2f}%)",
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=10, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig("outputs/charts/class_distribution.png", dpi=300)
plt.close()

# 3. Exploratory Analysis: Time & Amount
print("\n=== 3. EXPLORATORY DATA ANALYSIS ===")
# Chart 2: Transaction Time Distribution
plt.figure(figsize=(10, 4))
sns.histplot(data=df, x='Time', hue='Class', bins=50, kde=True, palette=palette, element='step')
plt.title("Transaction Time Distribution (Seconds since First Transaction)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Time (Seconds)", fontsize=11)
plt.ylabel("Transaction Volume", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/charts/transaction_time_distribution.png", dpi=300)
plt.close()

# Chart 3: Transaction Amount Distribution (Log-scaled)
plt.figure(figsize=(10, 4))
sns.histplot(df['Amount'] + 1, bins=50, kde=True, color='#3498db', log_scale=True)
plt.title("Transaction Amount Distribution (Log Scale)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Amount ($) [Log Scale]", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/charts/transaction_amount_distribution.png", dpi=300)
plt.close()

# Chart 4: Fraud Amount Comparison
plt.figure(figsize=(8, 5))
sns.boxplot(x='Class', y='Amount', data=df, hue='Class', palette=palette, showfliers=False, legend=False)
plt.xticks([0, 1], ['Legitimate (0)', 'Fraudulent (1)'])
plt.title("Transaction Amount Comparison (Outliers Omitted for Clarity)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Transaction Type", fontsize=11)
plt.ylabel("Amount ($)", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/charts/fraud_amount_comparison.png", dpi=300)
plt.close()

# Chart 5: Correlation Heatmap
plt.figure(figsize=(12, 10))
corr = df.corr()
cols_to_plot = corr['Class'].abs().sort_values(ascending=False).index[:15]
sns.heatmap(df[cols_to_plot].corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title("Correlation Heatmap (Top Features Correlated with Class)", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("outputs/charts/correlation_heatmap.png", dpi=300)
plt.close()

# 4. Strict Non-Leaking Data Preprocessing & Train/Test Split
print("\n=== 4. DATA PREPROCESSING & TRAIN/TEST SPLIT (STRICT LEAKAGE PREVENTION) ===")
# 1. Separate features X and target y FIRST
X = df.drop(columns=['Class'])
y = df['Class']

# 2. Stratified 80/20 train/test split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Copy to avoid modifying original views
X_train = X_train.copy()
X_test = X_test.copy()

# 3. Fit StandardScaler ONLY on training data to prevent data leakage
scaler = StandardScaler()
X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])

# 4. Transform test data using the ALREADY-FITTED training scaler
X_test[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])

print("StandardScaler fitted STRICTLY on X_train[['Amount', 'Time']]")
print(f"Training features shape : {X_train.shape}")
print(f"Testing features shape  : {X_test.shape}")
print(f"Train fraud count       : {y_train.sum()} ({y_train.mean()*100:.3f}%)")
print(f"Test fraud count        : {y_test.sum()} ({y_test.mean()*100:.3f}%)")

# Save processed export sample for verification
train_export = X_train.copy()
train_export['Class'] = y_train
train_export.to_csv("data/processed/fraud_detection_processed.csv", index=False)
print("Saved clean processed training dataset sample to data/processed/fraud_detection_processed.csv")

# 5. Model 1: Logistic Regression
print("\n=== 5. MODEL 1: LOGISTIC REGRESSION ===")
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

# 6. Model 2: Random Forest Classifier
print("\n=== 6. MODEL 2: RANDOM FOREST CLASSIFIER ===")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# 7. Model Evaluation Metrics & Reports
print("\n=== 7. MODEL EVALUATION ===")

def compute_metrics(y_true, y_pred, y_prob, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)
    return {
        'Model': model_name,
        'Accuracy': round(acc, 6),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'ROC_AUC': round(roc_auc, 4),
        'Average_Precision': round(avg_prec, 4)
    }

metrics_lr = compute_metrics(y_test, y_pred_lr, y_prob_lr, 'Logistic Regression')
metrics_rf = compute_metrics(y_test, y_pred_rf, y_prob_rf, 'Random Forest')

model_metrics_df = pd.DataFrame([metrics_lr, metrics_rf])
print(model_metrics_df.to_string(index=False))
model_metrics_df.to_csv("outputs/reports/model_metrics.csv", index=False)

# Detailed Classification Reports CSV
report_lr_dict = classification_report(y_test, y_pred_lr, target_names=['Legitimate', 'Fraudulent'], output_dict=True)
report_rf_dict = classification_report(y_test, y_pred_rf, target_names=['Legitimate', 'Fraudulent'], output_dict=True)

rows = []
for m_name, r_dict in [('Logistic Regression', report_lr_dict), ('Random Forest', report_rf_dict)]:
    for cls in ['Legitimate', 'Fraudulent', 'macro avg', 'weighted avg']:
        rows.append({
            'Model': m_name,
            'Class': cls,
            'Precision': round(r_dict[cls]['precision'], 4),
            'Recall': round(r_dict[cls]['recall'], 4),
            'F1-Score': round(r_dict[cls]['f1-score'], 4),
            'Support': int(r_dict[cls]['support'])
        })

clf_reports_df = pd.DataFrame(rows)
clf_reports_df.to_csv("outputs/reports/classification_reports.csv", index=False)

# Chart 6: Logistic Regression Confusion Matrix
plt.figure(figsize=(6, 5))
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraudulent'], yticklabels=['Legitimate', 'Fraudulent'])
plt.title("Logistic Regression Confusion Matrix", fontsize=12, fontweight='bold', pad=10)
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("Actual Label", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/charts/logistic_regression_confusion_matrix.png", dpi=300)
plt.close()

# Chart 7: Random Forest Confusion Matrix
plt.figure(figsize=(6, 5))
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', xticklabels=['Legitimate', 'Fraudulent'], yticklabels=['Legitimate', 'Fraudulent'])
plt.title("Random Forest Confusion Matrix", fontsize=12, fontweight='bold', pad=10)
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("Actual Label", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/charts/tree_confusion_matrix.png", dpi=300)
plt.savefig("outputs/charts/random_forest_confusion_matrix.png", dpi=300)
plt.close()

# Chart 8: ROC-AUC Comparison Curve
plt.figure(figsize=(8, 6))
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {metrics_lr['ROC_AUC']:.4f})", color='#3498db', lw=2)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {metrics_rf['ROC_AUC']:.4f})", color='#2ecc71', lw=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.5000)')
plt.title("ROC-AUC Curves Comparison", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Recall)", fontsize=11)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig("outputs/charts/roc_auc_comparison.png", dpi=300)
plt.close()

# Chart 9: Precision-Recall Curve Comparison
plt.figure(figsize=(8, 6))
prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_prob_lr)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)

plt.plot(rec_lr, prec_lr, label=f"Logistic Regression (PR-AUC = {metrics_lr['Average_Precision']:.4f})", color='#3498db', lw=2)
plt.plot(rec_rf, prec_rf, label=f"Random Forest (PR-AUC = {metrics_rf['Average_Precision']:.4f})", color='#2ecc71', lw=2)
plt.title("Precision-Recall Curves Comparison", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Recall (Sensitivity)", fontsize=11)
plt.ylabel("Precision", fontsize=11)
plt.legend(loc='lower left', frameon=True)
plt.tight_layout()
plt.savefig("outputs/charts/precision_recall_curve.png", dpi=300)
plt.close()

# Chart 10: Model Metrics Comparison Bar Chart
plt.figure(figsize=(9, 5))
metrics_melted = pd.melt(model_metrics_df, id_vars=['Model'], value_vars=['Precision', 'Recall', 'F1', 'ROC_AUC', 'Average_Precision'], var_name='Metric', value_name='Score')
ax = sns.barplot(data=metrics_melted, x='Metric', y='Score', hue='Model', palette=['#3498db', '#2ecc71'])
plt.title("Model Evaluation Metrics Comparison (Fraud Class = 1)", fontsize=13, fontweight='bold', pad=12)
plt.ylim(0.0, 1.08)
plt.ylabel("Metric Score", fontsize=11)

for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height) and height > 0:
        ax.annotate(f"{height:.4f}", (p.get_x() + p.get_width() / 2., height + 0.015),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig("outputs/charts/model_comparison.png", dpi=300)
plt.close()

# 8. Feature Importance
print("\n=== 8. FEATURE IMPORTANCE ===")
importances = rf_model.feature_importances_
feature_names = X_train.columns

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

feat_imp_df.to_csv("outputs/reports/feature_importance.csv", index=False)
print("Top 10 Most Important Features:")
print(feat_imp_df.head(10).to_string(index=False))

# Chart 11: Feature Importance Top 15
plt.figure(figsize=(9, 6))
sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature', hue='Feature', palette='crest', legend=False)
plt.title("Top 15 Feature Importances (Random Forest)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Importance Score", fontsize=11)
plt.ylabel("Feature Name", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/charts/feature_importance.png", dpi=300)
plt.close()

print("\n=== CORRECTED PIPELINE COMPLETED SUCCESSFULLY ===")
