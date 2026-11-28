
# ==========================================
# STEP 17C.4: EVALUATE TUNED CATBOOST
# ==========================================

import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# Project paths
project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "processed_data"
model_folder = project_root / "cancellation_prediction" / "saved_models"

# Load validation data
val_df = pd.read_csv(data_folder / "cp_validation.csv")

y_val = val_df["is_canceled"]

# Load saved CatBoost model and preprocessing information
model_package = joblib.load(
    model_folder / "catboost_tuned_v1.joblib"
)

tuned_model = model_package["model"]
X_val = val_df[model_package["feature_columns"]].copy()

# Prepare numerical features
for col, median in model_package["numeric_medians"].items():
    X_val[col] = X_val[col].fillna(median)

# Prepare categorical features
for col in model_package["categorical_features"]:
    X_val[col] = X_val[col].fillna("Unknown").astype(str)

# Predict cancellation probabilities
y_proba = tuned_model.predict_proba(X_val)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# Evaluate model performance
results = {
    "Accuracy": accuracy_score(y_val, y_pred),
    "Precision": precision_score(y_val, y_pred),
    "Recall": recall_score(y_val, y_pred),
    "F1-score": f1_score(y_val, y_pred),
    "ROC-AUC": roc_auc_score(y_val, y_proba),
    "PR-AUC": average_precision_score(y_val, y_proba)
}

print("\nSTEP 17C.4: Tuned CatBoost Evaluation")

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")







# ==========================================
# STEP 17C.5: COMPARE BASELINE VS TUNED CATBOOST
# ==========================================

from catboost import CatBoostClassifier

# Load training data
train_df = pd.read_csv(data_folder / "cp_train.csv")

X_train = train_df[model_package["feature_columns"]].copy()
y_train = train_df["is_canceled"]

# Apply the same preprocessing
for col, median in model_package["numeric_medians"].items():
    X_train[col] = X_train[col].fillna(median)

for col in model_package["categorical_features"]:
    X_train[col] = X_train[col].fillna("Unknown").astype(str)

# Create CatBoost baseline
baseline_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    random_seed=42,
    verbose=False,
    allow_writing_files=False
)

print("\nSTEP 17C.5: CatBoost Baseline vs Tuned")

# Train baseline only
baseline_model.fit(
    X_train,
    y_train,
    cat_features=model_package["categorical_features"]
)

# Evaluate baseline
baseline_proba = baseline_model.predict_proba(X_val)[:, 1]
baseline_pred = (baseline_proba >= 0.5).astype(int)

baseline_results = {
    "Accuracy": accuracy_score(y_val, baseline_pred),
    "Precision": precision_score(y_val, baseline_pred),
    "Recall": recall_score(y_val, baseline_pred),
    "F1-score": f1_score(y_val, baseline_pred),
    "ROC-AUC": roc_auc_score(y_val, baseline_proba),
    "PR-AUC": average_precision_score(y_val, baseline_proba)
}

# Compare both models
comparison = pd.DataFrame([
    {"Model": "Baseline", **baseline_results},
    {"Model": "Tuned V1", **results}
])

print("\nCatBoost Comparison:")
print(comparison.to_string(index=False, float_format="%.4f"))





# ==========================================
# STEP 17C.6: SAVE CATBOOST COMPARISON RESULTS
# ==========================================

import json

results_folder = project_root / "cancellation_prediction" / "results"
results_folder.mkdir(parents=True, exist_ok=True)

# Save comparison table
comparison.to_csv(
    results_folder / "catboost_baseline_vs_tuned.csv",
    index=False
)

# Update existing tuning summary with validation metrics
summary_path = results_folder / "catboost_tuning_summary_v1.json"

with open(summary_path, "r", encoding="utf-8") as file:
    summary = json.load(file)

summary["validation_metrics"] = {
    metric: float(value)
    for metric, value in results.items()
}

summary["baseline_validation_metrics"] = {
    metric: float(value)
    for metric, value in baseline_results.items()
}

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("\nCatBoost Comparison Results Saved!")
