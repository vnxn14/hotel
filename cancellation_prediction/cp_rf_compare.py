
# ==========================================
# STEP 17B.6: COMPARE BASELINE VS TUNED RANDOM FOREST
# ==========================================

import pandas as pd
import joblib
from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "processed_data"
model_folder = project_root / "cancellation_prediction" / "saved_models"
results_folder = project_root / "cancellation_prediction" / "results"

# Load training and validation data
train_df = pd.read_csv(data_folder / "cp_train.csv")
val_df = pd.read_csv(data_folder / "cp_validation.csv")

X_train = train_df.drop(columns=["is_canceled"])
y_train = train_df["is_canceled"]

X_val = val_df.drop(columns=["is_canceled"])
y_val = val_df["is_canceled"]

# Load the saved tuned model
tuned_rf = joblib.load(model_folder / "rf_tuned_v1.joblib")

# Create baseline with the same preprocessing
baseline_rf = clone(tuned_rf)

baseline_rf.set_params(
    classifier=RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
)

print("\nSTEP 17B.6: Random Forest Model Comparison")

# Train baseline only
baseline_rf.fit(X_train, y_train)

# Evaluate both models on the same validation set
def evaluate_model(model, name):
    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(y_val, predictions),
        "Recall": recall_score(y_val, predictions),
        "F1-score": f1_score(y_val, predictions),
        "ROC-AUC": roc_auc_score(y_val, probabilities),
        "PR-AUC": average_precision_score(y_val, probabilities)
    }

comparison = pd.DataFrame([
    evaluate_model(baseline_rf, "Baseline"),
    evaluate_model(tuned_rf, "Tuned V1")
])

print("\nRandom Forest Comparison:")
print(comparison.to_string(index=False, float_format="%.4f"))

# Save results for the report
comparison.to_csv(
    results_folder / "rf_baseline_vs_tuned.csv",
    index=False
)

print("\nComparison Results Saved!")
