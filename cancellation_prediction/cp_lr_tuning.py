
# ==========================================
# STEP 17A.1: LOAD DATA FOR HYPERPARAMETER TUNING
# ==========================================

import pandas as pd
from pathlib import Path

# Project directories
project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "processed_data"

# Load Training and Validation datasets
train_df = pd.read_csv(data_folder / "cp_train.csv")
val_df = pd.read_csv(data_folder / "cp_validation.csv")

# Separate features and target
X_train = train_df.drop(columns=["is_canceled"])
y_train = train_df["is_canceled"]

X_val = val_df.drop(columns=["is_canceled"])
y_val = val_df["is_canceled"]

# Display dataset information
print("\nSTEP 17A.1: Load Data for Hyperparameter Tuning")

print("Training Features:", X_train.shape)
print("Training Target:", y_train.shape)

print("\nValidation Features:", X_val.shape)
print("Validation Target:", y_val.shape)

print("\nTraining Cancellation Rate:",
      round(y_train.mean() * 100, 2), "%")

print("Validation Cancellation Rate:",
      round(y_val.mean() * 100, 2), "%")





# ==========================================
# STEP 17A.2: LOGISTIC REGRESSION PIPELINE
# ==========================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Identify categorical and numerical features
categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()


numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

# Numerical preprocessing: imputation + scaling
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing: imputation + one-hot encoding
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Unknown"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# Combine preprocessing and Logistic Regression
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

print("\nSTEP 17A.2: Logistic Regression Pipeline")

print("Categorical Features:", len(categorical_features))
print("Numerical Features:", len(numerical_features))
print("Total Features:",
      len(categorical_features) + len(numerical_features))

print("\nPipeline Created Successfully!")





# ==========================================
# STEP 17A.3: LOGISTIC REGRESSION HYPERPARAMETER TUNING
# ==========================================

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold
)
import time

# Define hyperparameter search space

# Refined hyperparameter search space (Tuning V2)
lr_param_distributions = {
    "classifier__C": [
        0.03, 0.1, 0.3, 1, 3, 10
    ],
    "classifier__class_weight": [
        None,
        {0: 1, 1: 1.1},
        {0: 1, 1: 1.25},
        {0: 1, 1: 1.4},
        {0: 1, 1: 1.6},
        "balanced"
    ]
}


print("\nSTEP 17A.3: Logistic Regression Hyperparameter Tuning")




# ==========================================
# GROUP-AWARE CROSS-VALIDATION
# ==========================================

# Identical feature combinations belong to the same group
train_groups = pd.util.hash_pandas_object(
    X_train,
    index=False
)

# Stratified Group 3-Fold Cross-Validation
group_cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

print("\nUnique Training Groups:", train_groups.nunique())
print("Cross-Validation: StratifiedGroupKFold (3 Folds)")



# Randomized Search with 3-Fold Cross-Validation
lr_random_search = RandomizedSearchCV(
    estimator=lr_pipeline,
    param_distributions=lr_param_distributions,
    n_iter=24,
    scoring="f1",
    cv=group_cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True,
    error_score="raise"
)

# Measure tuning time
start_time = time.perf_counter()

lr_random_search.fit(
    X_train,
    y_train,
    groups=train_groups
)

tuning_time = time.perf_counter() - start_time

# Display tuning results
print("\nLogistic Regression Tuning Completed!")

print("\nBest Hyperparameters:")
print(lr_random_search.best_params_)

print("\nBest Cross-Validation F1-score:",
      round(lr_random_search.best_score_, 4))

print("\nTotal Tuning Time:",
      round(tuning_time, 3), "seconds")




# ==========================================
# STEP 17A.4: EVALUATE TUNED LOGISTIC REGRESSION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

print("\nSTEP 17A.4: Tuned Logistic Regression Evaluation")

# Get best tuned model
best_lr_model = lr_random_search.best_estimator_

# Measure validation inference time
start_time = time.perf_counter()

# Predict cancellation probabilities
y_val_proba = best_lr_model.predict_proba(X_val)[:, 1]

# Use the same 0.5 threshold as the baseline
y_val_pred = (y_val_proba >= 0.5).astype(int)

inference_time = time.perf_counter() - start_time

# Calculate evaluation metrics
tuned_results = {
    "Accuracy": accuracy_score(y_val, y_val_pred),
    "Precision": precision_score(
        y_val, y_val_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_val, y_val_pred, zero_division=0
    ),
    "F1-score": f1_score(
        y_val, y_val_pred, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(y_val, y_val_proba),
    "PR-AUC": average_precision_score(
        y_val, y_val_proba
    )
}

print("\nTuned Logistic Regression Results:")

for metric, value in tuned_results.items():
    print(f"{metric}: {value:.4f}")

print("\nValidation Inference Time:",
      round(inference_time, 3), "seconds")






# ==========================================
# STEP 17A.5: SAVE LOGISTIC REGRESSION TUNING RESULTS
# ==========================================

import json
import joblib

print("\nSTEP 17A.5: Save Logistic Regression Tuning Results")

# Create directories for models and results
results_folder = project_root / "cancellation_prediction" / "results"
models_folder = project_root / "cancellation_prediction" / "saved_models"

results_folder.mkdir(parents=True, exist_ok=True)
models_folder.mkdir(parents=True, exist_ok=True)

# 1. Save tuned V2 model with preprocessing
model_path = models_folder / "lr_tuned_v2.joblib"

joblib.dump(
    best_lr_model,
    model_path
)

# 2. Save all hyperparameter search results
cv_results_df = pd.DataFrame(
    lr_random_search.cv_results_
)

cv_results_path = results_folder / "lr_cv_results_v2.csv"

cv_results_df.to_csv(
    cv_results_path,
    index=False
)

# 3. Save model performance summary
summary = {
    "model": "Logistic Regression",
    "version": "Tuned V2",
    "search_method": "RandomizedSearchCV",
    "cross_validation": "StratifiedGroupKFold",
    "cv_folds": 3,
    "n_iter": 24,
    "best_parameters": lr_random_search.best_params_,
    "best_cv_f1": float(lr_random_search.best_score_),
    "validation_metrics": {
        metric: float(value)
        for metric, value in tuned_results.items()
    },
    "tuning_time_seconds": float(tuning_time),
    "inference_time_seconds": float(inference_time)
}

summary_path = results_folder / "lr_tuning_summary_v2.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("\nFiles Saved Successfully!")
print("Model:", model_path.name)
print("CV Results:", cv_results_path.name)
print("Summary:", summary_path.name)






# ==========================================
# STEP 17A.6: COMPARE LOGISTIC REGRESSION VERSIONS
# ==========================================

baseline_f1 = 0.6616
tuned_v1_f1 = 0.6867
tuned_v2_f1 = tuned_results["F1-score"]

print("\nLogistic Regression F1-score Comparison:")
print(f"Baseline: {baseline_f1:.4f}")
print(f"Tuned V1: {tuned_v1_f1:.4f}")
print(f"Tuned V2: {tuned_v2_f1:.4f}")

print("\nF1-score Improvement vs Baseline:",
      round((tuned_v2_f1 - baseline_f1) * 100, 2),
      "percentage points")

print("F1-score Improvement vs Tuned V1:",
      round((tuned_v2_f1 - tuned_v1_f1) * 100, 2),
      "percentage points")
