
# ==========================================
# STEP 17C.1: LOAD DATA FOR CATBOOST TUNING
# ==========================================

import pandas as pd
from pathlib import Path

# Project directories
project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "processed_data"

# Load training and validation datasets
train_df = pd.read_csv(data_folder / "cp_train.csv")
val_df = pd.read_csv(data_folder / "cp_validation.csv")

# Separate features and target
X_train = train_df.drop(columns=["is_canceled"])
y_train = train_df["is_canceled"]

X_val = val_df.drop(columns=["is_canceled"])
y_val = val_df["is_canceled"]

# Check feature consistency
assert X_train.columns.tolist() == X_val.columns.tolist(), (
    "Training and validation features do not match."
)

# Identify numerical and categorical features
numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    col for col in X_train.columns
    if col not in numerical_features
]

print("\nSTEP 17C.1: Load Data for CatBoost Tuning")

print("Training Features:", X_train.shape)
print("Training Target:", y_train.shape)

print("\nValidation Features:", X_val.shape)
print("Validation Target:", y_val.shape)

print("\nNumerical Features:", len(numerical_features))
print("Categorical Features:", len(categorical_features))
print("Total Features:", len(X_train.columns))

print("\nFeature Consistency Check: Passed!")







# ==========================================
# STEP 17C.2: PREPARE CATBOOST DATA AND MODEL
# ==========================================

from catboost import CatBoostClassifier

# Create copies to keep original datasets unchanged
X_train_cb = X_train.copy()
X_val_cb = X_val.copy()

# Fill missing numerical values using training medians
numeric_medians = X_train[numerical_features].median()

for col in numerical_features:
    X_train_cb[col] = X_train_cb[col].fillna(numeric_medians[col])
    X_val_cb[col] = X_val_cb[col].fillna(numeric_medians[col])

# Convert categorical features to strings
for col in categorical_features:
    X_train_cb[col] = X_train_cb[col].fillna("Unknown").astype(str)
    X_val_cb[col] = X_val_cb[col].fillna("Unknown").astype(str)

# Create CatBoost model using baseline settings
catboost_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    random_seed=42,
    verbose=False,
    allow_writing_files=False
)

print("\nSTEP 17C.2: CatBoost Data Preparation")

print("Training Shape:", X_train_cb.shape)
print("Validation Shape:", X_val_cb.shape)

print("Training Missing Values:", X_train_cb.isna().sum().sum())
print("Validation Missing Values:", X_val_cb.isna().sum().sum())

print("\nCatBoost Model Created Successfully!")






# ==========================================
# STEP 17C.3: CATBOOST HYPERPARAMETER TUNING
# ==========================================

from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
import time
import joblib
import json

# Define hyperparameter search space
catboost_params = {
    "iterations": [300, 500, 700],
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.08],
    "l2_leaf_reg": [3, 5, 7]
}

# Group identical booking features
train_groups = pd.util.hash_pandas_object(
    X_train, index=False
)

# Group-aware cross-validation
group_cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

print("\nSTEP 17C.3: CatBoost Hyperparameter Tuning")
print("Unique Training Groups:", train_groups.nunique())

# Randomized hyperparameter search
catboost_search = RandomizedSearchCV(
    estimator=catboost_model,
    param_distributions=catboost_params,
    n_iter=8,
    scoring="f1",
    cv=group_cv,
    random_state=42,
    n_jobs=1,
    verbose=1,
    refit=True,
    error_score="raise"
)

# Start tuning
start_time = time.perf_counter()


catboost_search.fit(
    X_train_cb,
    y_train,
    groups=train_groups,
    cat_features=categorical_features
)


tuning_time = time.perf_counter() - start_time

print("\nCatBoost Tuning Completed!")

print("\nBest Hyperparameters:")
print(catboost_search.best_params_)

print("\nBest Cross-Validation F1-score:",
      round(catboost_search.best_score_, 4))

print("\nTotal Tuning Time:",
      round(tuning_time, 3), "seconds")


# ------------------------------------------
# SAVE TUNING CHECKPOINT
# ------------------------------------------

results_folder = project_root / "cancellation_prediction" / "results"
models_folder = project_root / "cancellation_prediction" / "saved_models"

results_folder.mkdir(parents=True, exist_ok=True)
models_folder.mkdir(parents=True, exist_ok=True)

# Save model and preprocessing information
model_package = {
    "model": catboost_search.best_estimator_,
    "numeric_medians": numeric_medians.to_dict(),
    "categorical_features": categorical_features,
    "feature_columns": X_train.columns.tolist()
}

joblib.dump(
    model_package,
    models_folder / "catboost_tuned_v1.joblib"
)

# Save all cross-validation results
pd.DataFrame(catboost_search.cv_results_).to_csv(
    results_folder / "catboost_cv_results_v1.csv",
    index=False
)

# Save best parameters and tuning information
summary = {
    "model": "CatBoost",
    "version": "Tuned V1",
    "best_parameters": catboost_search.best_params_,
    "best_cv_f1": float(catboost_search.best_score_),
    "cv_method": "StratifiedGroupKFold",
    "cv_folds": 3,
    "n_iter": 8,
    "tuning_time_seconds": float(tuning_time)
}

with open(
    results_folder / "catboost_tuning_summary_v1.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

print("\nCatBoost Tuning Files Saved Successfully!")
