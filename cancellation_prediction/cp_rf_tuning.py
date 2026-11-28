
# ==========================================
# STEP 17B.1: LOAD DATA FOR RANDOM FOREST TUNING
# ==========================================

import pandas as pd
from pathlib import Path

# Define project directories
project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "processed_data"

# Load training and validation datasets
train_df = pd.read_csv(data_folder / "cp_train.csv")
val_df = pd.read_csv(data_folder / "cp_validation.csv")

# Separate input features and target variable
X_train = train_df.drop(columns=["is_canceled"])
y_train = train_df["is_canceled"]

X_val = val_df.drop(columns=["is_canceled"])
y_val = val_df["is_canceled"]

# Verify feature consistency
assert X_train.columns.tolist() == X_val.columns.tolist(), (
    "Training and validation features do not match."
)

# Display dataset information
print("\nSTEP 17B.1: Load Data for Random Forest Tuning")

print("\nTraining Features:", X_train.shape)
print("Training Target:", y_train.shape)

print("\nValidation Features:", X_val.shape)
print("Validation Target:", y_val.shape)

print("\nTraining Cancellation Rate:",
      round(y_train.mean() * 100, 2), "%")

print("Validation Cancellation Rate:",
      round(y_val.mean() * 100, 2), "%")

print("\nFeature Consistency Check: Passed!")






# ==========================================
# STEP 17B.2: RANDOM FOREST PIPELINE
# ==========================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Identify feature types
numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    col for col in X_train.columns
    if col not in numerical_features
]

# Numerical features
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# Categorical features
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

# Combine preprocessing steps
rf_preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Create Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocessor", rf_preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    ))
])

print("\nSTEP 17B.2: Random Forest Pipeline")
print("Numerical Features:", len(numerical_features))
print("Categorical Features:", len(categorical_features))
print("Total Features:", len(X_train.columns))
print("Pipeline Created Successfully!")





# ==========================================
# STEP 17B.3: RANDOM FOREST HYPERPARAMETER TUNING
# ==========================================

from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
import time

# Define hyperparameter search space
rf_param_distributions = {
    "classifier__n_estimators": [100, 150],
    "classifier__max_depth": [None, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__class_weight": [None, "balanced_subsample"]
}

# Group identical feature combinations
train_groups = pd.util.hash_pandas_object(
    X_train,
    index=False
)

# Group-aware cross-validation
group_cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

print("\nSTEP 17B.3: Random Forest Hyperparameter Tuning")
print("Unique Training Groups:", train_groups.nunique())

# Randomized hyperparameter search
rf_random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_param_distributions,
    n_iter=10,
    scoring="f1",
    cv=group_cv,
    random_state=42,
    n_jobs=2,
    pre_dispatch=2,
    verbose=1,
    refit=True,
    error_score="raise"
)

# Start tuning
start_time = time.perf_counter()

rf_random_search.fit(
    X_train,
    y_train,
    groups=train_groups
)

tuning_time = time.perf_counter() - start_time

# Display results
print("\nRandom Forest Tuning Completed!")

print("\nBest Hyperparameters:")
print(rf_random_search.best_params_)

print("\nBest Cross-Validation F1-score:",
      round(rf_random_search.best_score_, 4))

print("\nTotal Tuning Time:",
      round(tuning_time, 3), "seconds")






# ==========================================
# STEP 17B.4: EVALUATE TUNED RANDOM FOREST
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

print("\nSTEP 17B.4: Tuned Random Forest Evaluation")

# Get the best model from RandomizedSearchCV
best_rf_model = rf_random_search.best_estimator_

# Measure prediction time
start_time = time.perf_counter()

# Predict cancellation probability
y_val_proba = best_rf_model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_proba >= 0.5).astype(int)

inference_time = time.perf_counter() - start_time

# Evaluate model performance
tuned_results = {
    "Accuracy": accuracy_score(y_val, y_val_pred),
    "Precision": precision_score(y_val, y_val_pred, zero_division=0),
    "Recall": recall_score(y_val, y_val_pred, zero_division=0),
    "F1-score": f1_score(y_val, y_val_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_val, y_val_proba),
    "PR-AUC": average_precision_score(y_val, y_val_proba)
}

print("\nTuned Random Forest Results:")

for metric, value in tuned_results.items():
    print(f"{metric}: {value:.4f}")

print("\nValidation Inference Time:",
      round(inference_time, 3), "seconds")








# ==========================================
# STEP 17B.5: SAVE RANDOM FOREST TUNING RESULTS
# ==========================================

import json
import joblib

print("\nSTEP 17B.5: Save Random Forest Results")

# Create output folders
results_folder = project_root / "cancellation_prediction" / "results"
models_folder = project_root / "cancellation_prediction" / "saved_models"

results_folder.mkdir(parents=True, exist_ok=True)
models_folder.mkdir(parents=True, exist_ok=True)

# Save model with preprocessing
joblib.dump(
    best_rf_model,
    models_folder / "rf_tuned_v1.joblib"
)

# Save cross-validation results
pd.DataFrame(rf_random_search.cv_results_).to_csv(
    results_folder / "rf_cv_results_v1.csv",
    index=False
)

# Save tuning summary
summary = {
    "model": "Random Forest",
    "version": "Tuned V1",
    "best_parameters": rf_random_search.best_params_,
    "best_cv_f1": float(rf_random_search.best_score_),
    "validation_metrics": {
        metric: float(value)
        for metric, value in tuned_results.items()
    },
    "tuning_time_seconds": float(tuning_time),
    "inference_time_seconds": float(inference_time)
}

with open(
    results_folder / "rf_tuning_summary_v1.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

print("\nRandom Forest Files Saved Successfully!")
