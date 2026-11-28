
import pandas as pd
import time
from pathlib import Path

# ==========================================
# STEP 15A: LOAD TRAINING DATASETS
# ==========================================

project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "processed_data"

train_df = pd.read_csv(data_folder / "cp_train.csv")
val_df = pd.read_csv(data_folder / "cp_validation.csv")

# Separate features and target
X_train = train_df.drop(columns=["is_canceled"])
y_train = train_df["is_canceled"]

X_val = val_df.drop(columns=["is_canceled"])
y_val = val_df["is_canceled"]

print("Training Features:", X_train.shape)
print("Training Target:", y_train.shape)

print("\nValidation Features:", X_val.shape)
print("Validation Target:", y_val.shape)

print("\nTraining Cancellation Rate:", round(y_train.mean() * 100, 2), "%")
print("Validation Cancellation Rate:", round(y_val.mean() * 100, 2), "%")





# ==========================================
# STEP 15B: LOGISTIC REGRESSION BASELINE
# ==========================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# Identify feature types
categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

# Numerical: Imputation + Scaling
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical: Imputation + One-Hot Encoding
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Unknown"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Full training pipeline
lr_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Training Time
start = time.perf_counter()
lr_pipeline.fit(X_train, y_train)
lr_training_time = time.perf_counter() - start

# Execution Time
start = time.perf_counter()
y_prob = lr_pipeline.predict_proba(X_val)[:, 1]
lr_inference_time = time.perf_counter() - start

y_pred = (y_prob >= 0.5).astype(int)

# Evaluation
print("\nLogistic Regression Baseline Results:")
print("Accuracy:", round(accuracy_score(y_val, y_pred), 4))
print("Precision:", round(precision_score(y_val, y_pred), 4))
print("Recall:", round(recall_score(y_val, y_pred), 4))
print("F1-score:", round(f1_score(y_val, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_val, y_prob), 4))
print("PR-AUC:", round(average_precision_score(y_val, y_prob), 4))
print("Training Time:", round(lr_training_time, 3), "seconds")
print("Inference Time:", round(lr_inference_time, 3), "seconds")





# ==========================================
# STEP 15C: RANDOM FOREST BASELINE
# ==========================================

from sklearn.ensemble import RandomForestClassifier

# Random Forest Preprocessing
# One-Hot Encoding without StandardScaler

rf_preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Random Forest Pipeline
rf_pipeline = Pipeline([
    ("preprocessing", rf_preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# Training Time
start = time.perf_counter()
rf_pipeline.fit(X_train, y_train)
rf_training_time = time.perf_counter() - start

# Execution Time
start = time.perf_counter()
rf_prob = rf_pipeline.predict_proba(X_val)[:, 1]
rf_inference_time = time.perf_counter() - start

rf_pred = (rf_prob >= 0.5).astype(int)

# Evaluation
print("\nRandom Forest Baseline Results:")
print("Accuracy:", round(accuracy_score(y_val, rf_pred), 4))
print("Precision:", round(precision_score(y_val, rf_pred), 4))
print("Recall:", round(recall_score(y_val, rf_pred), 4))
print("F1-score:", round(f1_score(y_val, rf_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_val, rf_prob), 4))
print("PR-AUC:", round(average_precision_score(y_val, rf_prob), 4))
print("Training Time:", round(rf_training_time, 3), "seconds")
print("Inference Time:", round(rf_inference_time, 3), "seconds")





# ==========================================
# STEP 15D: CATBOOST BASELINE
# ==========================================

from catboost import CatBoostClassifier

# Prepare CatBoost datasets
X_train_cb = X_train.copy()
X_val_cb = X_val.copy()

# Handle categorical features
for col in categorical_features:
    X_train_cb[col] = X_train_cb[col].fillna("Unknown").astype(str)
    X_val_cb[col] = X_val_cb[col].fillna("Unknown").astype(str)

# Handle numerical missing values using training medians
train_medians = X_train[numerical_features].median()

X_train_cb[numerical_features] = (
    X_train_cb[numerical_features].fillna(train_medians)
)

X_val_cb[numerical_features] = (
    X_val_cb[numerical_features].fillna(train_medians)
)

# Initialize CatBoost
cb_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    random_seed=42,
    verbose=False
)

# Training Time
start = time.perf_counter()
cb_model.fit(
    X_train_cb,
    y_train,
    cat_features=categorical_features
)
cb_training_time = time.perf_counter() - start

# Inference Time
start = time.perf_counter()
cb_prob = cb_model.predict_proba(X_val_cb)[:, 1]
cb_inference_time = time.perf_counter() - start

cb_pred = (cb_prob >= 0.5).astype(int)

# Evaluation
print("\nCatBoost Baseline Results:")
print("Accuracy:", round(accuracy_score(y_val, cb_pred), 4))
print("Precision:", round(precision_score(y_val, cb_pred), 4))
print("Recall:", round(recall_score(y_val, cb_pred), 4))
print("F1-score:", round(f1_score(y_val, cb_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_val, cb_prob), 4))
print("PR-AUC:", round(average_precision_score(y_val, cb_prob), 4))
print("Training Time:", round(cb_training_time, 3), "seconds")
print("Inference Time:", round(cb_inference_time, 3), "seconds")




# ==========================================
# STEP 15E: ANN (MLP) BASELINE
# ==========================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.base import clone

# For reproducible results
keras.utils.set_random_seed(42)

# ------------------------------------------
# 1. PREPROCESS DATA FOR ANN
# ------------------------------------------

# Reuse LR preprocessing rules:
# One-Hot Encoding + StandardScaler
ann_preprocessor = clone(
    lr_pipeline.named_steps["preprocessing"]
)

X_train_ann = ann_preprocessor.fit_transform(X_train)
X_val_ann = ann_preprocessor.transform(X_val)

# Convert sparse matrices to dense arrays
if hasattr(X_train_ann, "toarray"):
    X_train_ann = X_train_ann.toarray()

if hasattr(X_val_ann, "toarray"):
    X_val_ann = X_val_ann.toarray()

# TensorFlow uses float32
X_train_ann = X_train_ann.astype(np.float32)
X_val_ann = X_val_ann.astype(np.float32)

print("\nANN Training Shape:", X_train_ann.shape)
print("ANN Validation Shape:", X_val_ann.shape)

# ------------------------------------------
# 2. BUILD ANN MODEL
# ------------------------------------------

ann_model = keras.Sequential([
    keras.Input(shape=(X_train_ann.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# ------------------------------------------
# 3. COMPILE MODEL
# ------------------------------------------

ann_model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------------------
# 4. EARLY STOPPING
# ------------------------------------------

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ------------------------------------------
# 5. TRAIN ANN MODEL
# ------------------------------------------

# Training Time
start = time.perf_counter()
ann_history = ann_model.fit(
    X_train_ann,
    y_train,
    validation_data=(X_val_ann, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)
ann_training_time = time.perf_counter() - start

# ------------------------------------------
# 6. VALIDATION PREDICTIONS
# ------------------------------------------

# Execution Time
start = time.perf_counter()
ann_prob = ann_model.predict(
    X_val_ann,
    verbose=0
).ravel()
ann_inference_time = time.perf_counter() - start

ann_pred = (ann_prob >= 0.5).astype(int)

# ------------------------------------------
# 7. EVALUATE ANN
# ------------------------------------------

print("\nANN Baseline Results:")
print("Accuracy:", round(accuracy_score(y_val, ann_pred), 4))
print("Precision:", round(precision_score(y_val, ann_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_val, ann_pred), 4))
print("F1-score:", round(f1_score(y_val, ann_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_val, ann_prob), 4))
print("PR-AUC:", round(average_precision_score(y_val, ann_prob), 4))

print("\nEpochs Trained:", len(ann_history.history["loss"]))
print("Training Time:", round(ann_training_time, 3), "seconds")
print("Inference Time:", round(ann_inference_time, 3), "seconds")





# ==========================================
# STEP 16A: CHECK CSV DATA INTEGRITY
# ==========================================

print("\nCSV Data Integrity Check:")

for name, data in [
    ("Training", X_train),
    ("Validation", X_val)
]:
    print(f"\n{name} Dataset:")
    print("Missing country values:", data["country"].isna().sum())
    print("Country 'NA' count:", (data["country"] == "NA").sum())
    print("Missing values by column:")
    print(data.isna().sum()[data.isna().sum() > 0])
