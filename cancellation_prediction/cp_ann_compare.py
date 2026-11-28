
# ==========================================
# STEP 17D.5.1: LOAD BEST ANN CONFIGURATION
# ==========================================

import json
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "processed_data"
results_folder = project_root / "cancellation_prediction" / "results"

# Load training and validation data
train_df = pd.read_csv(data_folder / "cp_train.csv")
val_df = pd.read_csv(data_folder / "cp_validation.csv")

X_train = train_df.drop(columns=["is_canceled"])
y_train = train_df["is_canceled"]

X_val = val_df.drop(columns=["is_canceled"])
y_val = val_df["is_canceled"]

# Load best configuration from tuning results
with open(
    results_folder / "ann_tuning_summary_v1.json",
    "r",
    encoding="utf-8"
) as file:
    tuning_summary = json.load(file)

best_config = tuning_summary["best_config"]
best_epoch = tuning_summary["best_epoch"]

print("\nSTEP 17D.5: Selected ANN Configuration")
print("Best Hyperparameters:", best_config)
print("Selected Training Epochs:", best_epoch)

print("\nTraining Features:", X_train.shape)
print("Validation Features:", X_val.shape)




# ==========================================
# STEP 17D.5.2: TRAIN AND EVALUATE SELECTED ANN
# ==========================================

import time
import joblib
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# Identify feature types
numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    col for col in X_train.columns
    if col not in numerical_features
]

# Create preprocessing pipelines
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

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

ann_preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Fit preprocessing on training data only
X_train_ann = ann_preprocessor.fit_transform(
    X_train
).astype("float32")

X_val_ann = ann_preprocessor.transform(
    X_val
).astype("float32")

# Set random seed
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(42)

# Build selected ANN
model = Sequential()
model.add(Input(shape=(X_train_ann.shape[1],)))

for units in best_config["hidden_units"]:
    model.add(Dense(units, activation="relu"))

    if best_config["dropout_rate"] > 0:
        model.add(Dropout(best_config["dropout_rate"]))

model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=best_config["learning_rate"]
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nSTEP 17D.5.2: Train Selected ANN")

# Train using the selected number of epochs
start_time = time.perf_counter()

model.fit(
    X_train_ann,
    y_train.to_numpy(dtype="float32"),
    epochs=best_epoch,
    batch_size=best_config["batch_size"],
    verbose=1
)

training_time = time.perf_counter() - start_time

# Predict on validation data
start_time = time.perf_counter()

y_proba = model.predict(
    X_val_ann, verbose=0
).ravel()

y_pred = (y_proba >= 0.5).astype(int)

inference_time = time.perf_counter() - start_time

# Calculate evaluation metrics
results = {
    "Accuracy": accuracy_score(y_val, y_pred),
    "Precision": precision_score(y_val, y_pred),
    "Recall": recall_score(y_val, y_pred),
    "F1-score": f1_score(y_val, y_pred),
    "ROC-AUC": roc_auc_score(y_val, y_proba),
    "PR-AUC": average_precision_score(y_val, y_proba)
}

print("\nSelected ANN Validation Results:")

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

print("\nTraining Time:", round(training_time, 3), "seconds")
print("Inference Time:", round(inference_time, 3), "seconds")

# Save model and preprocessing
models_folder = project_root / "cancellation_prediction" / "saved_models"
models_folder.mkdir(parents=True, exist_ok=True)

model.save(models_folder / "ann_selected_v1.keras")

joblib.dump(
    ann_preprocessor,
    models_folder / "ann_preprocessor_v1.joblib"
)

# Save validation results
pd.DataFrame([
    {"Model": "Selected ANN V1", **results}
]).to_csv(
    results_folder / "ann_validation_results_v1.csv",
    index=False
)

print("\nSelected ANN Model and Results Saved Successfully!")

