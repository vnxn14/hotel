
# ==========================================
# STEP 17D.1: LOAD DATA FOR ANN TUNING
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

# Identify feature types
numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    col for col in X_train.columns
    if col not in numerical_features
]

print("\nSTEP 17D.1: Load Data for ANN Tuning")

print("Training Features:", X_train.shape)
print("Training Target:", y_train.shape)

print("\nValidation Features:", X_val.shape)
print("Validation Target:", y_val.shape)

print("\nNumerical Features:", len(numerical_features))
print("Categorical Features:", len(categorical_features))
print("Total Features:", len(X_train.columns))

print("\nFeature Consistency Check: Passed!")




# ==========================================
# STEP 17D.2: ANN DATA PREPROCESSING
# ==========================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Numerical features: fill missing values and scale
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical features: fill missing values and encode
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
ann_preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Fit preprocessing on training data only
X_train_ann = ann_preprocessor.fit_transform(X_train)
X_val_ann = ann_preprocessor.transform(X_val)

# Convert data to float32 for TensorFlow
X_train_ann = X_train_ann.astype("float32")
X_val_ann = X_val_ann.astype("float32")

y_train_ann = y_train.to_numpy(dtype="float32")
y_val_ann = y_val.to_numpy(dtype="float32")

print("\nSTEP 17D.2: ANN Data Preprocessing")

print("Training Shape:", X_train_ann.shape)
print("Validation Shape:", X_val_ann.shape)

print("Training Missing Values:",
      pd.isna(X_train_ann).sum())

print("Validation Missing Values:",
      pd.isna(X_val_ann).sum())

print("\nANN Data Preprocessing Completed!")





# ==========================================
# STEP 17D.3: BUILD ANN MODEL AND DEFINE HYPERPARAMETERS
# ==========================================

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(42)

# Function to build ANN with different hyperparameters
def build_ann(input_dim, hidden_units, dropout_rate, learning_rate):
    model = Sequential()

    model.add(Input(shape=(input_dim,)))

    # Hidden layers
    for units in hidden_units:
        model.add(Dense(units, activation="relu"))

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Output layer for binary classification
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Hyperparameter combinations to test
ann_configs = [
    {
        "hidden_units": (64, 32),
        "dropout_rate": 0.0,
        "learning_rate": 0.001,
        "batch_size": 64
    },
    {
        "hidden_units": (64, 32),
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 64
    },
    {
        "hidden_units": (128, 64),
        "dropout_rate": 0.2,
        "learning_rate": 0.0005,
        "batch_size": 64
    },
    {
        "hidden_units": (64, 32),
        "dropout_rate": 0.1,
        "learning_rate": 0.0005,
        "batch_size": 128
    }
]

# Test model creation using the baseline configuration
config = ann_configs[0]

ann_model = build_ann(
    input_dim=X_train_ann.shape[1],
    hidden_units=config["hidden_units"],
    dropout_rate=config["dropout_rate"],
    learning_rate=config["learning_rate"]
)

print("\nSTEP 17D.3: ANN Model Configuration")
print("Input Features:", X_train_ann.shape[1])
print("Hyperparameter Combinations:", len(ann_configs))
print("Model Parameters:", ann_model.count_params())
print("\nANN Model Created Successfully!")






# ==========================================
# STEP 17D.4: ANN HYPERPARAMETER TUNING
# ==========================================

import numpy as np
import json
import time
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)

# Group identical feature combinations
train_groups = pd.util.hash_pandas_object(
    X_train, index=False
)

# Split training data for internal validation
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

inner_train_idx, inner_val_idx = next(
    splitter.split(X_train, y_train, groups=train_groups)
)

X_inner_train = X_train.iloc[inner_train_idx]
X_inner_val = X_train.iloc[inner_val_idx]

y_inner_train = y_train.iloc[inner_train_idx].to_numpy(dtype="float32")
y_inner_val = y_train.iloc[inner_val_idx].to_numpy(dtype="float32")

# Fit preprocessing on internal training data only
inner_preprocessor = clone(ann_preprocessor)

X_inner_train_ann = inner_preprocessor.fit_transform(
    X_inner_train
).astype("float32")

X_inner_val_ann = inner_preprocessor.transform(
    X_inner_val
).astype("float32")

print("\nSTEP 17D.4: ANN Hyperparameter Tuning")
print("Internal Training:", X_inner_train_ann.shape)
print("Internal Validation:", X_inner_val_ann.shape)
print("Training Cancellation Rate:", round(y_inner_train.mean(), 4))
print("Validation Cancellation Rate:", round(y_inner_val.mean(), 4))

# Store experiment results
trial_results = []

for i, config in enumerate(ann_configs, start=1):

    print(f"\nTraining ANN Configuration {i}...")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    model = build_ann(
        input_dim=X_inner_train_ann.shape[1],
        hidden_units=config["hidden_units"],
        dropout_rate=config["dropout_rate"],
        learning_rate=config["learning_rate"]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    start_time = time.perf_counter()

    history = model.fit(
        X_inner_train_ann,
        y_inner_train,
        validation_data=(X_inner_val_ann, y_inner_val),
        epochs=35,
        batch_size=config["batch_size"],
        callbacks=[early_stopping],
        verbose=0
    )

    training_time = time.perf_counter() - start_time

    # Evaluate using internal validation data
    probabilities = model.predict(
        X_inner_val_ann, verbose=0
    ).ravel()

    predictions = (probabilities >= 0.5).astype(int)

    result = {
        "config_id": i,
        "hidden_units": str(config["hidden_units"]),
        "dropout_rate": config["dropout_rate"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "epochs_trained": len(history.history["loss"]),
        "F1-score": float(f1_score(y_inner_val, predictions)),
        "Precision": float(precision_score(y_inner_val, predictions)),
        "Recall": float(recall_score(y_inner_val, predictions)),
        "PR-AUC": float(average_precision_score(
            y_inner_val, probabilities
        )),
        "training_time": round(training_time, 3)
    }

    trial_results.append(result)

    print("F1-score:", round(result["F1-score"], 4))
    print("Precision:", round(result["Precision"], 4))
    print("Recall:", round(result["Recall"], 4))
    print("Epochs Trained:", result["epochs_trained"])

# Compare all configurations
results_df = pd.DataFrame(trial_results)

print("\nANN Hyperparameter Tuning Results:")
print(results_df.to_string(index=False))

# Select configuration with highest internal validation F1
best_result = max(
    trial_results,
    key=lambda result: result["F1-score"]
)

best_config = ann_configs[best_result["config_id"] - 1]

print("\nBest ANN Configuration:", best_result["config_id"])
print("Best Hyperparameters:", best_config)
print("Best Internal Validation F1-score:",
      round(best_result["F1-score"], 4))

# Save experiment results
results_folder = project_root / "cancellation_prediction" / "results"
results_folder.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    results_folder / "ann_tuning_results_v1.csv",
    index=False
)

# Save selected configuration for later final training
summary = {
    "best_config": best_config,
    "best_epoch": best_result["best_epoch"],
    "internal_validation_f1": best_result["F1-score"]
}

with open(
    results_folder / "ann_tuning_summary_v1.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

print("\nANN Tuning Results Saved Successfully!")



