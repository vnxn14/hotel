
import pandas as pd
from pathlib import Path

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "processed_data" / "0_base_clean.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ==========================================
# STEP 2: DATASET INFORMATION
# ==========================================

print("\nDataset Information:")
df.info()

# ==========================================
# STEP 3: MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False))

# ==========================================
# STEP 4: DUPLICATE RECORDS
# ==========================================

print("\nDuplicate Records:")
print(df.duplicated().sum())

# ==========================================
# STEP 5: TARGET DISTRIBUTION
# ==========================================

print("\nCancellation Counts:")
print(df["is_canceled"].value_counts())

print("\nCancellation Percentage:")
print(
    df["is_canceled"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ==========================================
# STEP 6: LIST ALL COLUMNS
# ==========================================

print("\nAvailable Columns:")
for column in df.columns:
    print(column)



# ==========================================
# STEP 7: REMOVE DATA LEAKAGE COLUMNS
# ==========================================

leakage_columns = [
    "reservation_status",
    "reservation_status_date",
    "booking_changes",
    "assigned_room_type",
    "days_in_waiting_list"
]

df_cp = df.drop(columns=leakage_columns).copy()

print("\nShape After Leakage Removal:")
print(df_cp.shape)


# ==========================================
# STEP 8: FEATURE ENGINEERING
# ==========================================

# 1. Has Children / Babies
df_cp["has_children"] = (
    (df_cp["children"] + df_cp["babies"]) > 0
).astype(int)

# 2. Previous Booking History
total_previous_bookings = (
    df_cp["previous_cancellations"]
    + df_cp["previous_bookings_not_canceled"]
)

df_cp["has_booking_history"] = (
    total_previous_bookings > 0
).astype(int)

# 3. Historical Cancellation Rate
df_cp["historical_cancellation_rate"] = (
    df_cp["previous_cancellations"]
    / total_previous_bookings.replace(0, float("nan"))
).fillna(0)

# 4. Has Agent
df_cp["has_agent"] = df_cp["agent"].notna().astype(int)

# 5. Has Company
df_cp["has_company"] = df_cp["company"].notna().astype(int)

# 6. Weekend Stay Ratio
df_cp["weekend_stay_ratio"] = (
    df_cp["stays_in_weekend_nights"]
    / df_cp["total_stay_nights"]
)

# Remove Original Agent and Company IDs
df_cp = df_cp.drop(columns=["agent", "company"])

print("\nShape After Feature Engineering:")
print(df_cp.shape)

print("\nNew Engineered Features:")
print(
    df_cp[
        [
            "has_children",
            "has_booking_history",
            "historical_cancellation_rate",
            "has_agent",
            "has_company",
            "weekend_stay_ratio"
        ]
    ].head()
)




# ==========================================
# STEP 9: FEATURE AVAILABILITY AUDIT
# ==========================================

review_columns = [
    "adr",
    "total_of_special_requests",
    "required_car_parking_spaces",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_month_num",
    "lead_time",
    "lead_time_category",
    "booking_value"
]

print("\nFeatures Requiring Review:")

for column in review_columns:
    if column in df_cp.columns:
        print(
            f"{column}: "
            f"{df_cp[column].nunique()} unique values, "
            f"{df_cp[column].isnull().sum()} missing"
        )

print("\nCurrent Columns:")
print(df_cp.columns.tolist())




# ==========================================
# STEP 10: INITIAL FEATURE SELECTION
# ==========================================

selected_features = [
    # Hotel Information
    "hotel",

    # Booking Information
    "lead_time",
    "deposit_type",
    "market_segment",
    "distribution_channel",
    "customer_type",

    # Arrival Information
    "arrival_month_num",
    "arrival_date_day_of_month",
    "arrival_season",
    "is_weekend_arrival",

    # Stay Information
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "total_stay_nights",
    "weekend_stay_ratio",

    # Guest Information
    "adults",
    "children",
    "babies",
    "total_guests",
    "has_children",
    "country",
    "meal",

    # Booking History
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "has_booking_history",
    "historical_cancellation_rate",

    # Other Information
    "reserved_room_type",
    "has_agent",
    "has_company"
]

# Input Features
X = df_cp[selected_features].copy()

# Target Variable
y = df_cp["is_canceled"].copy()

print("\nSelected Feature Count:")
print(X.shape[1])

print("\nSelected Dataset Shape:")
print(X.shape)

print("\nSelected Features:")
print(X.columns.tolist())

print("\nMissing Values After Feature Selection:")
print(X.isnull().sum().sum())

print("\nTarget Shape:")
print(y.shape)





# ==========================================
# STEP 11: TRAIN / VALIDATION / TEST SPLIT
# ==========================================

from sklearn.model_selection import GroupShuffleSplit

# Group bookings with identical input features
groups = pd.util.hash_pandas_object(X, index=False)

# Split 1: Training (approximately 70%) + Temporary (30%)
split_1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, temp_idx = next(
    split_1.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
y_train = y.iloc[train_idx].copy()

X_temp = X.iloc[temp_idx].copy()
y_temp = y.iloc[temp_idx].copy()

# Split 2: Divide Temporary Set into Validation and Testing
temp_groups = groups.iloc[temp_idx]

split_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_idx, test_idx = next(
    split_2.split(X_temp, y_temp, groups=temp_groups)
)

X_val = X_temp.iloc[val_idx].copy()
y_val = y_temp.iloc[val_idx].copy()

X_test = X_temp.iloc[test_idx].copy()
y_test = y_temp.iloc[test_idx].copy()

# ==========================================
# CHECK SPLITTING RESULTS
# ==========================================

print("\nData Splitting Results:")

for name, X_part, y_part in [
    ("Training", X_train, y_train),
    ("Validation", X_val, y_val),
    ("Testing", X_test, y_test)
]:
    print(
        f"{name}: {len(X_part)} records "
        f"({len(X_part) / len(X) * 100:.2f}%), "
        f"Cancellation Rate: {y_part.mean() * 100:.2f}%"
    )

# Check for identical feature group overlap
train_groups = set(groups.iloc[train_idx])
val_groups = set(temp_groups.iloc[val_idx])
test_groups = set(temp_groups.iloc[test_idx])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

print("\nGroup Overlap Check: Passed")



# ==========================================
# STEP 12A: IDENTIFY FEATURE TYPES
# ==========================================

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

print("\nTotal Categorical Features:")
print(len(categorical_features))

print("\nNumerical Features:")
print(numerical_features)

print("\nTotal Numerical Features:")
print(len(numerical_features))

print("\nTotal Features:")
print(len(categorical_features) + len(numerical_features))




# ==========================================
# STEP 12B: MODEL-SPECIFIC PREPROCESSING
# ==========================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------
# Pipeline A: Logistic Regression + ANN
# ------------------------------------------

numerical_scaled = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_encoded = Pipeline([
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Unknown"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor_lr_ann = ColumnTransformer([
    ("num", numerical_scaled, numerical_features),
    ("cat", categorical_encoded, categorical_features)
])

# ------------------------------------------
# Pipeline B: Random Forest
# ------------------------------------------

numerical_unscaled = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor_rf = ColumnTransformer([
    ("num", numerical_unscaled, numerical_features),
    ("cat", categorical_encoded, categorical_features)
])

# ------------------------------------------
# Pipeline C: CatBoost
# ------------------------------------------

# CatBoost will receive categorical columns as
# strings, rather than One-Hot Encoded columns.

def prepare_catboost_data(data, numeric_fill_values):
    result = data.copy()

    for col in categorical_features:
        result[col] = result[col].fillna("Unknown").astype(str)

    for col in numerical_features:
        result[col] = result[col].fillna(numeric_fill_values[col])

    return result


print("\nModel-Specific Preprocessors Created:")
print("Pipeline A: Logistic Regression + ANN")
print("Pipeline B: Random Forest")
print("Pipeline C: CatBoost")



# ==========================================
# STEP 13: TEST PREPROCESSING PIPELINES
# ==========================================

# Logistic Regression + ANN
X_train_scaled = preprocessor_lr_ann.fit_transform(X_train)
X_val_scaled = preprocessor_lr_ann.transform(X_val)

# Random Forest
X_train_rf = preprocessor_rf.fit_transform(X_train)
X_val_rf = preprocessor_rf.transform(X_val)

# CatBoost
numeric_fill_values = X_train[numerical_features].median()

X_train_catboost = prepare_catboost_data(
    X_train, numeric_fill_values
)

X_val_catboost = prepare_catboost_data(
    X_val, numeric_fill_values
)

print("\nPreprocessing Test Results:")

print("LR / ANN Training Shape:", X_train_scaled.shape)
print("LR / ANN Validation Shape:", X_val_scaled.shape)

print("Random Forest Training Shape:", X_train_rf.shape)
print("Random Forest Validation Shape:", X_val_rf.shape)

print("CatBoost Training Shape:", X_train_catboost.shape)
print("CatBoost Validation Shape:", X_val_catboost.shape)

print("\nAll Preprocessing Pipelines Tested Successfully!")




# ==========================================
# STEP 14: SAVE SPLIT DATASETS
# ==========================================

output_folder = project_root / "processed_data"
output_folder.mkdir(parents=True, exist_ok=True)

# Include target column in each dataset
train_data = X_train.copy()
train_data["is_canceled"] = y_train

val_data = X_val.copy()
val_data["is_canceled"] = y_val

test_data = X_test.copy()
test_data["is_canceled"] = y_test

# Save CSV files
train_data.to_csv(
    output_folder / "cp_train.csv",
    index=False
)

val_data.to_csv(
    output_folder / "cp_validation.csv",
    index=False
)

test_data.to_csv(
    output_folder / "cp_test.csv",
    index=False
)

print("\nCancellation Prediction Datasets Saved!")
print("Location:", output_folder)
print("Training:", train_data.shape)
print("Validation:", val_data.shape)
print("Testing:", test_data.shape)
