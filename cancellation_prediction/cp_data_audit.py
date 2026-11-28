
# ==========================================
# LOAD DATASETS FOR DATA AUDIT
# ==========================================

import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "processed_data"

# Load Training, Validation and Testing Data
train_df = pd.read_csv(data_folder / "cp_train.csv")
val_df = pd.read_csv(data_folder / "cp_validation.csv")
test_df = pd.read_csv(data_folder / "cp_test.csv")

# Separate Features and Target
X_train = train_df.drop(columns=["is_canceled"])
X_val = val_df.drop(columns=["is_canceled"])
X_test = test_df.drop(columns=["is_canceled"])


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






# ==========================================
# STEP 16B: FEATURE AVAILABILITY AUDIT
# ==========================================

print("\nSTEP 16B: Feature Availability Audit")

print("\nTotal Selected Features:", len(X_train.columns))

print("\nSelected Features:")
for index, feature in enumerate(X_train.columns, start=1):
    print(f"{index}. {feature}")

print("\nFeature Data Types:")
print(X_train.dtypes)

print("\nFeature Missing Values:")
print(X_train.isna().sum())

# Check duplicate feature names
print("\nDuplicate Feature Names:",
      X_train.columns.duplicated().sum())

# Check that target is not included in X
print("\nTarget Leakage Check:",
      "is_canceled" in X_train.columns)




# ==========================================
# STEP 16B.2: DATA LEAKAGE AUDIT
# ==========================================

# Check prohibited post-booking features
leakage_features = [
    "reservation_status",
    "reservation_status_date",
    "booking_changes",
    "assigned_room_type",
    "days_in_waiting_list"
]

print("\nSTEP 16B.2: Data Leakage Audit")

for name, data in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:
    found = [
        col for col in leakage_features
        if col in data.columns
    ]

    print(f"\n{name} Dataset")
    print("Prohibited Features Found:", found)
    print("Total Features:", data.shape[1] - 1)
    print("Missing Values:", data.isna().sum().sum())

# Inspect historical features
historical_features = [
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "has_booking_history",
    "historical_cancellation_rate"
]

print("\nHistorical Feature Summary:")
print(train_df[historical_features].describe().round(3))

# Check impossible cancellation rates
invalid_rates = (
    (train_df["historical_cancellation_rate"] < 0) |
    (train_df["historical_cancellation_rate"] > 1)
).sum()

print("\nInvalid Historical Cancellation Rates:",
      invalid_rates)





# ==========================================
# STEP 16C: FINAL DATA VALIDATION
# ==========================================
print("\n")
print("           Final Data Validation          ")
print("------------------------------------------")

# ------------------------------------------
# 1. COUNTRY CODE INTEGRITY
# ------------------------------------------

print("\nCountry Code Integrity:")

base_file = data_folder / "0_base_clean.csv"

if base_file.exists():
    base_df = pd.read_csv(
        base_file,
        keep_default_na=False
    )

    print(
        "Original 'NA' Country Code Count:",
        (base_df["country"] == "NA").sum()
    )

    print(
        "Original Empty Country Values:",
        (base_df["country"] == "").sum()
    )

    print(
        "Processed Training 'NA' Count:",
        (X_train["country"] == "NA").sum()
    )

    print(
        "Processed Validation 'NA' Count:",
        (X_val["country"] == "NA").sum()
    )

    print(
        "Processed Testing 'NA' Count:",
        (X_test["country"] == "NA").sum()
    )
else:
    print("Base dataset not found.")

# ------------------------------------------
# 2. DATA SPLIT INTEGRITY
# ------------------------------------------

print("\nData Split Integrity:")

# Create row fingerprints from feature values
train_hashes = set(
    pd.util.hash_pandas_object(
        X_train,
        index=False
    )
)

val_hashes = set(
    pd.util.hash_pandas_object(
        X_val,
        index=False
    )
)

test_hashes = set(
    pd.util.hash_pandas_object(
        X_test,
        index=False
    )
)

print(
    "Train / Validation Overlap:",
    len(train_hashes & val_hashes)
)

print(
    "Train / Test Overlap:",
    len(train_hashes & test_hashes)
)

print(
    "Validation / Test Overlap:",
    len(val_hashes & test_hashes)
)





# ==========================================
# STEP 16D: COUNTRY CODE PROVENANCE
# ==========================================

print("\nSTEP 16D: Country Code Provenance")

if base_file.exists():
    print("\nBase Country Distribution:")
    print(base_df["country"].value_counts(dropna=False).head(15))

    print("\nBase Country Values:")
    print("String 'NA':", (base_df["country"] == "NA").sum())
    print("String 'Unknown':", (base_df["country"] == "Unknown").sum())
    print("String 'nan':", (base_df["country"] == "nan").sum())
    print("Actual Missing:", base_df["country"].isna().sum())
else:
    print("Base dataset not found; country provenance not checked.")



# ==========================================
# CHECK TOTAL UNIQUE COUNTRY CODES
# ==========================================

print("\nTotal Unique Country Codes:")
print(base_df["country"].nunique())

print("\nTotal Country Values:")
print(base_df["country"].value_counts(dropna=False).sum())

print("\nAll Country Codes:")
print(sorted(base_df["country"].unique()))




# ==========================================
# STEP 16E.1: RAW COUNTRY VERIFICATION
# ==========================================

print("\nSTEP 16E.1: Raw Country Verification")

raw_file = project_root / "hotel_bookings.csv"

if raw_file.exists():
    raw_country = pd.read_csv(
        raw_file,
        usecols=["country"],
        keep_default_na=False
    )["country"]

    print("Raw Dataset Records:", len(raw_country))
    print("Raw 'NA' Count:", (raw_country == "NA").sum())
    print("Raw Empty Country Count:", (raw_country == "").sum())
    print("Raw 'NULL' Count:", (raw_country == "NULL").sum())
    print("Raw 'Undefined' Count:", (raw_country == "Undefined").sum())
    print("Raw Unique Country Values:", raw_country.nunique())
else:
    print("Raw CSV not found:", raw_file)
