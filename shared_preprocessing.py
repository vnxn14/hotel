"""
Shared preprocessing - Hotel Booking Analytics (used by ALL three tasks).

Run once on the raw CSV. Every branch (cancellation risk, clustering, forecasting)
then reads the saved base file or imports load_and_clean_base().

    python shared_preprocessing.py                 # hotel_bookings.csv -> processed_data/0_base_clean.csv

    # inside a branch script:
    from shared_preprocessing import load_and_clean_base
    df = load_and_clean_base("hotel_bookings.csv")

Rules for this file:
  * Only rule-based, row-by-row cleaning (constants, formulas, fixed bins). Nothing is learned
    from the data (no scaler, encoder, mean imputation or quantile caps), so it is safe to run
    BEFORE any train/validation/test split.
  * Duplicates are NOT dropped here, because forecasting needs every booking.
    Cancellation risk and clustering drop them inside their own branch.
  * Leakage columns (reservation_status, booking_changes, ...) are NOT dropped here, so the
    base file can still be used for EDA and the dashboard. The cancellation branch removes them.
"""
import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ==============================================================================
# CONFIGURATION - team decisions, change them here only
# ==============================================================================
ADR_MAX = 1000                      # ADR >= 1000 is a data-entry error (the file has one row at 5400)
LEAD_BINS = [-1, 30, 120, np.inf]   # Short <= 30, Medium 31-120, Long > 120 days
LEAD_LABELS = ["Short", "Medium", "Long"]
WEEKEND_ARRIVAL_DAYS = (4, 5)       # Fri/Sat. Use (5, 6) to match the dataset's own weekend nights (Sat/Sun).

REQUIRED_COLS = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "adr", "deposit_type",
]
MONTH_MAP = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
             "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
SEASON_MAP = {"March": "Spring", "April": "Spring", "May": "Spring",
              "June": "Summer", "July": "Summer", "August": "Summer",
              "September": "Autumn", "October": "Autumn", "November": "Autumn",
              "December": "Winter", "January": "Winter", "February": "Winter"}


# ==============================================================================
# SHARED CLEANING
# ==============================================================================
def load_and_clean_base(filepath: str) -> pd.DataFrame:
    """Load the raw CSV, fix missing values, remove invalid rows and add the universal features."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at '{filepath}'. Please check the path.")

    # 'NULL' (agent / company) and 'Undefined' (meal, market_segment, ...) become real NaN
    df = pd.read_csv(filepath, na_values=["NULL", "Undefined", ""])
    logging.info(f"Raw dataset loaded: {df.shape}")

    missing_req = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_req:
        raise ValueError(f"Missing required columns: {missing_req}")

    # 1. Constant, rule-based fills (agent / company stay NaN: blank means 'no agent / no company')
    df["children"] = df["children"].fillna(0).astype(int)
    df["country"] = df["country"].fillna("Unknown")
    df["meal"] = df["meal"].fillna("SC")                    # undefined meal = no meal package
    for c in ["market_segment", "distribution_channel"]:
        df[c] = df[c].fillna("Unknown")

    # 2. Remove impossible records and extreme outliers
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["total_stay_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    bad_guests = df["total_guests"] <= 0
    bad_nights = df["total_stay_nights"] <= 0
    bad_adr = (df["adr"] < 0) | (df["adr"] >= ADR_MAX)      # ADR = 0 is kept (complimentary stays)
    keep = ~(bad_guests | bad_nights | bad_adr)
    logging.info(f"Invalid rows removed: {int((~keep).sum())} (0 guests: {int(bad_guests.sum())}, "
                 f"0 nights: {int(bad_nights.sum())}, bad ADR: {int(bad_adr.sum())}; rows may overlap)")
    df = df[keep].copy()

    # 3. Date and seasonality
    df["arrival_month_num"] = df["arrival_date_month"].map(MONTH_MAP)
    df["arrival_season"] = df["arrival_date_month"].map(SEASON_MAP).fillna("Unknown")
    df["arrival_date"] = pd.to_datetime(
        dict(year=df["arrival_date_year"], month=df["arrival_month_num"], day=df["arrival_date_day_of_month"]),
        errors="coerce")
    n_bad_dates = int(df["arrival_date"].isna().sum())
    if n_bad_dates:
        logging.warning(f"{n_bad_dates} rows have an invalid arrival date and are dropped")
        df = df.dropna(subset=["arrival_date"])
    df["is_weekend_arrival"] = df["arrival_date"].dt.dayofweek.isin(WEEKEND_ARRIVAL_DAYS).astype(int)

    # 4. Revenue and lead-time features
    df["booking_value"] = df["adr"] * df["total_stay_nights"]
    df["lead_time_category"] = pd.cut(df["lead_time"], bins=LEAD_BINS, labels=LEAD_LABELS).astype(str)

    df = df.reset_index(drop=True)
    validate_base(df)
    logging.info(f"Base cleaning done: {df.shape}")
    return df


def validate_base(df: pd.DataFrame) -> None:
    """Stop with a clear error if the cleaned base data breaks an expected rule."""
    for c in ["children", "country", "meal", "market_segment", "distribution_channel", "adr",
              "arrival_date", "total_guests", "total_stay_nights", "booking_value"]:
        assert df[c].notna().all(), f"Missing values remain in '{c}'"
    assert (df["total_guests"] > 0).all() and (df["total_stay_nights"] > 0).all()
    assert df["adr"].between(0, ADR_MAX, inclusive="left").all()
    assert df["is_canceled"].isin([0, 1]).all()
    assert df["lead_time_category"].isin(LEAD_LABELS).all()


def save_cleaning_report(df_raw_rows: int, df: pd.DataFrame, path: str) -> None:
    """Short summary for the report (row counts, cancellation rate, date range, hotel split)."""
    lines = [
        f"Raw rows: {df_raw_rows:,}",
        f"Rows after shared cleaning: {len(df):,} (duplicates kept)",
        f"Exact duplicate rows still present: {int(df.duplicated().sum()):,}",
        f"Arrival dates: {df['arrival_date'].min().date()} to {df['arrival_date'].max().date()}",
        f"Cancellation rate: {df['is_canceled'].mean():.1%}",
        "Rows per hotel: " + ", ".join(f"{k}: {v:,}" for k, v in df["hotel"].value_counts().items()),
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Shared preprocessing for the hotel booking project")
    ap.add_argument("--input", default="hotel_bookings.csv")
    ap.add_argument("--out-dir", default="processed_data")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    n_raw = len(pd.read_csv(a.input, usecols=["hotel"]))
    base = load_and_clean_base(a.input)
    base.to_csv(os.path.join(a.out_dir, "0_base_clean.csv"), index=False)
    save_cleaning_report(n_raw, base, os.path.join(a.out_dir, "0_cleaning_report.txt"))
    logging.info(f"Saved {a.out_dir}/0_base_clean.csv and 0_cleaning_report.txt")