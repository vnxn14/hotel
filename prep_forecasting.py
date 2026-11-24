"""Prepare daily hotel booking targets and forecasting features.

The input is the shared cleaned booking dataset produced by
``shared_preprocessing.py``. One feature CSV is written per hotel.
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

LAGS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28)


def build_series(df: pd.DataFrame, hotel: str | None = None) -> pd.DataFrame:
    """Aggregate bookings and room nights by arrival date.

    When ``hotel`` is provided, only that hotel is included. Otherwise the
    series combines all hotels. Dates without arrivals are zero-filled.
    """
    required = {"hotel", "arrival_date", "total_stay_nights"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    hotel_df = df.loc[
        df["hotel"].eq(hotel) if hotel is not None else slice(None),
        ["arrival_date", "total_stay_nights"],
    ].copy()
    if hotel_df.empty:
        series_label = hotel if hotel is not None else "all hotels"
        raise ValueError(f"No bookings found for '{series_label}'")

    hotel_df["arrival_date"] = pd.to_datetime(hotel_df["arrival_date"], errors="coerce")
    hotel_df = hotel_df.dropna(subset=["arrival_date"])
    if hotel_df.empty:
        series_label = hotel if hotel is not None else "all hotels"
        raise ValueError(f"No valid arrival dates found for '{series_label}'")

    series = hotel_df.groupby("arrival_date").agg(
        y_bookings=("total_stay_nights", "size"),
        y_room_nights=("total_stay_nights", "sum"),
    )
    full_dates = pd.date_range(series.index.min(), series.index.max(), freq="D", name="arrival_date")
    series = series.reindex(full_dates, fill_value=0)
    series[["y_bookings", "y_room_nights"]] = series[["y_bookings", "y_room_nights"]].fillna(0)
    series[["y_bookings", "y_room_nights"]] = series[["y_bookings", "y_room_nights"]].astype("int64")
    return series


def make_features(series: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, lag, and rolling features while retaining both targets."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex named 'arrival_date'")
    if not {"y_bookings", "y_room_nights"}.issubset(series.columns):
        raise ValueError("series must contain y_bookings and y_room_nights")

    f = series.copy().sort_index()
    dates = f.index
    f["f_day_of_week"] = dates.dayofweek
    f["f_day_of_year"] = dates.dayofyear
    f["f_month"] = dates.month
    f["f_quarter"] = dates.quarter
    f["f_is_weekend"] = (dates.dayofweek >= 5).astype("int8")
    f["time_idx"] = np.arange(len(f))

    bookings = f["y_bookings"]
    for lag in LAGS:
        f[f"f_bookings_lag_{lag}"] = bookings.shift(lag)
    for window in ROLLING_WINDOWS:
        f[f"f_bookings_rolling_mean_{window}"] = bookings.shift(1).rolling(window=window).mean()

    return f.dropna()

def split_and_save(features: pd.DataFrame, slug: str, out_dir: str, test_ratio: float = 0.20):
    """Splits the data chronologically into train and test sets and saves them."""
    cutoff = features.index[int(len(features) * (1 - test_ratio))]
    train = features[features.index < cutoff]
    test = features[features.index >= cutoff]
    
    train.to_csv(os.path.join(out_dir, f"ts_{slug}_features_train.csv"), index_label="arrival_date")
    test.to_csv(os.path.join(out_dir, f"ts_{slug}_features_test.csv"), index_label="arrival_date")
    logging.info(f"Split {slug}: Train ends {train.index[-1].date()}, Test starts {test.index[0].date()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare hotel booking forecasting features")
    parser.add_argument("--input", default=os.path.join("processed_data", "0_base_clean.csv"),
                        help="Shared cleaned bookings CSV")
    parser.add_argument("--out-dir", default="processed_data", help="Directory for feature CSVs")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Cleaned dataset not found at '{args.input}'. Run shared_preprocessing.py first."
        )
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.input, parse_dates=["arrival_date"])
    if "hotel" not in df:
        raise ValueError("Input CSV is missing required column 'hotel'")

    series = (
        ("city", "City Hotel"),
        ("all", None),
        ("resort", "Resort Hotel"),
    )
    for slug, hotel in series:
        features = make_features(build_series(df, hotel))
        # Save the full dataset
        path = os.path.join(args.out_dir, f"forecasting_{slug}.csv")
        features.to_csv(path, index_label="arrival_date")
        
        # Split and save train/test files for ML models
        split_and_save(features, slug, args.out_dir)
        
        logging.info("Processed %s (%s rows, %s columns)", path, len(features), len(features.columns))


if __name__ == "__main__":
    main()
