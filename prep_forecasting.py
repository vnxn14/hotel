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


def build_series(df: pd.DataFrame, hotel: str) -> pd.DataFrame:
    """Aggregate bookings and room nights by arrival date for one hotel.

    Dates without arrivals are included with zero values in both target columns.
    """
    required = {"hotel", "arrival_date", "total_stay_nights"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    hotel_df = df.loc[df["hotel"].eq(hotel), ["arrival_date", "total_stay_nights"]].copy()
    if hotel_df.empty:
        raise ValueError(f"No bookings found for hotel '{hotel}'")

    hotel_df["arrival_date"] = pd.to_datetime(hotel_df["arrival_date"], errors="coerce")
    hotel_df = hotel_df.dropna(subset=["arrival_date"])
    if hotel_df.empty:
        raise ValueError(f"No valid arrival dates found for hotel '{hotel}'")

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

    return f


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

    for hotel in sorted(df["hotel"].dropna().unique()):
        features = make_features(build_series(df, hotel))
        features.insert(0, "hotel", hotel)
        slug = hotel.lower().replace(" ", "_")
        path = os.path.join(args.out_dir, f"forecasting_{slug}.csv")
        features.to_csv(path, index_label="arrival_date")
        logging.info("Saved %s (%s rows, %s columns)", path, len(features), len(features.columns))


if __name__ == "__main__":
    main()
