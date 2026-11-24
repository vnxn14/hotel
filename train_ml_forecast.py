"""Train XGBoost and Random Forest models for each hotel booking series."""
import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SERIES_NAMES = ("city", "all", "resort")
TARGET_COLUMNS = ("y_bookings", "y_room_nights")
DATE_COLUMNS = ("Date", "date", "arrival_date", "ds")


def load_features(path: Path) -> pd.DataFrame:
    """Load a feature CSV and use its date column as a sorted DatetimeIndex."""
    frame = pd.read_csv(path)
    date_column = next((column for column in DATE_COLUMNS if column in frame.columns), None)
    if date_column is None and str(frame.columns[0]).startswith("Unnamed:"):
        date_column = frame.columns[0]
    if date_column is None:
        raise ValueError(f"No date column found in {path}; expected one of {DATE_COLUMNS}")

    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    if frame[date_column].isna().any():
        raise ValueError(f"Invalid or missing dates found in {path}")
    frame = frame.set_index(date_column).sort_index()
    frame.index.name = "Date"

    missing_targets = [column for column in TARGET_COLUMNS if column not in frame.columns]
    if missing_targets:
        raise ValueError(f"{path} is missing target columns: {missing_targets}")
    return frame


def train_series(series_name: str, data_dir: Path, output_dir: Path) -> Path:
    """Fit both regressors for one series and save actuals with predictions."""
    train_path = data_dir / f"ts_{series_name}_features_train.csv"
    test_path = data_dir / f"ts_{series_name}_features_test.csv"
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required feature file not found: {path}")

    train = load_features(train_path)
    test = load_features(test_path)
    feature_columns = [column for column in train.columns if column not in TARGET_COLUMNS]
    if not feature_columns:
        raise ValueError(f"No feature columns found in {train_path}")
    missing_test_features = [column for column in feature_columns if column not in test.columns]
    if missing_test_features:
        raise ValueError(f"{test_path} is missing feature columns: {missing_test_features}")

    X_train = train[feature_columns].apply(pd.to_numeric, errors="raise")
    y_train = pd.to_numeric(train["y_bookings"], errors="raise")
    X_test = test[feature_columns].apply(pd.to_numeric, errors="raise")

    # Lag and rolling features can be undefined at the beginning of a series.
    valid_train = X_train.notna().all(axis=1) & y_train.notna()
    dropped_rows = int((~valid_train).sum())
    X_train = X_train.loc[valid_train]
    y_train = y_train.loc[valid_train]
    if X_train.empty:
        raise ValueError(f"No complete training rows remain for series '{series_name}'")
    if X_test.isna().any().any():
        raise ValueError(f"Test features in {test_path} contain missing values")

    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    predictions = pd.DataFrame(index=test.index)
    predictions["y_bookings"] = test["y_bookings"]
    predictions["y_room_nights"] = test["y_room_nights"]
    predictions["pred_xgboost"] = xgb_model.predict(X_test)
    predictions["pred_rf"] = rf_model.predict(X_test)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"preds_ml_{series_name}.csv"
    predictions.to_csv(output_path, index_label="Date")
    logging.info(
        "Saved %s (%s predictions; dropped %s incomplete training rows)",
        output_path,
        len(predictions),
        dropped_rows,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("processed_data"),
                        help="Directory containing ts_<series>_features_{train,test}.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("processed_data"),
                        help="Directory for preds_ml_<series>.csv outputs")
    parser.add_argument("--series", nargs="+", choices=SERIES_NAMES, default=SERIES_NAMES,
                        help="Series to train (default: city all resort)")
    args = parser.parse_args()

    for series_name in args.series:
        train_series(series_name, args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
