"""Preprocess hotel_bookings.csv into model-ready train/validation/test CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT = Path("hotel_bookings.csv")
OUTPUT_DIR = Path(".")
TARGET = "is_canceled"
RANDOM_STATE = 42
LEAKAGE_COLUMNS = [
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list",
]
BINARY_COLUMNS = {
    "is_repeated_guest",
    "has_agent",
    "has_company",
    "has_children",
    "is_weekend_arrival",
}
CATEGORICAL_COLUMNS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "lead_time_category",
    "arrival_season",
]


def report_step(report: dict, name: str, passed: bool, **details: object) -> None:
    status = "PASS" if passed else "FAIL"
    report["preprocessing"].append({"step": name, "status": status, **details})
    print(f"[{status}] {name}")
    if not passed:
        raise RuntimeError(f"Validation failed at preprocessing step: {name}")


def write_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(OUTPUT_DIR / filename, index=False)


def main() -> None:
    data = pd.read_csv(INPUT)
    original_rows, original_columns = data.shape
    initial_missing = data.isna().sum()
    report: dict = {
        "source_file": str(INPUT),
        "source_rows": int(original_rows),
        "source_columns": int(original_columns),
        "preprocessing": [],
    }

    report_step(
        report,
        "Input parsing",
        len(data) == original_rows and len(data.columns) == original_columns,
        rows=int(len(data)),
        columns=int(len(data.columns)),
    )

    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()
    report_step(
        report,
        "Duplicate analysis and removal",
        not data.duplicated().any(),
        exact_duplicates_found=duplicate_count,
        rows_after_deduplication=int(len(data)),
    )

    missing_by_column = {
        str(column): int(count)
        for column, count in initial_missing.items()
        if count
    }
    missing_target_count = int(data[TARGET].isna().sum())
    data = data.dropna(subset=[TARGET]).copy()
    report_step(
        report,
        "Missing value assessment and target handling",
        data[TARGET].notna().all(),
        missing_cells_by_column=missing_by_column,
        rows_removed_for_missing_target=missing_target_count,
        handling=(
            "Missing children and babies are treated as zero in guest-count features; "
            "remaining numeric predictors use training medians and categorical predictors "
            "use training modes."
        ),
    )

    # Convert the outcome and ID-like values to numeric data types where possible.
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce").astype("int64")
    for column in ("agent", "company"):
        if column in data:
            values = pd.to_numeric(data[column], errors="coerce")
            data[f"has_{column}"] = (values.notna() & values.ne(0)).astype("int8")
            data = data.drop(columns=column)

    # Feature engineering from information available when the booking is made.
    weekend_nights = pd.to_numeric(data["stays_in_weekend_nights"], errors="coerce").fillna(0)
    weekday_nights = pd.to_numeric(data["stays_in_week_nights"], errors="coerce").fillna(0)
    data["total_nights"] = weekend_nights + weekday_nights
    children = pd.to_numeric(data["children"], errors="coerce").fillna(0)
    babies = pd.to_numeric(data["babies"], errors="coerce").fillna(0)
    adults = pd.to_numeric(data["adults"], errors="coerce")
    data["has_children"] = ((children + babies) > 0).astype("int8")
    data["total_guests"] = adults + children + babies
    adr = pd.to_numeric(data["adr"], errors="coerce")
    data["booking_value"] = adr * data["total_nights"]

    previous_cancellations = pd.to_numeric(
        data["previous_cancellations"], errors="coerce"
    ).fillna(0)
    previous_not_cancelled = pd.to_numeric(
        data["previous_bookings_not_canceled"], errors="coerce"
    ).fillna(0)
    previous_total = previous_cancellations + previous_not_cancelled
    data["historical_cancellation_rate"] = np.divide(
        previous_cancellations,
        previous_total,
        out=np.zeros(len(data), dtype="float64"),
        where=previous_total.to_numpy() != 0,
    )

    data["weekend_stay_ratio"] = np.divide(
        weekend_nights,
        data["total_nights"],
        out=np.zeros(len(data), dtype="float64"),
        where=data["total_nights"].to_numpy() != 0,
    )
    data["weekday_stay_ratio"] = np.where(
        data["total_nights"].eq(0), 0, 1 - data["weekend_stay_ratio"]
    )

    lead_time = pd.to_numeric(data["lead_time"], errors="coerce")
    data["lead_time_category"] = pd.cut(
        lead_time,
        bins=[-np.inf, 30, 90, np.inf],
        labels=["short", "medium", "long"],
    ).astype("object")
    month_number = pd.to_datetime(
        data["arrival_date_month"], format="%B", errors="coerce"
    ).dt.month
    season_map = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }
    data["arrival_season"] = month_number.map(season_map)
    arrival_date = pd.to_datetime(
        {
            "year": pd.to_numeric(data["arrival_date_year"], errors="coerce"),
            "month": month_number,
            "day": pd.to_numeric(data["arrival_date_day_of_month"], errors="coerce"),
        },
        errors="coerce",
    )
    data["is_weekend_arrival"] = arrival_date.dt.dayofweek.isin([5, 6]).astype("int8")

    removed_leakage = [column for column in LEAKAGE_COLUMNS if column in data.columns]
    data = data.drop(columns=removed_leakage)
    report_step(
        report,
        "Data type conversion and leakage prevention",
        TARGET in data.columns and not set(removed_leakage).intersection(data.columns),
        removed_post_booking_or_outcome_columns=removed_leakage,
        agent_and_company_ids_replaced_with_binary_flags=True,
        target_column=TARGET,
    )
    engineering_columns = [
        "total_nights", "total_guests", "has_children", "booking_value",
        "historical_cancellation_rate", "weekend_stay_ratio", "weekday_stay_ratio",
        "lead_time_category", "arrival_season", "is_weekend_arrival",
        "has_agent", "has_company",
    ]
    ratio_valid = data[
        ["historical_cancellation_rate", "weekend_stay_ratio", "weekday_stay_ratio"]
    ].apply(lambda col: col.between(0, 1).all()).all()
    report_step(
        report,
        "Feature engineering",
        all(column in data.columns for column in engineering_columns) and ratio_valid,
        generated_features=engineering_columns,
        lead_time_categories="short: 0-30 days; medium: 31-90 days; long: over 90 days",
        arrival_seasons="Northern Hemisphere month ranges",
        zero_night_ratios="Both stay ratios are 0 when total_nights is 0",
    )

    # Keep the original deduplicated row index only for split-overlap validation.
    data = data.reset_index(drop=True)
    row_ids = data.index.to_numpy()
    development_ids, test_ids = train_test_split(
        row_ids,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data[TARGET],
    )
    train_ids, validation_ids = train_test_split(
        development_ids,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data.loc[development_ids, TARGET],
    )
    train = data.loc[train_ids].copy()
    validation = data.loc[validation_ids].copy()
    test = data.loc[test_ids].copy()

    predictors = [column for column in data.columns if column != TARGET]
    categorical = [column for column in CATEGORICAL_COLUMNS if column in predictors]
    numeric = [column for column in predictors if column not in categorical]
    for column in categorical:
        for frame in (train, validation, test):
            frame[column] = frame[column].astype("string")

    numeric_medians = train[numeric].median(numeric_only=True).fillna(0)
    categorical_modes = {
        column: (train[column].mode(dropna=True).iloc[0] if train[column].notna().any() else "Missing")
        for column in categorical
    }
    for frame in (train, validation, test):
        frame[numeric] = frame[numeric].fillna(numeric_medians)
        for column in categorical:
            frame[column] = frame[column].fillna(categorical_modes[column]).astype(str)

    train_encoded = pd.get_dummies(train[predictors], columns=categorical, dtype=np.int8)
    validation_encoded = pd.get_dummies(validation[predictors], columns=categorical, dtype=np.int8)
    test_encoded = pd.get_dummies(test[predictors], columns=categorical, dtype=np.int8)
    encoded_columns = train_encoded.columns.tolist()
    validation_encoded = validation_encoded.reindex(columns=encoded_columns, fill_value=0)
    test_encoded = test_encoded.reindex(columns=encoded_columns, fill_value=0)

    continuous = [column for column in numeric if column not in BINARY_COLUMNS]
    outlier_bounds = {
        column: (
            float(train_encoded[column].quantile(0.01)),
            float(train_encoded[column].quantile(0.99)),
        )
        for column in continuous
    }
    for frame in (train_encoded, validation_encoded, test_encoded):
        for column, (lower, upper) in outlier_bounds.items():
            frame[column] = frame[column].clip(lower, upper)

    means = train_encoded[continuous].mean()
    standard_deviations = train_encoded[continuous].std(ddof=0).replace(0, 1)
    for frame in (train_encoded, validation_encoded, test_encoded):
        frame[continuous] = (frame[continuous] - means) / standard_deviations

    # Remove constant predictors based on training data only.
    selected_columns = [
        column for column in encoded_columns if train_encoded[column].nunique(dropna=False) > 1
    ]
    train_encoded = train_encoded[selected_columns].copy()
    validation_encoded = validation_encoded[selected_columns].copy()
    test_encoded = test_encoded[selected_columns].copy()
    train_encoded[TARGET] = train[TARGET].astype("int8").to_numpy()
    validation_encoded[TARGET] = validation[TARGET].astype("int8").to_numpy()
    test_encoded[TARGET] = test[TARGET].astype("int8").to_numpy()

    split_ids_disjoint = (
        set(train_ids).isdisjoint(validation_ids)
        and set(train_ids).isdisjoint(test_ids)
        and set(validation_ids).isdisjoint(test_ids)
    )
    row_count_conserved = len(train) + len(validation) + len(test) == len(data)
    report_step(
        report,
        "Stratified train/validation/test split",
        split_ids_disjoint and row_count_conserved,
        train_rows=int(len(train)),
        validation_rows=int(len(validation)),
        test_rows=int(len(test)),
        train_target_counts={str(k): int(v) for k, v in train[TARGET].value_counts().sort_index().items()},
        validation_target_counts={str(k): int(v) for k, v in validation[TARGET].value_counts().sort_index().items()},
        test_target_counts={str(k): int(v) for k, v in test[TARGET].value_counts().sort_index().items()},
        split_policy="80% development / 20% test; development split 80% train / 20% validation (about 64/16/20 overall)",
    )

    outputs = {
        "hotel_bookings_train.csv": train_encoded,
        "hotel_bookings_validation.csv": validation_encoded,
        "hotel_bookings_test.csv": test_encoded,
    }
    for filename, frame in outputs.items():
        write_csv(frame, filename)
    outputs_numeric_and_complete = all(
        not frame.isna().any().any()
        and all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes)
        for frame in outputs.values()
    )
    schemas_match = len({tuple(frame.columns) for frame in outputs.values()}) == 1
    report_step(
        report,
        "Missing value imputation, categorical encoding, feature scaling, outlier handling, and feature selection",
        outputs_numeric_and_complete and schemas_match,
        missing_value_policy="training medians for numeric predictors; training modes for categorical predictors; children/babies treated as zero for guest-count features",
        categorical_columns_encoded=categorical,
        scaled_continuous_columns=continuous,
        unscaled_binary_columns=sorted(BINARY_COLUMNS.intersection(selected_columns)),
        outlier_caps="training 1st and 99th percentiles",
        fit_on="training subset only",
        constant_predictors_removed=int(len(encoded_columns) - len(selected_columns)),
        predictor_count=int(len(selected_columns)),
        target_column=TARGET,
    )

    report["outputs"] = list(outputs)
    report["validation"] = {
        "row_count_conserved": bool(row_count_conserved),
        "split_rows_disjoint": bool(split_ids_disjoint),
        "output_schema_consistent": bool(schemas_match),
        "all_output_values_numeric_and_complete": bool(outputs_numeric_and_complete),
        "all_output_files_exist": all((OUTPUT_DIR / filename).is_file() for filename in outputs),
    }
    with (OUTPUT_DIR / "hotel_bookings_preprocessing_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print("\nValidation summary:")
    for check, passed in report["validation"].items():
        print(f"[{'PASS' if passed else 'FAIL'}] {check}: {passed}")


if __name__ == "__main__":
    main()
