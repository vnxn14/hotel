import pandas as pd
import numpy as np


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads the INN Hotels Reservations dataset and performs shared data cleaning:
    - Drops non-predictive IDs
    - Removes duplicate records
    - Fixes/removes invalid calendar dates (e.g., 2018-02-29)
    - Filters logical anomalies (0 guests or 0 stay nights)
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please check the file path.")

    initial_rows = len(df)
    print(f"Initial dataset shape: {df.shape}")

    # 1. Drop unique identifier to prevent overfitting
    if 'Booking_ID' in df.columns:
        df = df.drop(columns=['Booking_ID'])

    # 2. Remove exact duplicate rows
    duplicates_count = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {duplicates_count} exact duplicate rows.")

    # 3. Construct and validate arrival date (handles invalid dates like 2018-02-29)
    date_df = df[['arrival_year', 'arrival_month', 'arrival_date']].rename(
        columns={'arrival_year': 'year', 'arrival_month': 'month', 'arrival_date': 'day'}
    )
    df['full_arrival_date'] = pd.to_datetime(date_df, errors='coerce')

    invalid_dates = df['full_arrival_date'].isna().sum()
    df = df.dropna(subset=['full_arrival_date']).reset_index(drop=True)
    print(f"Removed {invalid_dates} rows with invalid calendar dates (e.g., Feb 29 on non-leap year).")

    # 4. Filter logical anomalies: bookings must have at least 1 guest and 1 night
    total_guests_check = df['no_of_adults'] + df['no_of_children']
    total_nights_check = df['no_of_weekend_nights'] + df['no_of_week_nights']

    valid_mask = (total_guests_check > 0) & (total_nights_check > 0) & (df['avg_price_per_room'] >= 0)
    anomalies_removed = (~valid_mask).sum()
    df = df[valid_mask].reset_index(drop=True)
    print(f"Removed {anomalies_removed} anomalous rows (0 guests, 0 nights, or negative price).")

    print(f"Cleaned dataset shape: {df.shape} (Total removed: {initial_rows - len(df)} rows)\n")
    return df


def engineer_shared_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates domain-specific features aligned with the project design document:
    - Stay duration & weekend/weekday ratios
    - Guest composition & booking value
    - Historical cancellation rate
    - Lead time category, arrival season, and weekend arrival flag
    """
    df = df.copy()

    # a. Total stay nights
    df['total_stay_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']

    # b. Total guests & c. Has children flag
    df['total_guests'] = df['no_of_adults'] + df['no_of_children']
    df['has_children'] = (df['no_of_children'] > 0).astype(int)

    # d. Total booking value (ADR * total stay nights)
    df['booking_value'] = df['avg_price_per_room'] * df['total_stay_nights']

    # e. Historical cancellation rate (safely handling division by zero for first-time guests)
    prev_total = df['no_of_previous_cancellations'] + df['no_of_previous_bookings_not_canceled']
    df['historical_cancellation_rate'] = np.where(
        prev_total > 0,
        df['no_of_previous_cancellations'] / prev_total,
        0.0
    )

    # f. Weekend stay ratio & g. Weekday stay ratio
    df['weekend_stay_ratio'] = df['no_of_weekend_nights'] / df['total_stay_nights']
    df['weekday_stay_ratio'] = 1.0 - df['weekend_stay_ratio']

    # h. Lead time category (Short: <=7 days, Medium: 8-60 days, Long: >60 days)
    df['lead_time_category'] = pd.cut(
        df['lead_time'],
        bins=[-1, 7, 60, np.inf],
        labels=['Short', 'Medium', 'Long']
    ).astype(str)

    # i. Arrival season (Northern Hemisphere meteorological seasons)
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df['arrival_season'] = df['arrival_month'].map(season_map)

    # j. Is weekend arrival (Friday=4, Saturday=5, Sunday=6 in pandas dayofweek)
    df['arrival_day_of_week'] = df['full_arrival_date'].dt.dayofweek
    df['is_weekend_arrival'] = df['arrival_day_of_week'].isin([4, 5, 6]).astype(int)

    # Binary target encoding for convenience in B.1 (Canceled = 1, Not_Canceled = 0)
    df['is_canceled'] = (df['booking_status'] == 'Canceled').astype(int)

    print("Shared feature engineering completed successfully.")
    return df


if __name__ == "__main__":
    # Define input and output file paths
    RAW_DATA_PATH = "Hotel Reservations.csv"
    CLEAN_OUTPUT_PATH = "cleaned_hotel_reservations.csv"

    # Execute Stage 1 pipeline
    cleaned_df = load_and_clean_data(RAW_DATA_PATH)
    master_df = engineer_shared_features(cleaned_df)

    # Save master dataset for Stage 2 branches (B.1, B.2, B.3)
    master_df.to_csv(CLEAN_OUTPUT_PATH, index=False)
    print(f"Saved master preprocessed dataset to '{CLEAN_OUTPUT_PATH}' with shape {master_df.shape}.")
