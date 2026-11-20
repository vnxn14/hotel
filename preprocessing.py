import os
import joblib
import logging
import numpy as np
import pandas as pd
from regex import sub
from scipy import datasets
from sklearn import datasets
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Configure logging for clean pipeline tracing
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ==============================================================================
# 1. SHARED BASE CLEANING (RUN ONCE ON RAW DATA)
# ==============================================================================
def load_and_clean_base(filepath: str) -> pd.DataFrame:
    """
    Loads raw hotel booking CSV, standardizes null values, removes invalid/corrupt
    records, and computes universal temporal & domain features.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at '{filepath}'. Please check the path.")

    try:
        # Convert string 'NULL' and 'Undefined' into true NaN values
        df = pd.read_csv(filepath, na_values=['NULL', 'Undefined', ''])
        logging.info(f"Raw dataset loaded: {df.shape}")

        required_cols = [
            'hotel', 'is_canceled', 'lead_time', 'arrival_date_year',
            'arrival_date_month', 'arrival_date_day_of_month',
            'stays_in_weekend_nights', 'stays_in_week_nights',
            'adults', 'children', 'babies', 'adr', 'deposit_type'
        ]
        missing_req = [c for c in required_cols if c not in df.columns]
        if missing_req:
            raise ValueError(f"Missing required columns: {missing_req}")

        # 1. Universal Missing Value Imputation
        df['children'] = df['children'].fillna(0).astype(int)
        df['country'] = df['country'].fillna('Unknown')
        df['meal'] = df['meal'].fillna('SC')

        # 2. Remove Impossible Records & Extreme Outliers
        # Justification: Bookings with 0 total guests or negative/extreme ADR (>=1000) are system entry errors.
        initial_len = len(df)
        df['total_guests'] = df['adults'] + df['children'] + df['babies']
        df = df[(df['total_guests'] > 0) & (df['adr'] >= 0) & (df['adr'] < 1000)].copy()
        logging.info(f"Removed {initial_len - len(df)} invalid/outlier rows.")

        # 3. Universal Date & Seasonality Construction
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        season_map = {
            'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
            'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
            'September': 'Autumn', 'October': 'Autumn', 'November': 'Autumn',
            'December': 'Winter', 'January': 'Winter', 'February': 'Winter'
        }
        df['arrival_month_num'] = df['arrival_date_month'].map(month_map)
        df['arrival_season'] = df['arrival_date_month'].map(season_map).fillna('Unknown')
        df['arrival_date'] = pd.to_datetime(
            dict(year=df['arrival_date_year'], month=df['arrival_month_num'], day=df['arrival_date_day_of_month']),
            errors='coerce'
        )
        # 4 = Friday, 5 = Saturday
        df['is_weekend_arrival'] = df['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int)

        # 4. Universal Stay & Revenue Features
        df['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        df['booking_value'] = df['adr'] * df['total_stay_nights']

        # Kept strictly for EDA charts and UI dashboard filtering (dropped before ML training)
        df['lead_time_category'] = pd.cut(
            df['lead_time'], bins=[-1, 30, 120, np.inf], labels=['Short', 'Medium', 'Long']
        ).astype(str)

        return df.reset_index(drop=True)

    except Exception as e:
        logging.error(f"Error in load_and_clean_base: {e}")
        raise


# ==============================================================================
# 2. CUSTOM FEATURE ENGINEER (FOR SUPERVISED & CLUSTERING PIPELINES)
# ==============================================================================
class HotelFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Automated feature engineering transformer.
    Refinements applied:
    - Dropped collinear 'weekday_stay_ratio' (kept only 'weekend_stay_ratio').
    - Dropped 'lead_time_category' to avoid redundancy with continuous 'lead_time'.
    - Added 'adr_per_person', 'is_deposit_required', and cyclical 'month_sin'/'month_cos'.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        try:
            # Binary channel flags (replaces high-cardinality agent/company IDs)
            df['has_agent'] = df['agent'].notna().astype(int)
            df['is_company'] = df['company'].notna().astype(int)

            # Guest & deposit indicators
            df['has_children'] = ((df['children'] + df['babies']) > 0).astype(int)
            df['is_deposit_required'] = (df['deposit_type'] != 'No Deposit').astype(int)

            # Normalized price per guest (differentiates solo travelers vs. families)
            df['adr_per_person'] = df['adr'] / df['total_guests']

            # Safe division for historical cancellation rate
            prev_total = df['previous_cancellations'] + df['previous_bookings_not_canceled']
            df['historical_cancellation_rate'] = np.where(
                prev_total > 0, df['previous_cancellations'] / prev_total, 0.0
            )

            # Weekend stay ratio only (avoids perfect multicollinearity with weekday ratio)
            df['weekend_stay_ratio'] = np.where(
                df['total_stay_nights'] > 0,
                df['stays_in_weekend_nights'] / df['total_stay_nights'],
                0.0
            )

            # Cyclical month encoding (captures continuity between Dec=12 and Jan=1)
            df['month_sin'] = np.sin(2 * np.pi * df['arrival_month_num'] / 12.0)
            df['month_cos'] = np.cos(2 * np.pi * df['arrival_month_num'] / 12.0)

            # Drop raw IDs and redundant helper columns
            cols_to_drop = ['agent', 'company', 'lead_time_category', 'arrival_date']
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
            return df

        except KeyError as e:
            logging.error(f"Missing column in HotelFeatureEngineer: {e}")
            raise


# ==============================================================================
# 3. TASK-SPECIFIC BRANCH A: SUPERVISED LEARNING (CANCELLATION RISK)
# ==============================================================================
def prepare_supervised_data(
    df_base: pd.DataFrame,
    imputation_strategy: str = 'simple',  # 'simple' (Part A) vs 'knn' (Part B)
    scaler_type: str = 'standard'         # 'standard' (Part A) vs 'robust' (Part B)
):
    """
    Prepares deduplicated, leak-free Train (70%), Val (15%), and Test (15%) sets
    wrapped in an automated scikit-learn Pipeline.
    """
    # 1. Drop exact duplicates so models do not overfit on repeated group rows
    df = df_base.drop_duplicates().copy()

    # 2. Drop post-booking columns to prevent target leakage
    leakage_cols = ['reservation_status', 'reservation_status_date', 'assigned_room_type', 'booking_changes']
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    # 3. Group rare countries (<1%) into 'Other' to prevent sparse one-hot explosion
    country_freq = df['country'].value_counts(normalize=True)
    rare_countries = country_freq[country_freq < 0.01].index
    df['country'] = df['country'].replace(rare_countries, 'Other')

    X = df.drop(columns=['is_canceled'])
    y = df['is_canceled']

    # 4. Stratified 70 / 15 / 15 split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # Identify column types after feature engineering
    sample_eng = HotelFeatureEngineer().transform(X_train.head(5))
    cat_cols = sample_eng.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = sample_eng.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()

    # Configure Part A vs Part B transformers
    num_imputer = KNNImputer(n_neighbors=5) if imputation_strategy == 'knn' else SimpleImputer(strategy='median')
    scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', num_imputer), ('scaler', scaler)]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ],
        remainder='drop'
    )

    full_pipeline = Pipeline(steps=[
        ('feature_eng', HotelFeatureEngineer()),
        ('preprocessor', preprocessor)
    ])

    # Fit strictly on X_train to guarantee zero data leakage
    X_train_proc = full_pipeline.fit_transform(X_train)
    X_val_proc = full_pipeline.transform(X_val)
    X_test_proc = full_pipeline.transform(X_test)

    logging.info(f"Supervised splits ready -> Train: {X_train_proc.shape}, Val: {X_val_proc.shape}, Test: {X_test_proc.shape}")
    return X_train_proc, X_val_proc, X_test_proc, y_train, y_val, y_test, full_pipeline


# ==============================================================================
# 4. TASK-SPECIFIC BRANCH B: UNSUPERVISED LEARNING (GUEST CLUSTERING)
# ==============================================================================
def prepare_clustering_data(df_base: pd.DataFrame):
    """
    Prepares deduplicated dataset using the 10 recommended clustering features
    plus 'adr_per_person' for sharper guest persona separation.
    """
    df = df_base.drop_duplicates().copy()
    df['adr_per_person'] = df['adr'] / df['total_guests']
    cluster_features = [
        'lead_time', 'previous_cancellations', 'total_stay_nights',
        'market_segment', 'total_guests', 'distribution_channel',
        'adr', 'adr_per_person', 'customer_type', 'is_repeated_guest', 'reserved_room_type'
    ]

    df_cluster = df[cluster_features].copy()
    cat_cols = ['market_segment', 'distribution_channel', 'customer_type', 'reserved_room_type']
    num_cols = [c for c in cluster_features if c not in cat_cols]
    cluster_pipeline = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    X_cluster_scaled = cluster_pipeline.fit_transform(df_cluster)
    logging.info(f"Clustering matrix ready -> Shape: {X_cluster_scaled.shape}")
    # Returns scaled matrix (for K-Means/GMM) and raw dataframe (for K-Prototypes & profiling)
    return X_cluster_scaled, df_cluster, cluster_pipeline

# ==============================================================================
# 5. TASK-SPECIFIC BRANCH C: TIME-SERIES FORECASTING (DEMAND)
# ==============================================================================

def prepare_forecasting_data(df_base: pd.DataFrame, freq: str = 'D', test_ratio: float = 0.20):
    """
    Aggregates daily ('D') or weekly ('W') demand WITHOUT dropping duplicates,
    and performs a chronological Train/Test split per hotel type.
    Also generates lag & rolling features for ML regressors (XGBoost/RF).
    """
    df = df_base.copy()
    df = df.dropna(subset=['arrival_date'])
    # Aggregate booking demand per hotel and date
    ts_grouped = (
        df.groupby(['hotel', pd.Grouper(key='arrival_date', freq=freq)])
        .size()
        .reset_index(name='booking_demand')
    )
    datasets = {}
    for hotel_name in ['City Hotel', 'Resort Hotel']:
        sub = ts_grouped[ts_grouped['hotel'] == hotel_name].set_index('arrival_date')['booking_demand']
        # Fill missing calendar dates with 0 demand to maintain continuous frequency
        sub = sub.asfreq(freq, fill_value=0)
    
        # Build supervised tabular features (lags + rolling mean + cyclical month) for XGBoost/RF
        ts_df = pd.DataFrame({'y': sub})
        ts_df['lag_1'] = ts_df['y'].shift(1)
        ts_df['lag_7'] = ts_df['y'].shift(7)
        ts_df['rolling_7_mean'] = ts_df['y'].shift(1).rolling(window=7).mean()
        ts_df['dayofweek'] = ts_df.index.dayofweek
        ts_df['month_sin'] = np.sin(2 * np.pi * ts_df.index.month / 12.0)
        ts_df['month_cos'] = np.cos(2 * np.pi * ts_df.index.month / 12.0)
        ts_df = ts_df.dropna()
    
        # Chronological split (never shuffle time-series data)
        split_idx = int(len(ts_df) * (1 - test_ratio))
        train_df = ts_df.iloc[:split_idx]
        test_df = ts_df.iloc[split_idx:]
        datasets[hotel_name] = {'train': train_df, 'test': test_df}
        logging.info(f"Forecasting [{hotel_name}] ready -> Train: {len(train_df)}, Test: {len(test_df)}")
    return datasets


# ==============================================================================
# MAIN EXECUTION CHECK
# ==============================================================================

if __name__ == "__main__":
    # Create a dedicated folder to keep your project directory clean
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Run Shared Base Cleaning & Save (Useful for EDA & Streamlit Dashboard)
    df_clean = load_and_clean_base("hotel_bookings.csv")
    df_clean.to_csv(os.path.join(output_dir, "0_base_clean.csv"), index=False)

    # --------------------------------------------------------------------------
    # 2. Save Supervised Data (Cancellation Risk Module)
    # --------------------------------------------------------------------------
    X_tr, X_va, X_te, y_tr, y_va, y_te, sup_pipe = prepare_supervised_data(
        df_clean, imputation_strategy='simple', scaler_type='robust'
    )
    # Extract transformed feature names for SHAP & feature importance plots
    sup_feature_names = sup_pipe.named_steps['preprocessor'].get_feature_names_out()

    pd.DataFrame(X_tr, columns=sup_feature_names).to_csv(os.path.join(output_dir, "sup_X_train.csv"), index=False)
    pd.DataFrame(X_va, columns=sup_feature_names).to_csv(os.path.join(output_dir, "sup_X_val.csv"), index=False)
    pd.DataFrame(X_te, columns=sup_feature_names).to_csv(os.path.join(output_dir, "sup_X_test.csv"), index=False)
    y_tr.to_csv(os.path.join(output_dir, "sup_y_train.csv"), index=False)
    y_va.to_csv(os.path.join(output_dir, "sup_y_val.csv"), index=False)
    y_te.to_csv(os.path.join(output_dir, "sup_y_test.csv"), index=False)
    joblib.dump(sup_pipe, os.path.join(output_dir, "sup_pipeline.joblib"))

    # --------------------------------------------------------------------------
    # 3. Save Unsupervised Data (Guest Clustering Module)
    # --------------------------------------------------------------------------
    X_clust, df_clust_raw, clust_pipe = prepare_clustering_data(df_clean)
    clust_feature_names = clust_pipe.get_feature_names_out()

    pd.DataFrame(X_clust, columns=clust_feature_names).to_csv(os.path.join(output_dir, "clust_X_scaled.csv"), index=False)
    df_clust_raw.to_csv(os.path.join(output_dir, "clust_raw_features.csv"), index=False)
    joblib.dump(clust_pipe, os.path.join(output_dir, "clust_pipeline.joblib"))

    # --------------------------------------------------------------------------
    # 4. Save Time-Series Data (Demand Forecasting Module)
    # --------------------------------------------------------------------------
    ts_data = prepare_forecasting_data(df_clean, freq='D', test_ratio=0.20)
    for hotel_name, splits in ts_data.items():
        prefix = "city" if hotel_name == "City Hotel" else "resort"
        # index=True is critical here to preserve the arrival_date DatetimeIndex
        splits['train'].to_csv(os.path.join(output_dir, f"ts_{prefix}_train.csv"), index=True)
        splits['test'].to_csv(os.path.join(output_dir, f"ts_{prefix}_test.csv"), index=True)

    logging.info(f"All processed files successfully saved inside '{output_dir}/'!")