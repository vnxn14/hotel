import os
import logging
import math
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def prepare_eda_charts(base_path: str, out_dir: str):
    """Aggregates historical bookings by month and weekday."""
    df = pd.read_csv(base_path, parse_dates=["arrival_date"])
    
    # 1. Bookings by Month
    month_counts = df.groupby("arrival_month_num").size().reindex(range(1, 13), fill_value=0)
    month_names = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    df_month = pd.DataFrame({"Month": month_names, "Bookings": month_counts.values})
    
    # 2. Bookings by Weekday (Monday=0, Sunday=6)
    weekday_counts = df.groupby(df["arrival_date"].dt.dayofweek).size().reindex(range(7), fill_value=0)
    weekday_names = ["M", "T", "W", "T", "F", "S", "S"]
    df_weekday = pd.DataFrame({"Weekday": weekday_names, "Bookings": weekday_counts.values})
    
    df_month.to_csv(os.path.join(out_dir, "dashboard_eda_month.csv"), index=False)
    df_weekday.to_csv(os.path.join(out_dir, "dashboard_eda_weekday.csv"), index=False)
    logging.info("Saved dashboard_eda_month.csv and dashboard_eda_weekday.csv")


def prepare_staffing_plan(preds_path: str, out_dir: str):
    """Calculates weekly staffing requirements for the next 4 weeks."""
    # Using City Hotel predictions as the primary driver for the dashboard example
    df = pd.read_csv(preds_path, parse_dates=["Date"])
    
    # Take the first 28 days of the forecast and group into 4 weeks
    df = df.sort_values("Date").head(28).copy()
    
    # Create a Week column (1, 2, 3, 4)
    df["Week_Num"] = (np.arange(len(df)) // 7) + 1
    
    # Aggregate predictions by week
    # We use the winning model (Random Forest) for the forecast bookings
    weekly = df.groupby("Week_Num").agg(
        Forecast_Bookings=("pred_rf", "sum"),
        Forecast_Room_Nights=("y_room_nights", "sum")  # Using ground truth/trend for nights
    ).reset_index()
    
    weekly["Week"] = "Week " + weekly["Week_Num"].astype(str)
    
    # Apply Business Logic Rules
    # Front Desk: 1 per 75 bookings
    weekly["Front_Desk"] = (weekly["Forecast_Bookings"] / 600).apply(math.ceil)
    
    # Housekeeping: 1 per 48 room-nights
    weekly["Housekeeping"] = (weekly["Forecast_Room_Nights"] / 75).apply(math.ceil)
    
    # Clean up final table
    final_plan = weekly[["Week", "Forecast_Bookings", "Front_Desk", "Housekeeping"]].copy()
    final_plan["Forecast_Bookings"] = final_plan["Forecast_Bookings"].astype(int)
    
    out_path = os.path.join(out_dir, "dashboard_staffing_plan.csv")
    final_plan.to_csv(out_path, index=False)
    logging.info(f"Saved dashboard_staffing_plan.csv")


if __name__ == "__main__":
    out_directory = "processed_data"
    
    # Generate EDA Data
    prepare_eda_charts(os.path.join(out_directory, "0_base_clean.csv"), out_directory)
    
    # Generate Staffing Plan (Using City Hotel as the baseline)
    preds_file = os.path.join(out_directory, "preds_ml_city.csv")
    if os.path.exists(preds_file):
        prepare_staffing_plan(preds_file, out_directory)
    else:
        logging.error(f"Cannot generate staffing plan. Missing {preds_file}")