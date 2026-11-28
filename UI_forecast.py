"""
Hotel Demand Forecast & Decision Support Dashboard.
Powered by Streamlit and Plotly.
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Hotel Demand Dashboard", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. DATA LOADING & CACHING
# ==========================================
@st.cache_data
def load_dashboard_data() -> dict:
    """Loads all required CSV files for the dashboard from the processed_data directory."""
    out_dir = "processed_data"
    
    # Load ML Predictions
    preds_city = pd.read_csv(os.path.join(out_dir, "preds_ml_city.csv"), parse_dates=["Date"])
    preds_resort = pd.read_csv(os.path.join(out_dir, "preds_ml_resort.csv"), parse_dates=["Date"])
    preds_all = pd.read_csv(os.path.join(out_dir, "preds_ml_all.csv"), parse_dates=["Date"])
    
    # Load Historical Training Data (standardizing the date column name to 'Date')
    train_city = pd.read_csv(os.path.join(out_dir, "ts_city_features_train.csv"), parse_dates=["arrival_date"])
    train_resort = pd.read_csv(os.path.join(out_dir, "ts_resort_features_train.csv"), parse_dates=["arrival_date"])
    train_all = pd.read_csv(os.path.join(out_dir, "ts_all_features_train.csv"), parse_dates=["arrival_date"])
    
    for df in [train_city, train_resort, train_all]:
        df.rename(columns={"arrival_date": "Date"}, inplace=True)
    
    # Load Aggregated Tables
    eda_month = pd.read_csv(os.path.join(out_dir, "dashboard_eda_month.csv"))
    eda_weekday = pd.read_csv(os.path.join(out_dir, "dashboard_eda_weekday.csv"))
    staffing_plan = pd.read_csv(os.path.join(out_dir, "dashboard_staffing_plan.csv"))
    metrics = pd.read_csv(os.path.join(out_dir, "evaluation_metrics.csv"))
    
    return {
        "preds_city": preds_city, 
        "preds_resort": preds_resort,
        "preds_all": preds_all,
        "train_city": train_city, 
        "train_resort": train_resort,
        "train_all": train_all,
        "eda_month": eda_month, 
        "eda_weekday": eda_weekday,
        "staffing_plan": staffing_plan, 
        "metrics": metrics
    }

try:
    data = load_dashboard_data()
except Exception as e:
    st.error(f"Error loading data. Ensure all CSVs exist in 'processed_data'. Details: {e}")
    st.stop()

# ==========================================
# 3. UI CONTROLS & STATE
# ==========================================
st.title("Demand Forecast & Decision Support")

# Control Panel
ctrl_col1, ctrl_col2 = st.columns(2)
with ctrl_col1:
    hotel_choice = st.radio("Select Hotel View", ["City", "Resort", "All"], horizontal=True)
with ctrl_col2:
    model_choice = st.selectbox(
        "Select Forecast Model", 
        ["Random Forest", "XGBoost", "All Models", "Seasonal Naive (Pending)", "SARIMA (Pending)", "Prophet (Pending)"]
    )

# Retrieve selected data
hotel_key = hotel_choice.lower()
df_hist = data[f"train_{hotel_key}"]
df_pred = data[f"preds_{hotel_key}"]

# Map dropdown selections to dataset columns
model_col_map = {
    "Random Forest": "pred_rf",
    "XGBoost": "pred_xgboost",
    "Seasonal Naive (Pending)": "pred_rf",  # Placeholder
    "SARIMA (Pending)": "pred_rf",          # Placeholder
    "Prophet (Pending)": "pred_rf"          # Placeholder
}

# ==========================================
# 4. BUSINESS LOGIC (RECOMMENDATIONS)
# ==========================================
def generate_recommendation(forecast_value: float, historical_median: float) -> str:
    """Returns HTML formatted action items based on forecast deviation from the median."""
    if historical_median == 0:
        return "Insufficient data for recommendation."

    diff_pct = ((forecast_value - historical_median) / historical_median) * 100
    
    if diff_pct > 15:
        return (f"🔥 HIGH DEMAND ALERT (+{diff_pct:.1f}%):<br>"
                "- Raise room prices (Dynamic Pricing)<br>"
                "- Tighten booking rules (Min-stay reqs)<br>"
                "- Add extra staff shifts<br>"
                "- Order extra supplies/linens")
    
    elif diff_pct < -15:
        return (f"❄️ LOW DEMAND ALERT ({diff_pct:.1f}%):<br>"
                "- Lower prices & run promotions<br>"
                "- Boost targeted marketing ads<br>"
                "- Reduce labor costs/shifts<br>"
                "- Delay non-urgent spending")
    
    else:
        return (f"✅ NORMAL DEMAND ({diff_pct:.1f}%):<br>"
                "- Standard pricing & staffing levels<br>"
                "- Proceed with regular operations")

# ==========================================
# 5. DATA PREPARATION (WEEKLY AGGREGATION)
# ==========================================
# Aggregate to weekly numbers for the main line chart
hist_weekly = df_hist.set_index("Date")["y_bookings"].resample("W-SUN").sum().reset_index()
pred_weekly = df_pred.set_index("Date").resample("W-SUN").sum().reset_index()

# Calculate the weekly historical median to drive the recommendation system
hist_weekly_median = df_hist["y_bookings"].median() * 7 

# ==========================================
# 6. MAIN CHART: FORECAST VISUALIZATION
# ==========================================
st.subheader(f"Weekly Booking Demand ({hotel_choice} Hotel) - {model_choice}")

fig_forecast = go.Figure()

# Plot: Historical Baseline
fig_forecast.add_trace(go.Scatter(
    x=hist_weekly["Date"], 
    y=hist_weekly["y_bookings"], 
    mode="lines", 
    name="Historical History", 
    line=dict(color="white")
))

# Plot: Forecast Line(s)
if model_choice == "All Models":
    models_to_plot = {
        "Random Forest": {"column": "pred_rf", "color": "#00d2b4"},
        "XGBoost": {"column": "pred_xgboost", "color": "#ff6b6b"}
    }
    last_hist_date = hist_weekly["Date"].iloc[-1]
    last_hist_value = hist_weekly["y_bookings"].iloc[-1]
    
    for name, model in models_to_plot.items():
        col_name = model["column"]
        
        # Generate recommendations for this specific model
        recommendations = pred_weekly[col_name].apply(
            lambda x: generate_recommendation(x, hist_weekly_median)
        )
        
        fig_forecast.add_trace(go.Scatter(
            # Start from the last historical observation so both forecasts visibly
            # branch off the same baseline point.
            x=pd.concat([pd.Series([last_hist_date]), pred_weekly["Date"]], ignore_index=True),
            y=pd.concat([pd.Series([last_hist_value]), pred_weekly[col_name]], ignore_index=True),
            mode="lines", 
            name=f"Forecast ({name})", 
            line=dict(color=model["color"], dash="dash"),
            hovertemplate=f"<b>Week:</b> %{{x}}<br><b>{name}:</b> %{{y:.0f}} bookings<br><br>%{{customdata}}<extra></extra>",
            customdata=["Historical baseline"] + recommendations.tolist(),
            showlegend=True
        ))
else:
    selected_col = model_col_map[model_choice]
    
    # Generate recommendations for the single selected model
    pred_weekly["Recommendation"] = pred_weekly[selected_col].apply(
        lambda x: generate_recommendation(x, hist_weekly_median)
    )
    
    fig_forecast.add_trace(go.Scatter(
        x=pred_weekly["Date"], 
        y=pred_weekly[selected_col], 
        mode="lines", 
        name="Forecast", 
        line=dict(color="#00d2b4", dash="dash"),
        hovertemplate="<b>Week:</b> %{x}<br><b>Forecast:</b> %{y:.0f} bookings<br><br>%{customdata}<extra></extra>",
        customdata=pred_weekly["Recommendation"]
    ))

# Chart Styling
fig_forecast.update_layout(
    plot_bgcolor="#162127", 
    paper_bgcolor="#162127", 
    font_color="white",
    hovermode="x unified", 
    height=400, 
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_forecast, use_container_width=True)

# ==========================================
# 7. LOWER DASHBOARD WIDGETS
# ==========================================
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown("#### Compare forecast models")
    hotel_metrics = data["metrics"][data["metrics"]["Hotel"] == hotel_choice]
    st.dataframe(hotel_metrics, hide_index=True, use_container_width=True)

    
    with st.expander("View Model Optimization & Feature Importance 🛠️"):
        st.markdown("### Hyperparameter Tuning")
        st.markdown("""
        **XGBoost:** `n_estimators=300`, `learning_rate=0.05`, `max_depth=5`, `subsample=0.8`  
        **Random Forest:** `n_estimators=300`, `max_depth=15`, `min_samples_leaf=2`
        """)
        
        st.markdown("### Feature Importance (Top 10)")
        imp_col1, imp_col2 = st.columns(2)
        
        # XGBoost Chart
        xgb_imp_path = os.path.join("processed_data", f"feature_importance_xgb_{hotel_key}.csv")
        if os.path.exists(xgb_imp_path):
            df_xgb = pd.read_csv(xgb_imp_path).head(10)
            with imp_col1:
                st.markdown("**XGBoost**")
                fig_xgb = px.bar(df_xgb, x="Importance", y="Feature", orientation='h')
                fig_xgb.update_layout(plot_bgcolor="#162127", paper_bgcolor="#162127", font_color="white", height=300, margin=dict(l=0, r=0, t=0, b=0))
                fig_xgb.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_xgb, use_container_width=True)

        # Random Forest Chart
        rf_imp_path = os.path.join("processed_data", f"feature_importance_rf_{hotel_key}.csv")
        if os.path.exists(rf_imp_path):
            df_rf = pd.read_csv(rf_imp_path).head(10)
            with imp_col2:
                st.markdown("**Random Forest**")
                fig_rf = px.bar(df_rf, x="Importance", y="Feature", orientation='h')
                fig_rf.update_layout(plot_bgcolor="#162127", paper_bgcolor="#162127", font_color="white", height=300, margin=dict(l=0, r=0, t=0, b=0))
                fig_rf.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_rf, use_container_width=True)
   

with row1_col2:
    st.markdown("#### Staffing plan, next 4 weeks")
    st.caption(" front-desk per 600 bookings, 1 housekeeper per 75 room-nights.")
    st.dataframe(data["staffing_plan"], hide_index=True, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

# Common chart layout settings
chart_layout = dict(
    plot_bgcolor="#162127", 
    paper_bgcolor="#162127", 
    font_color="white", 
    margin=dict(l=0, r=0, t=10, b=0), 
    height=250
)

with row2_col1:
    st.markdown("#### Bookings by month")
    fig_month = px.bar(data["eda_month"], x="Month", y="Bookings")
    fig_month.update_traces(marker_color="#00d2b4")
    fig_month.update_layout(**chart_layout)
    st.plotly_chart(fig_month, use_container_width=True)

with row2_col2:
    st.markdown("#### Bookings by weekday")
    fig_week = px.bar(data["eda_weekday"], x="Weekday", y="Bookings")
    fig_week.update_traces(marker_color="#00d2b4")
    fig_week.update_layout(**chart_layout)
    st.plotly_chart(fig_week, use_container_width=True)
