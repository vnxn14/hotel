# Project Guidelines: Hotel Booking Analytics (Cancellation Risk, Clustering & Demand Forecasting)

## 1. Environment & Execution Rules
- **Python Execution:** Always execute Python scripts using `"C:\Users\User\AppData\Local\Python\pythoncore-3.14-64\python.exe"` (or `py`).
- **Package Management:** Install any missing libraries from `requirements.txt` using `"C:\Users\User\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m pip install -r requirements.txt`.
- **Verification:** Whenever creating or modifying a script, always run it to verify execution and confirm all checks print `PASS` before finishing a task.
- **Data Integrity:** Never delete or overwrite the raw Kaggle dataset (`hotel_bookings.csv`).

---

## 2. Project Overview & Domain Context
- **Objective:** Build an end-to-end Machine Learning decision support system with three main branches:
  1. **Cancellation Risk:** Predict cancellations at booking creation time and prioritize high-risk bookings.
  2. **Guest Segments:** Cluster guests based on booking behaviors to enable targeted marketing and personalized service.
  3. **Demand Forecasting:** Predict future weekly/daily demand and calculate staffing requirements.
- **Dataset:** Kaggle Hotel Booking Demand dataset (`City Hotel` and `Resort Hotel`).

---

## 3. Critical Rules & Preprocessing Workflow
- **Two-Stage Preprocessing Pipeline:**
  1. **`shared_preprocessing.py`:** Applies universal rule-based cleaning (e.g., date parsing, missing value fills, feature creation like `total_stay_nights`, `booking_value`). **Must NOT drop duplicates or leakage columns**, as the dashboard EDA and Forecasting branches require them.
  2. **Branch-Specific Preprocessing:**
     - *Cancellation Branch:* Drops duplicates and strict leakage columns (`reservation_status`, `reservation_status_date`, `booking_changes`).
     - *Clustering Branch:* Drops duplicates and filters to include ONLY completed/successful stays (where `is_canceled == 0`). Uses behavioral features (e.g., lead time, length of stay, special requests, ADR). Standard scaling is mandatory before clustering.
     - *Forecasting Branch (`prep_forecasting.py`):* Keeps duplicates (every booking is real demand), aggregates by date/week, and engineers time-series features (lags, rolling means, time index).

---

## 4. Machine Learning Models & Evaluation Standards

### Task 1: Cancellation Risk (Classification)
- **Required Models:** Logistic Regression, Decision Tree, Random Forest, XGBoost, CatBoost, LightGBM.
- **Required Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, and Confusion Matrix.

### Task 2: Guest Segments (Clustering)
- **Required Algorithms:** K-Means, DBSCAN, and Agglomerative Hierarchical Clustering.
- **Required Metrics:** Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
- **Analysis:** Must generate cluster profiles (mean/median of original features per cluster) and visualize using PCA or t-SNE 2D/3D scatter plots.

### Task 3: Demand Forecasting (Time Series / Regression)
- **Required Models:** Seasonal Naive (Baseline), SARIMA, Prophet, XGBoost, Random Forest.
- **Required Metrics:** MAE, RMSE, and MASE (Mean Absolute Scaled Error).
- **Validation:** Must use chronological splitting (never shuffle time-series data).

---

## 5. System Functions & Business Logic

### Core Functions (Must Implement)
1. **Cancellation Risk Prediction:** Output calibrated probability $P(\text{Cancellation})$.
2. **Risk Level Classification:** Categorize probability into **Low**, **Medium**, or **High** risk tiers.
3. **Booking Revenue at Risk:** Calculate using the exact formula:
   $$\text{GrossRevenueAtRisk} = P(\text{Cancellation}) \times \text{ADR} \times \text{Total Nights}$$
4. **Guest Segmentation Profiling:** Assign incoming bookings to a designated customer persona/cluster to trigger specific marketing workflows (e.g., "Family Planners", "Last-Minute Business", "Budget Weekend").
5. **Demand Forecasting:** Predict `y_bookings` (count of bookings) and `y_room_nights` (total stay nights) over a continuous daily/weekly calendar.
6. **Staffing Plan Calculation:** Derive 4-week future staffing needs using business rules:
   - *Housekeeping:* 1 housekeeper per 48 forecasted room-nights.
   - *Front Desk:* 1 front-desk staff per 75 forecasted bookings.
7. **Model Performance Dashboard:** Display classification comparison metrics (ROC/PR), clustering visualizations (PCA scatters, radar charts for profiles), forecasting comparisons (MASE/MAE/RMSE), and EDA charts.

### Recommended / Secondary Functions
- **Booking Priority Ranking:** Sort bookings by `GrossRevenueAtRisk` descending for actionable concierge follow-up.
- **Risk Explanation (SHAP):** Provide feature-level SHAP explanations for individual cancellation predictions.
- **Hotel Type Analysis:** Compare performance, clustering profiles, and cancellation behavior between `City Hotel` and `Resort Hotel`.