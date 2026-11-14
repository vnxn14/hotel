# Project Guidelines: Hotel Booking Cancellation Risk Assessment & Decision Support System

## 1. Environment & Execution Rules
- **Python Execution:** Always execute Python scripts using `"C:\Users\User\AppData\Local\Python\pythoncore-3.14-64\python.exe"` (or `py`).
- **Package Management:** Install any missing libraries from `requirements.txt` using `"C:\Users\User\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m pip install -r requirements.txt`.
- **Verification:** Whenever creating or modifying a script, always run it to verify execution and confirm all checks print `PASS` before finishing a task.
- **Data Integrity:** Never delete or overwrite the raw Kaggle dataset (`hotel_bookings.csv`).

---

## 2. Project Overview & Domain Context
- **Objective:** Build an end-to-end Machine Learning decision support system to predict hotel booking cancellations at **booking creation time** and prioritize high-risk bookings based on revenue exposure.
- **Dataset:** Kaggle Hotel Booking Demand dataset (`City Hotel` and `Resort Hotel`).
- **Target Variable:** `is_canceled` (binary classification).

---

## 3. Critical Rules & Data Leakage Prevention
- **Strict Prediction Timing:** Predictions occur at **booking creation time**. Never include post-booking or outcome-leaking columns in training features.
- **Mandatory Dropped Columns:** Always exclude `reservation_status`, `reservation_status_date`, and `booking_changes` (and any other post-booking fields).
- **Preprocessing Consistency (`preprocessing.py`):**
  - `company` and `agent` must be converted into binary flags (`1` if not null/numeric, `0` if null) rather than imputed with raw IDs.
  - Maintain required engineered features: `total_stay_nights`, `total_guests`, `has_children`, `booking_value` (`ADR * total_nights`), `historical_cancellation_rate`, `weekend_stay_ratio`, `weekday_stay_ratio`, `lead_time_category`, `arrival_season`, `is_weekend_arrival`, and `has_agent`.
  - Keep scaling applied only to models that require it (e.g., Logistic Regression) and preserve clean Train / Validation / Test splits without data leakage.

---

## 4. Machine Learning Models & Evaluation Standards
- **Required Models:** Train and compare all 6 recommended models:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. XGBoost
  5. CatBoost
  6. LightGBM
- **Required Metrics:** Evaluate and log all 7 metrics for every model:
  - Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, and Confusion Matrix.

---

## 5. System Functions & Business Logic
When building the modeling, evaluation, or dashboard/app components, adhere strictly to these priority tiers and formulas:

### Core Functions (Must Implement)
1. **Cancellation Risk Prediction:** Output calibrated probability $P(\text{Cancellation})$ from booking details.
2. **Risk Level Classification:** Categorize probability into **Low**, **Medium**, or **High** risk tiers.
3. **Risk Explanation (SHAP):** Provide feature-level SHAP explanations for individual predictions.
4. **Booking Revenue at Risk:** Calculate using the exact formula:
   $$\text{GrossRevenueAtRisk} = P(\text{Cancellation}) \times \text{ADR} \times \text{Total Nights}$$
5. **Booking Priority Ranking & Actionable Prescriptions:** Sort bookings by `GrossRevenueAtRisk` descending and assign recommended actions:
   - *High Risk + High Revenue:* Trigger manual concierge follow-up.
   - *Medium Risk:* Send automated promotional offer or flexible rebooking email.
   - *Low Risk:* Standard processing.
6. **Model Performance Dashboard:** Display model comparison metrics, Confusion Matrices, and ROC / PR curves.

### Recommended / Secondary Functions
- **Hotel Type Analysis:** Compare performance and cancellation behavior between `City Hotel` and `Resort Hotel`.
- **Room Type Analysis:** Analyze cancellation patterns across `reserved_room_type`.
- **Batch Booking Prediction:** Support CSV upload for multi-booking scoring (lower priority).
- **Optional / Advanced (If time permits):** What-if parameter simulator; Model Reliability Dashboard (Calibration, Robustness, Error Analysis).

---

## 6. Team Collaboration & Git Hygiene
- Do not commit large raw datasets, virtual environments (`.venv/`), or `__pycache__/` folders.
- Keep code modular (separate scripts/modules for `preprocessing`, `training/evaluation`, `shap_explainer`, and `app/dashboard`).
