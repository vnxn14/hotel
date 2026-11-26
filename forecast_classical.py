"""Forecast daily hotel bookings with classical time-series models.

Input files are created by ``prep_forecasting.py``:
    processed_data/forecasting_{city,all,resort}.csv

Outputs:
    forecasts.csv       Actual values and predictions for each test date
    metrics.csv         MAE, RMSE, and weekly Seasonal Naive MASE
    diagnostics.txt     Training-series diagnostics and selected SARIMAX orders
    plots/              Diagnostics and forecast comparison plots

The last ``test_ratio`` of dates is held out. Each model forecasts that full
test window without refitting on test observations.
"""
import argparse
import itertools
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

SERIES_NAMES = {"all": "All hotels", "city": "City Hotel", "resort": "Resort Hotel"}
WEEK = 7
FOURIER_K = 3
SARIMA_ORDERS = ((1, 0, 1), (2, 0, 2), (1, 1, 1))
SEASONAL_ORDERS = ((1, 1, 1, WEEK), (0, 1, 1, WEEK))


# Data and evaluation
def load_series(data_dir: Path, key: str, test_ratio: float) -> tuple[pd.Series, pd.Series]:
    """Load one prepared daily series and split it chronologically."""
    path = data_dir / f"forecasting_{key}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared forecasting file not found: {path}. Run prep_forecasting.py first.")

    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if "y_bookings" not in frame.columns:
        raise ValueError(f"{path} must contain the 'y_bookings' column")

    series = frame["y_bookings"].astype(float).sort_index()
    if series.index.has_duplicates or series.index.isna().any():
        raise ValueError(f"{path} contains duplicate or invalid dates")
    series.index.name = "Date"
    if not series.index.equals(pd.date_range(series.index[0], series.index[-1], freq="D")):
        raise ValueError(f"{path} must contain one row for every calendar day")

    cutoff = int(len(series) * (1 - test_ratio))
    if cutoff <= 0 or cutoff >= len(series):
        raise ValueError("test_ratio must leave at least one date in both train and test sets")

    train, test = series.iloc[:cutoff], series.iloc[cutoff:]
    train.index.freq = test.index.freq = "D"
    return train, test


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, train: pd.Series) -> dict[str, float]:
    """Calculate MAE, RMSE, and MASE scaled by weekly naive training error."""
    errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    scale = np.mean(np.abs(train.values[WEEK:] - train.values[:-WEEK]))
    mase = np.mean(np.abs(errors)) / scale if scale else np.nan
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MASE": float(mase),
    }


# Training-series diagnostics
def save_diagnostics(train: pd.Series, key: str, plot_dir: Path, log_lines: list[str]) -> None:
    """Save stationarity, autocorrelation, decomposition, and calendar plots."""
    adf_raw = adfuller(train)[1]
    adf_diff = adfuller(train.diff().dropna())[1]
    adf_weekly = adfuller(train.diff(WEEK).dropna())[1]
    log_lines.append(
        f"[{key}] ADF p-value: raw {adf_raw:.3f} | d=1 {adf_diff:.3f} | "
        f"seasonal diff (lag 7) {adf_weekly:.3f}"
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    plot_acf(train, lags=40, ax=axes[0, 0], title="ACF - raw")
    plot_pacf(train, lags=40, ax=axes[0, 1], title="PACF - raw", method="ywm")
    weekly_diff = train.diff(WEEK).dropna()
    plot_acf(weekly_diff, lags=40, ax=axes[1, 0], title="ACF - weekly difference")
    plot_pacf(weekly_diff, lags=40, ax=axes[1, 1], title="PACF - weekly difference", method="ywm")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{key}_acf_pacf.png", dpi=120)
    plt.close(fig)

    decomposition = STL(train, period=WEEK, robust=True).fit()
    fig = decomposition.plot()
    fig.set_size_inches(10, 7)
    fig.savefig(plot_dir / f"{key}_stl_weekly.png", dpi=120)
    plt.close(fig)

    weekday_means = train.groupby(train.index.dayofweek).mean()
    month_means = train.groupby(train.index.month).mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], weekday_means.values)
    axes[0].set_title("Average bookings by weekday")
    axes[1].bar(month_means.index, month_means.values)
    axes[1].set_title("Average bookings by month (training period)")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{key}_seasonality.png", dpi=120)
    plt.close(fig)


# Forecast models
def seasonal_naive(train: pd.Series, test_index: pd.DatetimeIndex, period: int) -> pd.DataFrame:
    """Repeat the most recent training observation from the same seasonal position."""
    last_train_date = train.index[-1]
    forecasts = []
    for date in test_index:
        periods_back = int(np.ceil((date - last_train_date).days / period))
        source_date = date - pd.Timedelta(days=periods_back * period)
        forecasts.append(train.loc[source_date])
    return pd.DataFrame({"yhat": forecasts}, index=test_index)


def fourier_features(index: pd.DatetimeIndex, k: int = FOURIER_K) -> pd.DataFrame:
    """Create smooth yearly sine/cosine features for SARIMAX."""
    days = np.asarray((index - pd.Timestamp("2015-01-01")).days, dtype=float)
    features = {}
    for harmonic in range(1, k + 1):
        angle = 2 * np.pi * harmonic * days / 365.25
        features[f"sin{harmonic}"] = np.sin(angle)
        features[f"cos{harmonic}"] = np.cos(angle)
    return pd.DataFrame(features, index=index)


def fit_sarimax(
    train: pd.Series,
    test_index: pd.DatetimeIndex,
    log_lines: list[str],
    key: str,
) -> pd.DataFrame:
    """Choose a small SARIMAX specification by training AIC and forecast the test window."""
    exog_train = fourier_features(train.index)
    exog_test = fourier_features(test_index)
    best = None

    for order, seasonal_order in itertools.product(SARIMA_ORDERS, SEASONAL_ORDERS):
        try:
            result = SARIMAX(
                train,
                exog=exog_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)
            if best is None or result.aic < best[0]:
                best = (result.aic, order, seasonal_order, result)
        except Exception as exc:
            logging.warning("SARIMAX %s x %s failed: %s", order, seasonal_order, exc)

    if best is None:
        raise RuntimeError(f"All SARIMAX parameter combinations failed for '{key}'")

    aic, order, seasonal_order, result = best
    log_lines.append(
        f"[{key}] SARIMAX order {order} x {seasonal_order} + Fourier K={FOURIER_K} (AIC {aic:.1f})"
    )
    forecast = result.get_forecast(steps=len(test_index), exog=exog_test)
    interval = forecast.conf_int(alpha=0.05)
    return pd.DataFrame(
        {
            "yhat": forecast.predicted_mean.to_numpy(),
            "lower": interval.iloc[:, 0].to_numpy(),
            "upper": interval.iloc[:, 1].to_numpy(),
        },
        index=test_index,
    )


def fit_prophet(
    train: pd.Series,
    test_index: pd.DatetimeIndex,
    holidays: str,
) -> pd.DataFrame:
    """Fit Prophet with weekly/yearly seasonality and return its 95% forecast interval."""
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError("Prophet is required. Install it with `python -m pip install prophet`.") from exc

    model = Prophet(
        interval_width=0.95,
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
    )
    if holidays:
        model.add_country_holidays(country_name=holidays)

    training_frame = pd.DataFrame({"ds": train.index, "y": train.values})
    model.fit(training_frame)
    forecast = model.predict(pd.DataFrame({"ds": test_index}))
    return pd.DataFrame(
        {
            "yhat": forecast["yhat"].to_numpy(),
            "lower": forecast["yhat_lower"].to_numpy(),
            "upper": forecast["yhat_upper"].to_numpy(),
        },
        index=test_index,
    )


# Run and save one series
def run_series(
    key: str,
    data_dir: Path,
    output_dir: Path,
    test_ratio: float,
    holidays: str,
    log_lines: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit all models for one hotel series and save its forecast plot."""
    train, test = load_series(data_dir, key, test_ratio)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    logging.info(
        "[%s] train %s days (%s to %s), test %s days (from %s)",
        key, len(train), train.index[0].date(), train.index[-1].date(), len(test), test.index[0].date(),
    )
    save_diagnostics(train, key, plot_dir, log_lines)

    forecasts = {
        "Seasonal Naive (weekly)": seasonal_naive(train, test.index, WEEK),
        "Seasonal Naive (yearly)": seasonal_naive(train, test.index, 365),
        "SARIMAX": fit_sarimax(train, test.index, log_lines, key),
        "Prophet": fit_prophet(train, test.index, holidays),
    }

    metric_rows = []
    forecast_frames = []
    for model_name, prediction in forecasts.items():
        prediction["yhat"] = prediction["yhat"].clip(lower=0)
        metric_rows.append({
            "key": key,
            "model": model_name,
            **calculate_metrics(test, prediction["yhat"], train),
        })
        forecast_frames.append(pd.DataFrame({
            "key": key,
            "model": model_name,
            "date": test.index,
            "y_true": test.values,
            "yhat": prediction["yhat"].values,
            "lower": prediction["lower"] if "lower" in prediction else np.nan,
            "upper": prediction["upper"] if "upper" in prediction else np.nan,
        }))
        logging.info(
            "[%s] %-26s MAE %.1f  RMSE %.1f  MASE %.2f",
            key, model_name, metric_rows[-1]["MAE"], metric_rows[-1]["RMSE"], metric_rows[-1]["MASE"],
        )

    _save_forecast_plot(train, test, forecasts, key, plot_dir)
    return pd.DataFrame(metric_rows), pd.concat(forecast_frames, ignore_index=True)


def _save_forecast_plot(
    train: pd.Series,
    test: pd.Series,
    forecasts: dict[str, pd.DataFrame],
    key: str,
    plot_dir: Path,
) -> None:
    """Plot recent training history, test actuals, and model forecasts."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(train.index[-120:], train.values[-120:], color="grey", label="Train (history)")
    ax.plot(test.index, test.values, color="black", label="Actual (test)")
    for model_name, prediction in forecasts.items():
        ax.plot(test.index, prediction["yhat"], label=model_name, linewidth=1.2)

    prophet_forecast = forecasts["Prophet"]
    ax.fill_between(
        test.index,
        prophet_forecast["lower"].clip(lower=0),
        prophet_forecast["upper"],
        alpha=0.12,
        label="Prophet 95%",
    )
    ax.set_title(f"Daily booking demand - {SERIES_NAMES[key]}")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{key}_forecast.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("processed_data"),
                        help="Directory containing forecasting_<key>.csv files")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/forecasting_classical"),
                        help="Directory for metrics, forecasts, diagnostics, and plots")
    parser.add_argument("--keys", nargs="+", choices=SERIES_NAMES, default=list(SERIES_NAMES),
                        help="Series to forecast (default: all city resort)")
    parser.add_argument("--test-ratio", type=float, default=0.20,
                        help="Fraction of latest dates reserved for testing")
    parser.add_argument("--holidays", default="PT",
                        help="Prophet country code for holidays; pass an empty string to disable")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []
    all_metrics = []
    all_forecasts = []
    for key in args.keys:
        metrics, forecasts = run_series(
            key, args.data_dir, args.out_dir, args.test_ratio, args.holidays, log_lines
        )
        all_metrics.append(metrics)
        all_forecasts.append(forecasts)

    metrics_path = args.out_dir / "metrics.csv"
    forecasts_path = args.out_dir / "forecasts.csv"
    diagnostics_path = args.out_dir / "diagnostics.txt"
    pd.concat(all_metrics, ignore_index=True).round(3).to_csv(metrics_path, index=False)
    pd.concat(all_forecasts, ignore_index=True).to_csv(forecasts_path, index=False)
    diagnostics_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines))
    logging.info("Saved metrics.csv, forecasts.csv, diagnostics.txt, and plots/ in '%s'", args.out_dir)


if __name__ == "__main__":
    main()
