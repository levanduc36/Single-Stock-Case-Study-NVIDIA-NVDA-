import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox


# =========================================================
# WEEK 5: PRICE FORECASTING + RETURN MODELING + MODEL SELECTION
# =========================================================
# This script is intentionally compact and focused on the Week 5 tasks.
# It includes:
# 1) Price forecasting
# 2) Return modeling
# 3) Model selection and evaluation


# =========================================================
# 1. LOAD AND CLEAN DATA
# =========================================================
DATA_PATH = "HistoricalData_1754061510662.csv"  # change if your filename is different
TEST_SIZE = 0.20


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if "Close/Last" in df.columns:
        df = df.rename(columns={"Close/Last": "Close"})

    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])

    for col in ["Close", "Open", "High", "Low"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].replace(r"[\$,]", "", regex=True).astype(float)

    if "Volume" in df.columns and df["Volume"].dtype == "object":
        df["Volume"] = df["Volume"].replace(r"[,]", "", regex=True).astype(float)

    df = df.sort_values("Date").reset_index(drop=True)
    df = df[["Date", "Close"]].copy()
    df["log_price"] = np.log(df["Close"])
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna().reset_index(drop=True)

    return df


def temporal_split(df: pd.DataFrame, test_size: float = 0.20):
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


# =========================================================
# 2. METRICS
# =========================================================
def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def evaluate_forecast(model_name, y_true, y_pred):
    return {
        "Model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
    }


# =========================================================
# 3. PRICE FORECASTING
# =========================================================
def trend_forecast(train_close: pd.Series, test_len: int) -> np.ndarray:
    x_train = np.arange(len(train_close)).reshape(-1, 1)
    x_test = np.arange(len(train_close), len(train_close) + test_len).reshape(-1, 1)

    trend_model = LinearRegression()
    trend_model.fit(x_train, np.log(train_close.values))

    pred_log = trend_model.predict(x_test)
    return np.exp(pred_log)



def arima_select_and_forecast(series: pd.Series, test_len: int, candidates, label: str):
    rows = []
    best_fit = None
    best_order = None
    best_aic = np.inf

    for order in candidates:
        try:
            fitted = ARIMA(series, order=order).fit()
            rows.append({
                "Series": label,
                "Order": order,
                "AIC": fitted.aic,
                "BIC": fitted.bic,
            })

            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_fit = fitted
                best_order = order
        except Exception:
            continue

    results_df = pd.DataFrame(rows).sort_values(["AIC", "BIC"]).reset_index(drop=True)

    if best_fit is None:
        raise ValueError(f"No ARIMA model could be fitted for {label}.")

    forecast_res = best_fit.get_forecast(steps=test_len)
    forecast_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)

    return results_df, best_order, forecast_mean, conf_int


# =========================================================
# 4. RETURN MODELING
# =========================================================
def arma_select_and_forecast(train_returns: pd.Series, test_len: int, candidates):
    rows = []
    best_fit = None
    best_order = None
    best_aic = np.inf

    for p, q in candidates:
        try:
            fitted = ARIMA(train_returns, order=(p, 0, q)).fit()
            rows.append({
                "Model": f"ARMA({p},{q})",
                "Order": (p, q),
                "AIC": fitted.aic,
                "BIC": fitted.bic,
            })

            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_fit = fitted
                best_order = (p, q)
        except Exception:
            continue

    results_df = pd.DataFrame(rows).sort_values(["AIC", "BIC"]).reset_index(drop=True)

    if best_fit is None:
        raise ValueError("No ARMA model could be fitted for returns.")

    forecast = best_fit.forecast(steps=test_len)
    residuals = best_fit.resid

    lb_resid = acorr_ljungbox(residuals, lags=[10], return_df=True)

    diagnostics = {
        "Best ARMA Order": best_order,
        "Ljung-Box p-value (lag 10)": float(lb_resid["lb_pvalue"].iloc[0]),
    }

    return results_df, best_order, forecast, residuals, diagnostics


# =========================================================
# 5. MAIN WORKFLOW
# =========================================================
def main():
    df = load_data(DATA_PATH)
    train, test = temporal_split(df, TEST_SIZE)

    print("=" * 70)
    print("WEEK 5 - NVDA TIME SERIES ANALYSIS")
    print("=" * 70)
    print(f"Dataset size : {len(df)}")
    print(f"Train size   : {len(train)}")
    print(f"Test size    : {len(test)}")
    print(f"Date range   : {df['Date'].min().date()} to {df['Date'].max().date()}")

    # -----------------------------
    # Part A. Price Forecasting
    # -----------------------------
    print("\n" + "=" * 70)
    print("PART A. PRICE FORECASTING")
    print("=" * 70)

    price_candidates = [(1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 1, 2), (3, 1, 1)]

    # Trend model
    trend_pred = trend_forecast(train["Close"], len(test))

    # ARIMA on price
    price_sel_df, best_price_order, price_forecast, price_ci = arima_select_and_forecast(
        train["Close"], len(test), price_candidates, "Price"
    )

    # ARIMA on log price
    log_sel_df, best_log_order, log_forecast, log_ci = arima_select_and_forecast(
        train["log_price"], len(test), price_candidates, "Log Price"
    )
    log_price_pred = np.exp(log_forecast)
    log_ci_price = np.exp(log_ci)

    price_eval = pd.DataFrame([
        evaluate_forecast("Trend (Log-Linear)", test["Close"], trend_pred),
        evaluate_forecast(f"ARIMA Price {best_price_order}", test["Close"], price_forecast),
        evaluate_forecast(f"ARIMA Log Price {best_log_order}", test["Close"], log_price_pred),
    ]).sort_values("RMSE").reset_index(drop=True)

    best_price_model = price_eval.iloc[0]["Model"]

    print("\nARIMA selection on PRICE:")
    print(price_sel_df)
    print("\nARIMA selection on LOG PRICE:")
    print(log_sel_df)
    print("\nPrice forecasting performance:")
    print(price_eval)
    print(f"\nSelected price forecasting model: {best_price_model}")

    # -----------------------------
    # Part B. Return Modeling
    # -----------------------------
    print("\n" + "=" * 70)
    print("PART B. RETURN MODELING")
    print("=" * 70)

    return_candidates = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)]

    ret_sel_df, best_ret_order, ret_forecast, ret_resid, ret_diag = arma_select_and_forecast(
        train["log_return"], len(test), return_candidates
    )

    return_eval = pd.DataFrame([
        evaluate_forecast(f"ARMA{best_ret_order}", test["log_return"], ret_forecast)
    ])

    print("\nARMA selection on RETURNS:")
    print(ret_sel_df)
    print("\nReturn forecasting performance:")
    print(return_eval)
    print("\nResidual diagnostics:")
    for k, v in ret_diag.items():
        print(f"{k}: {v}")

    # -----------------------------
    # Part C. Model Selection Summary
    # -----------------------------
    print("\n" + "=" * 70)
    print("PART C. FINAL MODEL SELECTION")
    print("=" * 70)

    summary_df = pd.DataFrame([
        {
            "Task": "Price Forecasting",
            "Selected Model": best_price_model,
            "Criterion": "Lowest test RMSE"
        },
        {
            "Task": "Return Modeling",
            "Selected Model": f"ARMA{best_ret_order}",
            "Criterion": "Lowest AIC (validated on test set)"
        }
    ])

    print(summary_df)

    # =====================================================
    # 6. PLOTS
    # =====================================================
    plt.style.use("seaborn-v0_8-whitegrid")

    # Price Forecasting Plot
    plt.figure(figsize=(14, 6))
    plt.plot(train["Date"], train["Close"], label="Train", linewidth=2)
    plt.plot(test["Date"], test["Close"], label="Actual", linewidth=2)
    plt.plot(test["Date"], trend_pred, "--", linewidth=2, label="Trend (Log-Linear)")
    plt.plot(test["Date"], price_forecast, "--", linewidth=2, label=f"ARIMA Price {best_price_order}")
    plt.plot(test["Date"], log_price_pred, "--", linewidth=2, label=f"ARIMA Log Price {best_log_order}")

    # Confidence interval only for the best ARIMA-based price model
    if "ARIMA Price" in best_price_model:
        lower = price_ci.iloc[:, 0].astype(float).values
        upper = price_ci.iloc[:, 1].astype(float).values
        plt.fill_between(test["Date"], lower, upper, alpha=0.2, label="95% CI")
    elif "ARIMA Log Price" in best_price_model:
        lower = log_ci_price.iloc[:, 0].astype(float).values
        upper = log_ci_price.iloc[:, 1].astype(float).values
        plt.fill_between(test["Date"], lower, upper, alpha=0.2, label="95% CI")

    plt.title("Week 5 - Price Forecasting Comparison", fontsize=15)
    plt.xlabel("Date")
    plt.ylabel("NVDA Close Price")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    # Return Forecast Plot
    plt.figure(figsize=(14, 5))
    plt.plot(train["Date"], train["log_return"], label="Train Returns", alpha=0.7)
    plt.plot(test["Date"], test["log_return"], label="Actual Returns", linewidth=1.8)
    plt.plot(test["Date"], ret_forecast, "--", label=f"Forecast ARMA{best_ret_order}", linewidth=2)
    plt.title("Week 5 - Return Modeling", fontsize=15)
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    # Residual Plot
    plt.figure(figsize=(14, 4))
    plt.plot(ret_resid, linewidth=1.2)
    plt.title(f"Residuals of Selected Return Model ARMA{best_ret_order}", fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
