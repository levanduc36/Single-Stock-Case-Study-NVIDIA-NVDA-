import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


# =========================================================
# 1. LOAD AND PREPARE DATA
# =========================================================

df = pd.read_csv("HistoricalData_1754061510662.csv")

# Standardize column name
if "Close/Last" in df.columns:
    df = df.rename(columns={"Close/Last": "Close"})

# Remove $ and commas from price columns if needed
for col in ["Close", "Open", "High", "Low"]:
    if col in df.columns and df[col].dtype == "object":
        df[col] = df[col].replace(r"[\$,]", "", regex=True).astype(float)

# Parse date and sort
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

# Create log price and log return
df["log_price"] = np.log(df["Close"])
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

df = df.dropna().copy()

print("Data prepared successfully.")
print(df[["Close", "log_price", "log_return"]].head())


# =========================================================
# 2. TRAIN / TEST SPLIT
# =========================================================
# Topic 5 mentions hold-out validation and suggests temporal split.
# Since your task is model selection only, we mainly use train for AIC/BIC,
# but we still keep a hold-out split ready if your team wants it later.

split_idx = int(len(df) * 0.8)

train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

train_log_price = train["log_price"]
train_log_return = train["log_return"] * 100  # scale for GARCH stability

print("\nTrain size:", len(train))
print("Test size:", len(test))


# =========================================================
# 3. ARIMA MODEL SELECTION
# =========================================================
# We model log_price with ARIMA(p,1,q)
# d=1 is reasonable because stock prices are typically non-stationary.

arima_candidates = [
    (1, 1, 1),
    (1, 1, 2),
    (2, 1, 1),
    (2, 1, 2),
    (3, 1, 1),
    (3, 1, 2),
    (4, 1, 1)
]

arima_results = []

print("\nRunning ARIMA model selection...")

for order in arima_candidates:
    try:
        model = ARIMA(train_log_price, order=order)
        fitted = model.fit()

        arima_results.append({
            "Model": f"ARIMA{order}",
            "order": order,
            "AIC": fitted.aic,
            "BIC": fitted.bic
        })

        print(f"Done: ARIMA{order} | AIC={fitted.aic:.3f} | BIC={fitted.bic:.3f}")

    except Exception as e:
        print(f"Failed: ARIMA{order} | Error: {e}")

arima_results_df = pd.DataFrame(arima_results)

if not arima_results_df.empty:
    arima_results_df = arima_results_df.sort_values(["AIC", "BIC"]).reset_index(drop=True)

    print("\n=== ARIMA Model Selection Results ===")
    print(arima_results_df)

    best_arima = arima_results_df.iloc[0]
    print("\nBest ARIMA model based on AIC/BIC:")
    print(best_arima)
else:
    print("\nNo ARIMA model was fitted successfully.")


# =========================================================
# 4. GARCH MODEL SELECTION
# =========================================================
# We model volatility on log returns.
# Start with simple GARCH structures.

garch_candidates = [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2)
]

garch_results = []

print("\nRunning GARCH model selection...")

for p, q in garch_candidates:
    try:
        garch = arch_model(
            train_log_return,
            mean="Zero",      # keep mean simple for volatility selection
            vol="GARCH",
            p=p,
            q=q,
            dist="normal"
        )

        fitted = garch.fit(disp="off")

        garch_results.append({
            "Model": f"GARCH({p},{q})",
            "p": p,
            "q": q,
            "AIC": fitted.aic,
            "BIC": fitted.bic
        })

        print(f"Done: GARCH({p},{q}) | AIC={fitted.aic:.3f} | BIC={fitted.bic:.3f}")

    except Exception as e:
        print(f"Failed: GARCH({p},{q}) | Error: {e}")

garch_results_df = pd.DataFrame(garch_results)

if not garch_results_df.empty:
    garch_results_df = garch_results_df.sort_values(["AIC", "BIC"]).reset_index(drop=True)

    print("\n=== GARCH Model Selection Results ===")
    print(garch_results_df)

    best_garch = garch_results_df.iloc[0]
    print("\nBest GARCH model based on AIC/BIC:")
    print(best_garch)
else:
    print("\nNo GARCH model was fitted successfully.")


# =========================================================
# 5. FINAL SUMMARY
# =========================================================
print("\n================ FINAL SELECTED MODELS ================")

if not arima_results_df.empty:
    print(f"Best Mean Model      : {best_arima['Model']}")
    print(f"AIC = {best_arima['AIC']:.3f}, BIC = {best_arima['BIC']:.3f}")

if not garch_results_df.empty:
    print(f"Best Volatility Model: {best_garch['Model']}")
    print(f"AIC = {best_garch['AIC']:.3f}, BIC = {best_garch['BIC']:.3f}")

print("=======================================================")