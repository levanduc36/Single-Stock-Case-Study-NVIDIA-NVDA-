import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot configuration
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
try:
    # Replace with your actual file path
    file_path = "HistoricalData_1754061510662 (1).csv" 
    df = pd.read_csv(file_path)

    # Standardize column names
    if 'Close/Last' in df.columns:
        df.rename(columns={'Close/Last': 'Close'}, inplace=True)
    df.columns = df.columns.str.strip()

    # Remove '$' sign and convert columns to numeric
    cols_to_fix = ['Close', 'Open', 'High', 'Low']
    for col in cols_to_fix:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('$', '', regex=False)
            df[col] = pd.to_numeric(df[col])

    # Set Date as Index
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Calculate Log Returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True) # Remove the first row (NaN)

    print("✅ Data is ready for plotting!")

except Exception as e:
    print(f"❌ Error: {e}")

# ==========================================
# 2. EDA PLOTS
# ==========================================

# --- PLOT 1: Price Trend & Rolling Mean ---
window_size = 20 # 20 days (approx. 1 trading month)
df['Rolling_Mean'] = df['Close'].rolling(window=window_size).mean()

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'], label='Close Price', color='navy', alpha=0.5)
plt.plot(df.index, df['Rolling_Mean'], label=f'{window_size}-Day Rolling Mean', color='red', linewidth=2)
plt.title('Price Trend: Close Price vs. Moving Average', fontsize=16)
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# --- PLOT 2: Volatility Clustering ---
# Calculate Rolling Standard Deviation of Returns
df['Volatility'] = df['Log_Return'].rolling(window=window_size).std()

plt.figure(figsize=(14, 4))
plt.plot(df.index, df['Volatility'], color='orange', label=f'{window_size}-Day Rolling Volatility')
plt.title('Risk Analysis: Volatility Clustering', fontsize=16)
plt.ylabel('Standard Deviation (Std Dev)')
plt.legend()
plt.show()

# --- PLOT 3: Time Series Decomposition ---
# Decompose into Trend, Seasonal, and Residual components
# period=252 represents the approximate number of trading days in a year
decomposition = seasonal_decompose(df['Close'], model='multiplicative', period=252)

fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.suptitle('Time Series Decomposition', fontsize=16, y=1.02) # y=1.02 avoids title overlap
plt.show()

# --- PLOT 4: Autocorrelation (ACF) ---
# Plot ACF for both Price and Returns for comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

#Log Return
# 1. Vẽ ACF (Autocorrelation)
# Giúp xác định tham số MA (Moving Average) - q
plot_acf(df['Log_Return'], ax=axes[0], lags=40, title='Autocorrelation (ACF) - Log Returns')
axes[0].set_xlabel('Lags (Độ trễ)')
axes[0].set_ylabel('Correlation')

# 2. Vẽ PACF (Partial Autocorrelation)
# Giúp xác định tham số AR (AutoRegressive) - p
plot_pacf(df['Log_Return'], ax=axes[1], lags=40, title='Partial Autocorrelation (PACF) - Log Returns')
axes[1].set_xlabel('Lags (Độ trễ)')
axes[1].set_ylabel('Correlation')
plt.show()


#Close Price
# 1. ACF for Close Price
# Expectation: Slow decay (signs of non-stationarity)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(df['Close'], ax=axes[0], lags=40, title='Autocorrelation (ACF) - Close Price')
axes[0].set_xlabel('Lags')
axes[0].set_ylabel('Correlation')

# 2. PACF for Close Price
# Expectation: Significant spike at lag 1, then drops
plot_pacf(df['Close'], ax=axes[1], lags=40, title='Partial Autocorrelation (PACF) - Close Price')
axes[1].set_xlabel('Lags')
axes[1].set_ylabel('Correlation')

plt.tight_layout()
plt.show()
