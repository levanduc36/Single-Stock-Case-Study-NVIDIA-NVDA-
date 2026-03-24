import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("HistoricalData_1754061510662.csv")

# Rename column
df = df.rename(columns={"Close/Last": "Close"})

# Remove $
for col in ["Close", "Open", "High", "Low"]:
    df[col] = df[col].replace(r'[\$,]', '', regex=True).astype(float)

# Date
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

# Log price and log return
df['log_price'] = np.log(df['Close'])
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()

# Plot: Stock price over time
plt.figure(figsize=(12,5))
plt.plot(df["Close"])
plt.title("NVDA Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid()
plt.show()

# Rolling mean & std
rolling_mean = df["Close"].rolling(window=20).mean()
rolling_std = df["Close"].rolling(window=20).std()

plt.figure(figsize=(12,6))
plt.plot(df["Close"], label="Original Price")
plt.plot(rolling_mean, label="Rolling Mean (20 days)")
plt.plot(rolling_std, label="Rolling Std (20 days)")
plt.legend()
plt.title("Rolling Mean & Std (20 Trading Days)")
plt.xlabel("Date")
plt.ylabel("Value")
plt.grid()
plt.show()

# ACF of Stock price
from statsmodels.graphics.tsaplots import plot_acf

fig, ax = plt.subplots(figsize=(10,4))
plot_acf(df["Close"], lags=50, ax=ax)
ax.set_title("ACF of NVDA Stock Price")
ax.grid()
plt.show()

# Log transformed price
plt.figure(figsize=(12,5))
plt.plot(df['log_price'])
plt.title("Log Transformed Price")
plt.xlabel("Time")
plt.ylabel("Log Price")
plt.grid()
plt.show()

# Log return
plt.figure(figsize=(12,5))
plt.plot(df['log_return'])
plt.title("Log Return")
plt.xlabel("Time")
plt.ylabel("Log Return")
plt.grid()
plt.show()

# Rolling statistics
rolling_mean = df['log_return'].rolling(20).mean()
rolling_std = df['log_return'].rolling(20).std()

plt.figure(figsize=(12,5))
plt.plot(df['log_return'], label='Log Return')
plt.plot(rolling_mean, label='Rolling Mean (20)')
plt.plot(rolling_std, label='Rolling Std (20)')
plt.title("Rolling Statistics (Log Return)")
plt.legend()
plt.grid()
plt.show()