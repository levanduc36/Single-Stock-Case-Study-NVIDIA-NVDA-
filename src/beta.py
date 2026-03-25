import numpy as np
import pandas as pd
import yfinance as yf

def compute_beta(df):
    start = df.index.min().strftime('%Y-%m-%d')
    end = df.index.max().strftime('%Y-%m-%d')

    spy = yf.download('SPY', start=start, end=end)

    spy = spy[['Close']]
    spy.columns = ['SPY_Close']

    merged = df[['Close']].rename(columns={'Close': 'NVDA_Close'}).join(spy, how='inner')

    merged['NVDA_Return'] = np.log(merged['NVDA_Close'] / merged['NVDA_Close'].shift(1))
    merged['SPY_Return'] = np.log(merged['SPY_Close'] / merged['SPY_Close'].shift(1))

    merged = merged.dropna()

    beta = merged[['NVDA_Return','SPY_Return']].cov().iloc[0,1] / merged['SPY_Return'].var()
    return beta