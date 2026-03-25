import numpy as np

def add_returns(df):
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    return df.dropna()