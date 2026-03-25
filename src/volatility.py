import numpy as np

def compute_volatility(df, window=21):
    df['volatility'] = df['log_return'].rolling(window).std() * np.sqrt(252)
    return df