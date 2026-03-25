from statsmodels.tsa.stattools import adfuller

def adf_test(series):
    result = adfuller(series)
    return result[0], result[1]