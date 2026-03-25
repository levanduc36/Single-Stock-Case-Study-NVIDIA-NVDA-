import pandas as pd

def clean_data(df):
    df = df.rename(columns={"Close/Last": "Close"})
    df.columns = df.columns.str.strip()

    df.replace({r'\$': '', ',': ''}, regex=True, inplace=True)

    for col in ['Close','Open','High','Low','Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    return df.dropna()
    