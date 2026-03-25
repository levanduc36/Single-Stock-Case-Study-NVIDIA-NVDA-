from src.data_loader import load_data
from src.preprocessing import clean_data
from src.features import add_returns

df = load_data("data/raw/HistoricalData_1754061510662.csv")
df = clean_data(df)
df = add_returns(df)

print(df.head())