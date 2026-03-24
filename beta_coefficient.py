import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Cài đặt giao diện biểu đồ
sns.set_theme(style="whitegrid")

# 1. ĐỌC VÀ LÀM SẠCH DỮ LIỆU NVIDIA TỪ FILE CSV CỦA BẠN
file_path = "HistoricalData_1754061510662.csv"
df_nvda = pd.read_csv(file_path)

# Đổi tên cột và làm sạch ký tự $
if 'Close/Last' in df_nvda.columns:
    df_nvda.rename(columns={'Close/Last': 'Close'}, inplace=True)
df_nvda.columns = df_nvda.columns.str.strip()

# Loại bỏ ký tự $ và chuyển đổi sang số
df_nvda.replace({r'\$': '', r',': ''}, regex=True, inplace=True)

for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
    if col in df_nvda.columns:
        df_nvda[col] = pd.to_numeric(df_nvda[col], errors='coerce')

df_nvda["Date"] = pd.to_datetime(df_nvda["Date"])
df_nvda = df_nvda.set_index("Date").sort_index()
df_nvda.dropna(inplace=True)

# Lấy ngày bắt đầu và ngày kết thúc từ dataset của bạn
start_date = df_nvda.index.min().strftime('%Y-%m-%d')
end_date = df_nvda.index.max().strftime('%Y-%m-%d')
print(f"Khoảng thời gian phân tích: Từ {start_date} đến {end_date}")

# 2. TẢI DỮ LIỆU S&P 500 (MÃ SPY) KHỚP VỚI THỜI GIAN TRÊN
print("Đang tải dữ liệu S&P 500 từ Yahoo Finance...")
df_spy = yf.download('SPY', start=start_date, end=end_date)

# Lấy cột giá đóng cửa của SPY
if isinstance(df_spy.columns, pd.MultiIndex): # Xử lý format mới của yfinance
    df_spy = df_spy['Close']
    df_spy.columns = ['SPY_Close']
else:
    df_spy = df_spy[['Close']]
    df_spy.columns = ['SPY_Close']

# Đảm bảo numeric
df_spy['SPY_Close'] = pd.to_numeric(df_spy['SPY_Close'], errors='coerce')

# 3. GHÉP 2 TẬP DỮ LIỆU LẠI VỚI NHAU
# Ghép (merge) theo đúng ngày giao dịch để tránh sai lệch do ngày lễ
df_merged = df_nvda[['Close']].rename(columns={'Close': 'NVDA_Close'}).join(df_spy, how='inner')

# Clean lại cho chắc
df_merged['NVDA_Close'] = pd.to_numeric(df_merged['NVDA_Close'], errors='coerce')
df_merged['SPY_Close'] = pd.to_numeric(df_merged['SPY_Close'], errors='coerce')

df_merged.dropna(inplace=True)

print("\nKiểu dữ liệu sau khi merge:")
print(df_merged.dtypes)

# 4. TÍNH LỢI SUẤT LOG VÀ HỆ SỐ BETA
# Tính Log Returns
df_merged['NVDA_Return'] = np.log(df_merged['NVDA_Close'] / df_merged['NVDA_Close'].shift(1))
df_merged['SPY_Return'] = np.log(df_merged['SPY_Close'] / df_merged['SPY_Close'].shift(1))
df_merged.dropna(inplace=True)

# Tính Hiệp phương sai (Covariance) và Phương sai (Variance)
covariance_matrix = df_merged[['NVDA_Return', 'SPY_Return']].cov()
cov_nvda_spy = covariance_matrix.loc['NVDA_Return', 'SPY_Return']
var_spy = df_merged['SPY_Return'].var()

# CÔNG THỨC BETA: Beta = Cov(NVDA, SPY) / Var(SPY)
beta_nvda = cov_nvda_spy / var_spy
print(f"\n=> HỆ SỐ BETA CỦA NVIDIA LÀ: {beta_nvda:.2f}")

# VẼ BIỂU ĐỒ SCATTER PLOT ĐỂ MINH HỌA TRỰC QUAN HỆ SỐ BETA

plt.figure(figsize=(10, 6))

# Vẽ biểu đồ phân tán (Scatter)
sns.regplot(x='SPY_Return', 
            y='NVDA_Return', 
            data=df_merged, 
            scatter_kws={'alpha':0.3, 'color': 'navy'}, 
            line_kws={'color':'red', 'linewidth': 2, 'label': f'Đường hồi quy (Beta = {beta_nvda:.2f})'})

plt.title('Hệ số Beta: Lợi suất NVIDIA so với S&P 500 (SPY)', fontsize=16, fontweight='bold')
plt.xlabel('Lợi suất Thị trường chung (S&P 500)', fontsize=12)
plt.ylabel('Lợi suất NVIDIA', fontsize=12)

# Thêm đường tham chiếu y=x (Beta = 1) để so sánh
limits = [max(df_merged['SPY_Return'].min(), df_merged['NVDA_Return'].min()), 
          min(df_merged['SPY_Return'].max(), df_merged['NVDA_Return'].max())]
plt.plot(limits, limits, 'k--', alpha=0.5, label='Biến động ngang thị trường (Beta = 1)')

plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig('nvda_beta_scatter.png', dpi=300)