import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình hiển thị
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ==========================================
# 1. TẢI DỮ LIỆU & LÀM SẠCH (DATA CLEANING)
# ==========================================
try:
    # Thay tên file bằng đúng tên file bạn tải lên
    file_path = "C:\\Users\\htc\\Desktop\\notebook\\Kì 4\\TIme series\\HistoricalData_1754061510662.csv"
    df = pd.read_csv(file_path)

    print("1. Initial column name:", df.columns.tolist())

    # --- BƯỚC SỬA LỖI QUAN TRỌNG ---
    # 1. Đổi tên cột 'Close/Last' thành 'Close' nếu có
    if 'Close/Last' in df.columns:
        df.rename(columns={'Close/Last': 'Close'}, inplace=True)
    
    # 2. Xóa khoảng trắng thừa trong tên cột (nếu có)
    df.columns = df.columns.str.strip()

    # 3. Xử lý ký tự '$' trong các cột giá và chuyển sang số (float)
    cols_to_fix = ['Close', 'Open', 'High', 'Low']
    for col in cols_to_fix:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('$', '', regex=False)
            df[col] = pd.to_numeric(df[col]) # Chuyển thành số
            print(f"   -> Handled character '$' in column {col}")

    # 4. Xử lý ngày tháng
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    
    print("✅ Data Cleaned")
    print(f"   Data from: {df.index.min().date()} đến {df.index.max().date()}")

    # ==========================================
    # 2. FEATURE ENGINEERING (TẠO BIẾN MỚI)
    # ==========================================
    # Tính Lợi suất Log (Log Returns)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Tính Lợi suất đơn (Simple Returns)
    df['Simple_Return'] = df['Close'].pct_change()
    
    # Loại bỏ dòng đầu tiên bị NaN
    df.dropna(inplace=True)

    # ==========================================
    # 3. THỐNG KÊ & TRỰC QUAN HÓA
    # ==========================================
    print("\n--- Descriptive Statistics ---")
    print(df[['Close', 'Log_Return']].describe().round(4))

    # Vẽ biểu đồ Giá đóng cửa
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='navy')
    plt.title('Close Price History of NVIDIA', fontsize=16)
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

    # Vẽ biểu đồ Phân phối Lợi suất Log
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Log_Return'], bins=100, kde=True, color='darkgreen')
    plt.title('Log Returns Distribution', fontsize=16)
    plt.xlabel('Log Return')
    plt.show()

except Exception as e:
    print(f"❌ Error: {e}")