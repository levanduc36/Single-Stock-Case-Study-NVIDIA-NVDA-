import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Cấu hình hiển thị
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# 1. TẢI DỮ LIỆU & LÀM SẠCH (DATA CLEANING)
try:
    # Thay tên file bằng đúng tên file bạn tải lên
    file_path = "HistoricalData_1754061510662.csv"
    df = pd.read_csv(file_path)

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
            df[col] = pd.to_numeric(df[col])

    # 4. Xử lý ngày tháng
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    
    print(" Làm sạch dữ liệu thành công!")
    print(f"   Dữ liệu từ: {df.index.min().date()} đến {df.index.max().date()}")

    # 2. FEATURE ENGINEERING (TẠO BIẾN MỚI)
    # Tính Lợi suất Log (Log Returns)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Loại bỏ NaN values
    df.dropna(inplace=True)
    
    # Bình phương lợi suất (Squared Returns) đại diện cho độ lớn rủi ro
    df['Squared_Return'] = df['Log_Return'] ** 2
    
    print(" Feature engineering hoàn tất!")

except Exception as e:
    print(f" Lỗi: {e}")

# BIỂU ĐỒ 1: BIẾN ĐỘNG THỰC TẾ HÀNG NĂM (ANNUALIZED REALIZED VOLATILITY)
print("\n Vẽ biểu đồ 1: Biến động thực tế hàng năm...")

window = 21
trading_days = 252
df['Realized_Vol'] = df['Log_Return'].rolling(window=window).std() * np.sqrt(trading_days)

plt.figure(figsize=(14, 5))
plt.plot(df.index, df['Realized_Vol'], color='navy', linewidth=1.5, label='Biến động thực tế (21 ngày)')
plt.title('1. Biến động thực tế hàng năm (Annualized Realized Volatility)', fontsize=16, fontweight='bold')
plt.ylabel('Độ biến động (Volatility)')
plt.xlabel('Năm')
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig('1_realized_volatility.png', dpi=300)

# BIỂU ĐỒ 2: CÁC CHẾ ĐỘ BIẾN ĐỘNG (VOLATILITY REGIMES)
print(" Vẽ biểu đồ 2: Phân vùng chế độ biến động...")

# Xác định ngưỡng Tứ phân vị
vol_q25 = df['Realized_Vol'].quantile(0.25)
vol_q75 = df['Realized_Vol'].quantile(0.75)

plt.figure(figsize=(14, 5))
plt.plot(df.index, df['Realized_Vol'], color='black', linewidth=1.2, label='Đường Biến Động')

# Tô màu vùng Rủi ro cao (>Q3)
plt.fill_between(df.index, 0, df['Realized_Vol'].max() * 1.05, 
                 where=df['Realized_Vol'] > vol_q75, 
                 color='red', alpha=0.2, label=f'Chế độ Biến động Cao (> {vol_q75:.1%})')

# Tô màu vùng An toàn (<Q1)
plt.fill_between(df.index, 0, df['Realized_Vol'].max() * 1.05, 
                 where=df['Realized_Vol'] < vol_q25, 
                 color='green', alpha=0.2, label=f'Chế độ Biến động Thấp (< {vol_q25:.1%})')

plt.title('2. Phân vùng Chế độ Biến động (Volatility Regimes)', fontsize=16, fontweight='bold')
plt.ylabel('Độ biến động')
plt.xlabel('Năm')
plt.ylim(0, df['Realized_Vol'].max() * 1.05)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
# plt.savefig('2_volatility_regimes.png', dpi=300)

# BIỂU ĐỒ 3: BÌNH PHƯƠNG LỢI SUẤT (SQUARED RETURNS - VOLATILITY CLUSTERING)
print(" Vẽ biểu đồ 3: Bình phương lợi suất...")

plt.figure(figsize=(14, 5))
plt.plot(df.index, df['Squared_Return'], color='purple', alpha=0.8, linewidth=1.2)
plt.title('3. Bình phương Lợi suất (Squared Returns) - Tụ cụm biến động', fontsize=16, fontweight='bold')
plt.ylabel('Squared Log Returns')
plt.xlabel('Năm')

# Đánh dấu các sự kiện tụ cụm lớn
plt.axvspan(pd.to_datetime('2020-02-01'), pd.to_datetime('2020-05-01'), color='red', alpha=0.15, label='Covid-19 Panic')
plt.axvspan(pd.to_datetime('2022-01-01'), pd.to_datetime('2022-12-31'), color='orange', alpha=0.15, label='Tech Sell-off')

plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig('3_squared_returns.png', dpi=300)

# BIỂU ĐỒ 4: ACF CỦA BÌNH PHƯƠNG LỢI SUẤT (ARCH EFFECT DETECTION)
print(" Vẽ biểu đồ 4: ACF của bình phương lợi suất...")

fig, ax = plt.subplots(figsize=(12, 5))
_ = plot_acf(df['Squared_Return'], ax=ax, lags=40, color='darkred', alpha=0.05)

ax.set_title('4. ACF của Bình phương Lợi suất (ARCH Effect Detection)', fontsize=14, fontweight='bold')
ax.set_xlabel('Độ trễ (Lags)', fontsize=12)
ax.set_ylabel('Autocorrelation', fontsize=12)

plt.tight_layout()
plt.show()
# plt.savefig('4_acf_squared_returns.png', dpi=300)

# BIỂU ĐỒ 5: ACF & PACF CỦA BÌNH PHƯƠNG LỢI SUẤT (KIỂM ĐỊNH HIỆU ỨNG ARCH)
print(" Vẽ biểu đồ 5: ACF & PACF của bình phương lợi suất...")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Đồ thị ACF (Autocorrelation Function)
plot_acf(df['Squared_Return'], ax=axes[0], lags=40, color='navy')
axes[0].set_title('ACF - Kiểm định Hiệu ứng ARCH (Tự tương quan của Rủi ro)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Mức độ tương quan')
axes[0].set_xlabel('Độ trễ (Lags - Số ngày)')

# Đồ thị PACF (Partial Autocorrelation Function)
plot_pacf(df['Squared_Return'], ax=axes[1], lags=40, color='darkred', method='ywm')
axes[1].set_title('PACF - Tự tương quan Riêng phần của Rủi ro', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Mức độ tương quan')
axes[1].set_xlabel('Độ trễ (Lags - Số ngày)')

plt.tight_layout()
plt.show()
# plt.savefig('5_arch_effect_acf_pacf.png', dpi=300)

print("\n Hoàn tất phân tích biến động!")

# 3. TRANG TRÍ TIÊU ĐỀ CHO ĐẸP ĐỂ ĐƯA LÊN SLIDE
ax.set_title('ACF của Bình phương Lợi suất (Squared Returns)', fontsize=14, fontweight='bold')
ax.set_xlabel('Độ trễ (Lags)', fontsize=12)
ax.set_ylabel('Autocorrelation', fontsize=12)

# 4. LỆNH DỌN DẸP CUỐI CÙNG
# Lệnh này tự động cắt gọt các phần chữ rườm rà xung quanh để biểu đồ không bị lẹm viền
plt.tight_layout()

# 5. HIỂN THỊ
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# BIỂU ĐỒ 2: KIỂM ĐỊNH HIỆU ỨNG ARCH (ACF & PACF CỦA SQUARED RETURNS)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Đồ thị ACF (Autocorrelation Function)
plot_acf(df['Squared_Return'], ax=axes[0], lags=40, color='navy')
axes[0].set_title('ACF - Kiểm định Hiệu ứng ARCH (Tự tương quan của Rủi ro)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Mức độ tương quan')
axes[0].set_xlabel('Độ trễ (Lags - Số ngày)')

# Đồ thị PACF (Partial Autocorrelation Function)
plot_pacf(df['Squared_Return'], ax=axes[1], lags=40, color='darkred', method='ywm')
axes[1].set_title('PACF - Tự tương quan Riêng phần của Rủi ro', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Mức độ tương quan')
axes[1].set_xlabel('Độ trễ (Lags - Số ngày)')

plt.tight_layout()
plt.show()
# Lưu biểu đồ (nếu cần): plt.savefig('arch_effect_acf_pacf_slide.png', dpi=300)