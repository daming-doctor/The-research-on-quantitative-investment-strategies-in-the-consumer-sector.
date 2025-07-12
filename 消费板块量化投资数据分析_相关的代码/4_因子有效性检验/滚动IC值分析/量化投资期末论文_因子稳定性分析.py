import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import numpy as np

# 计算滚动24个月IC均值的函数
def calculate_rolling_ic(ic_series, window=504):
    """计算滚动24个月IC均值"""
    return ic_series.rolling(window=window, min_periods=window//2).mean()

# 计算时间序列的线性趋势强度的函数
def calculate_trend_strength(series):
    """计算时间序列的线性趋势强度"""
    n = len(series.dropna())
    x = np.arange(n)
    y = series.dropna().values
    x2 = x * x
    xy = x * y
    slope = (n * np.sum(xy) - np.sum(x) * np.sum(y)) / (n * np.sum(x2) - np.sum(x) ** 2)
    return slope * 1000  # 千分比趋势斜率

# 绘制滚动IC稳定性分析图的函数
def plot_rolling_ic_analysis(ic_series):
    """绘制滚动IC稳定性分析图"""
    # 缩小图片整体尺寸
    plt.figure(figsize=(8, 6))

    # 1. 滚动IC均值
    rolling_ic = calculate_rolling_ic(ic_series)
    ax1 = plt.subplot(2, 1, 1)
    rolling_ic.plot(color='dodgerblue', lw=2, label='24个月滚动IC均值')

    # 标记统计特征
    max_ic = rolling_ic.idxmax()
    min_ic = rolling_ic.idxmin()
    plt.scatter(max_ic, rolling_ic.max(), s=50, c='red', label=f'峰值: {rolling_ic.max():.4f}')
    plt.scatter(min_ic, rolling_ic.min(), s=50, c='green', label=f'谷值: {rolling_ic.min():.4f}')

    # 2. 趋势指标
    trend = rolling_ic.rolling(252).apply(calculate_trend_strength)
    ax2 = plt.subplot(2, 1, 2)
    trend.plot(color='purple', lw=2, label='12个月IC趋势斜率')

    # 3. 格式设置
    # 进一步调整标题字体大小
    ax1.set_title('滚动24个月IC均值分析', fontsize=10, weight='bold')
    # 调整坐标轴标签字体大小
    ax1.set_xlabel('', fontsize=8)
    ax1.set_ylabel('IC均值', fontsize=8)
    # 调整图例字体大小
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(0, color='gray', ls='--')
    # 进一步调整标题字体大小
    ax2.set_title('IC趋势强度分析', fontsize=10, weight='bold')
    # 调整坐标轴标签字体大小
    ax2.set_xlabel('日期', fontsize=8)
    ax2.set_ylabel('趋势斜率(‰)', fontsize=8)
    # 调整图例字体大小
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 读取 CSV 文件
df = pd.read_csv('IC.csv')

df.set_index(df.columns[0], inplace=True)
df.index = pd.to_datetime(df.index, format='%Y%m%d')

selected_data = df.loc['20230606':'20250606']

plt.rcParams['figure.dpi'] = 300

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 计算 pe_IC 值的统计量
mean_value = selected_data.iloc[:, 0].mean()
std_value = selected_data.iloc[:, 0].std()
skew_value = skew(selected_data.iloc[:, 0])
kurtosis_value = kurtosis(selected_data.iloc[:, 0])

print('20230606 到 20250606 期间 pe_IC 值的统计信息：')
print(f'均值：{mean_value:.4f}')
print(f'标准差：{std_value:.4f}')
print(f'偏度：{skew_value:.4f}')
print(f'峰度：{kurtosis_value:.4f}')

# 进行滚动IC分析并绘图
ic_series = selected_data.iloc[:, 0]
plot_rolling_ic_analysis(ic_series)

# 稳定性指标评估
rolling_ic = calculate_rolling_ic(ic_series)
volatility_range = rolling_ic.max() - rolling_ic.min()
start_ic = rolling_ic.iloc[0] if len(rolling_ic) > 0 else 0
end_ic = rolling_ic.iloc[-1] if len(rolling_ic) > 0 else 0
decay = start_ic - end_ic
max_rolling_ic = rolling_ic.max()
trough_ic = rolling_ic.min()
drawdown = (max_rolling_ic - trough_ic) / max_rolling_ic * 100 if max_rolling_ic != 0 else 0

# 打印稳定性指标结果
print('\n稳定性指标评估：')
print(f'波动范围: {volatility_range:.4f}')
print(f'衰减幅度: {decay:.4f}')
print(f'回撤深度: {drawdown:.2f}%')

# 稳定性评级
if volatility_range < 0.03 and drawdown < 20:
    rating = 'A+(卓越)'
elif 0.03 <= volatility_range < 0.05 and 20 <= drawdown < 30:
    rating = 'A(优秀)'
elif 0.05 <= volatility_range < 0.08 and 30 <= drawdown < 40:
    rating = 'B(良好)'
elif 0.08 <= volatility_range < 0.12 and 40 <= drawdown < 50:
    rating = 'C(一般)'
else:
    rating = 'D(淘汰)'

print(f'\n稳定性评级: {rating}')