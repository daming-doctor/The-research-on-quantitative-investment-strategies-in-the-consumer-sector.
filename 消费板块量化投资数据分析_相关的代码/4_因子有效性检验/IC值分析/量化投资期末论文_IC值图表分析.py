#encoding:gbk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.dates as mdates
import calendar
from matplotlib.ticker import FuncFormatter


plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体为SimSun（宋体）
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示为方块的问题
# 1. 数据准备
ic_series = pd.read_csv('IC.csv', header=None, names=['date', 'IC'])
ic_series['date'] = pd.to_datetime(ic_series['date'], format='%Y%m%d')
ic_series.set_index('date', inplace=True)

# 2. IC时间序列分析
plt.figure(figsize=(14, 7))
plt.plot(ic_series.index, ic_series['IC'], color='steelblue', linewidth=1.5)
plt.axhline(y=ic_series['IC'].mean(), color='r', linestyle='--', label=f'均值: {ic_series["IC"].mean():.4f}')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

# 添加滚动均值（120天窗口）
rolling_mean = ic_series['IC'].rolling(window=120).mean()
plt.plot(rolling_mean.index, rolling_mean, color='darkorange', linewidth=2, label='120日滚动均值')

# 格式化和标签
plt.title('IC值时间序列分析', fontsize=15, fontweight='bold')
plt.xlabel('日期', fontsize=12)
plt.ylabel('IC值', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('IC时间序列.png', dpi=300)
plt.show()

# 3. 信息比率（IR）分析
IR = ic_series['IC'].mean() / ic_series['IC'].std()
print(f"信息比率(IR): {IR:.4f}")

# 4. IC直方图与分布
plt.figure(figsize=(14, 7))
sns.histplot(ic_series['IC'], bins=30, kde=True, color='royalblue', alpha=0.7)

# 添加统计信息
stats_text = f"均值: {ic_series['IC'].mean():.4f}\n标准差: {ic_series['IC'].std():.4f}\n正IC比例: {len(ic_series[ic_series['IC']>0])/len(ic_series):.2%}\nIR: {IR:.4f}"
plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# 添加正态分布曲线
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, ic_series['IC'].mean(), ic_series['IC'].std())
plt.plot(x, p, 'k', linewidth=1.5, label='正态分布')

plt.title('IC值分布直方图', fontsize=15, fontweight='bold')
plt.xlabel('IC值', fontsize=12)
plt.ylabel('频率', fontsize=12)
plt.axvline(x=0, color='gray', linestyle='--')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('IC直方图.png', dpi=300)
plt.show()

# 5. IC的QQ图（正态性检验）
plt.figure(figsize=(10, 8))
res = stats.probplot(ic_series['IC'], dist="norm", plot=plt)
plt.title('IC值QQ图（正态性检验）', fontsize=15, fontweight='bold')
plt.xlabel('理论分位数', fontsize=12)
plt.ylabel('样本分位数', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('IC_QQ图.png', dpi=300)
plt.show()

# 6. 月度IC热力图
# 创建年份-月份矩阵
ic_monthly = ic_series.copy()
ic_monthly['year'] = ic_monthly.index.year
ic_monthly['month'] = ic_monthly.index.month
monthly_ic = ic_monthly.groupby(['year', 'month'])['IC'].mean().reset_index()

# 创建数据透视表
pivot_table = monthly_ic.pivot(index='year', columns='month', values='IC')

# 确保包含所有月份（1-12月）
for month in range(1, 13):
    if month not in pivot_table.columns:
        pivot_table[month] = np.nan
pivot_table = pivot_table.sort_index(axis=1)

# 中文月份标签
month_names = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']

plt.figure(figsize=(14, 10))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="coolwarm",
            center=0, linewidths=0.5, annot_kws={"size": 9},
            cbar_kws={'label': '月度平均IC值'})

# 设置中文月份标签
plt.xticks(np.arange(12) + 0.5, month_names, rotation=0)
plt.yticks(rotation=0)
plt.title('月度IC值热力图', fontsize=15, fontweight='bold')
plt.xlabel('月份', fontsize=12)
plt.ylabel('年份', fontsize=12)
plt.tight_layout()
plt.savefig('月度IC热力图.png', dpi=300)
plt.show()

# 7. 月度IC统计（柱状图）
plt.figure(figsize=(14, 7))
monthly_avg = monthly_ic.groupby('month')['IC'].mean()
monthly_pos = monthly_ic.groupby('month')['IC'].apply(lambda x: (x > 0).mean())

ax1 = plt.gca()
ax2 = ax1.twinx()

# 月度平均IC
bars = ax1.bar(monthly_avg.index, monthly_avg.values, color='royalblue', alpha=0.7)
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(month_names)
ax1.set_ylabel('平均IC值', fontsize=12)
ax1.set_ylim(min(monthly_avg)*1.2, max(monthly_avg)*1.2)

# 添加柱状图标签
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9)

# 正IC比例
ax2.plot(monthly_pos.index, monthly_pos.values * 100,
         marker='o', color='darkorange', linewidth=2, markersize=8)
ax2.set_ylabel('正IC比例 (%)', fontsize=12)
ax2.set_ylim(0, 100)

# 添加数据点标签
for i, value in enumerate(monthly_pos.values):
    ax2.text(i+1, value*100 + 2, f'{value*100:.1f}%',
             ha='center', va='bottom', fontsize=9, color='darkorange')

plt.title('月度IC表现分析', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('月度IC分析.png', dpi=300)
plt.show()

# 8. 年度IC表现
annual_ic = ic_monthly.groupby('year')['IC'].agg(['mean', 'std'])
annual_ic['IR'] = annual_ic['mean'] / annual_ic['std']
annual_ic['positive_ratio'] = ic_monthly.groupby('year')['IC'].apply(lambda x: (x > 0).mean())

print("\n年度IC表现分析:")
print(annual_ic)