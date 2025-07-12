#encoding:gbk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.dates as mdates
import calendar
from matplotlib.ticker import FuncFormatter


plt.rcParams['font.sans-serif'] = ['SimSun']  # ָ��Ĭ������ΪSimSun�����壩
plt.rcParams['axes.unicode_minus'] = False     # ���������ʾΪ���������
# 1. ����׼��
ic_series = pd.read_csv('IC.csv', header=None, names=['date', 'IC'])
ic_series['date'] = pd.to_datetime(ic_series['date'], format='%Y%m%d')
ic_series.set_index('date', inplace=True)

# 2. ICʱ�����з���
plt.figure(figsize=(14, 7))
plt.plot(ic_series.index, ic_series['IC'], color='steelblue', linewidth=1.5)
plt.axhline(y=ic_series['IC'].mean(), color='r', linestyle='--', label=f'��ֵ: {ic_series["IC"].mean():.4f}')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

# ��ӹ�����ֵ��120�촰�ڣ�
rolling_mean = ic_series['IC'].rolling(window=120).mean()
plt.plot(rolling_mean.index, rolling_mean, color='darkorange', linewidth=2, label='120�չ�����ֵ')

# ��ʽ���ͱ�ǩ
plt.title('ICֵʱ�����з���', fontsize=15, fontweight='bold')
plt.xlabel('����', fontsize=12)
plt.ylabel('ICֵ', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ICʱ������.png', dpi=300)
plt.show()

# 3. ��Ϣ���ʣ�IR������
IR = ic_series['IC'].mean() / ic_series['IC'].std()
print(f"��Ϣ����(IR): {IR:.4f}")

# 4. ICֱ��ͼ��ֲ�
plt.figure(figsize=(14, 7))
sns.histplot(ic_series['IC'], bins=30, kde=True, color='royalblue', alpha=0.7)

# ���ͳ����Ϣ
stats_text = f"��ֵ: {ic_series['IC'].mean():.4f}\n��׼��: {ic_series['IC'].std():.4f}\n��IC����: {len(ic_series[ic_series['IC']>0])/len(ic_series):.2%}\nIR: {IR:.4f}"
plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# �����̬�ֲ�����
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, ic_series['IC'].mean(), ic_series['IC'].std())
plt.plot(x, p, 'k', linewidth=1.5, label='��̬�ֲ�')

plt.title('ICֵ�ֲ�ֱ��ͼ', fontsize=15, fontweight='bold')
plt.xlabel('ICֵ', fontsize=12)
plt.ylabel('Ƶ��', fontsize=12)
plt.axvline(x=0, color='gray', linestyle='--')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('ICֱ��ͼ.png', dpi=300)
plt.show()

# 5. IC��QQͼ����̬�Լ��飩
plt.figure(figsize=(10, 8))
res = stats.probplot(ic_series['IC'], dist="norm", plot=plt)
plt.title('ICֵQQͼ����̬�Լ��飩', fontsize=15, fontweight='bold')
plt.xlabel('���۷�λ��', fontsize=12)
plt.ylabel('������λ��', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('IC_QQͼ.png', dpi=300)
plt.show()

# 6. �¶�IC����ͼ
# �������-�·ݾ���
ic_monthly = ic_series.copy()
ic_monthly['year'] = ic_monthly.index.year
ic_monthly['month'] = ic_monthly.index.month
monthly_ic = ic_monthly.groupby(['year', 'month'])['IC'].mean().reset_index()

# ��������͸�ӱ�
pivot_table = monthly_ic.pivot(index='year', columns='month', values='IC')

# ȷ�����������·ݣ�1-12�£�
for month in range(1, 13):
    if month not in pivot_table.columns:
        pivot_table[month] = np.nan
pivot_table = pivot_table.sort_index(axis=1)

# �����·ݱ�ǩ
month_names = ['1��', '2��', '3��', '4��', '5��', '6��', '7��', '8��', '9��', '10��', '11��', '12��']

plt.figure(figsize=(14, 10))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="coolwarm",
            center=0, linewidths=0.5, annot_kws={"size": 9},
            cbar_kws={'label': '�¶�ƽ��ICֵ'})

# ���������·ݱ�ǩ
plt.xticks(np.arange(12) + 0.5, month_names, rotation=0)
plt.yticks(rotation=0)
plt.title('�¶�ICֵ����ͼ', fontsize=15, fontweight='bold')
plt.xlabel('�·�', fontsize=12)
plt.ylabel('���', fontsize=12)
plt.tight_layout()
plt.savefig('�¶�IC����ͼ.png', dpi=300)
plt.show()

# 7. �¶�ICͳ�ƣ���״ͼ��
plt.figure(figsize=(14, 7))
monthly_avg = monthly_ic.groupby('month')['IC'].mean()
monthly_pos = monthly_ic.groupby('month')['IC'].apply(lambda x: (x > 0).mean())

ax1 = plt.gca()
ax2 = ax1.twinx()

# �¶�ƽ��IC
bars = ax1.bar(monthly_avg.index, monthly_avg.values, color='royalblue', alpha=0.7)
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(month_names)
ax1.set_ylabel('ƽ��ICֵ', fontsize=12)
ax1.set_ylim(min(monthly_avg)*1.2, max(monthly_avg)*1.2)

# �����״ͼ��ǩ
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9)

# ��IC����
ax2.plot(monthly_pos.index, monthly_pos.values * 100,
         marker='o', color='darkorange', linewidth=2, markersize=8)
ax2.set_ylabel('��IC���� (%)', fontsize=12)
ax2.set_ylim(0, 100)

# ������ݵ��ǩ
for i, value in enumerate(monthly_pos.values):
    ax2.text(i+1, value*100 + 2, f'{value*100:.1f}%',
             ha='center', va='bottom', fontsize=9, color='darkorange')

plt.title('�¶�IC���ַ���', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('�¶�IC����.png', dpi=300)
plt.show()

# 8. ���IC����
annual_ic = ic_monthly.groupby('year')['IC'].agg(['mean', 'std'])
annual_ic['IR'] = annual_ic['mean'] / annual_ic['std']
annual_ic['positive_ratio'] = ic_monthly.groupby('year')['IC'].apply(lambda x: (x > 0).mean())

print("\n���IC���ַ���:")
print(annual_ic)