import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

mv_data = pd.read_csv("消费50总市值数据.csv")

mv_data = mv_data.rename(columns={
    'trade_date': 'trade_date',
    '股票代码': 'stock_code',
    '股票名称': 'stock_name',
    'total_mv': 'total_mv'
})

# 估值因子
val_data = pd.read_csv(r"C:\Users\86133\Desktop\各种资料\python\量化\标准化后因子\processed_消费50估值数据_R_Z.csv",
                       header=None,
                       names=['trade_date', 'stock_code', 'PEG', 'PB', 'PE', 'PS'])

val_data['trade_date'] = pd.to_datetime(val_data['trade_date']).dt.strftime('%Y-%m-%d')

# 成长因子
growth_data = pd.read_csv(
    r"C:\Users\86133\Desktop\各种资料\python\量化\标准化后因子\processed_消费50成长因子（补齐)_R_Z.csv", header=None,
    names=['trade_date', 'stock_code', 'NetProfit_Growth', 'NetAsset_Growth',
           'Revenue_Growth', 'ROE', 'ROA'])

growth_data['trade_date'] = pd.to_datetime(growth_data['trade_date']).dt.strftime('%Y-%m-%d')

# 品质因子
quality_data = pd.read_csv(
    r"C:\Users\86133\Desktop\各种资料\python\量化\标准化后因子\processed_消费50品质因子数据（补齐)_R_Z.csv", header=None,
    names=['trade_date', 'stock_code', 'Inventory_Turnover',
           'TotalAsset_Turnover', 'CurrentAsset_Turnover', 'Capital_Fixed_Ratio'])

quality_data['trade_date'] = pd.to_datetime(quality_data['trade_date']).dt.strftime('%Y-%m-%d')

# 2. 合并总市值数据
def merge_mv(df):
    return pd.merge(df, mv_data[['trade_date', 'stock_code', 'total_mv']],
                    on=['trade_date', 'stock_code'], how='left')


val_data = merge_mv(val_data)
growth_data = merge_mv(growth_data)
quality_data = merge_mv(quality_data)

# 3. 市值对数转换
for df in [val_data, growth_data, quality_data]:
    df['total_mv'] = df['total_mv'].replace(0, np.nan)
    df['log_mv'] = np.log(df['total_mv'])


# 4. 使用LinearRegression进行因子中性化函数
def neutralize_factors_linear(df, factor_columns):
    """
    使用sklearn的LinearRegression进行因子中性化
    步骤：
    1. 对每个交易日分组
    2. 对每个因子，建立回归模型：因子值 = a * 市值 + b
    3. 计算残差：中性化因子值 = 原始因子值 - 预测因子值
    """
    result = df.copy()

    # 按交易日分组处理
    grouped = df.groupby('trade_date')

    for factor in factor_columns:
        print(f"开始中性化处理因子: {factor}")
        neutral_values = []  # 存储中性化后的值

        # 对每个交易日进行处理
        for date, group in grouped:
            # 获取当前交易日的因子值和市值
            X = group['log_mv'].values.reshape(-1, 1)  # 自变量：市值（需要二维数组）
            y = group[factor].values  # 因变量：因子值

            # 创建有效数据索引（同时非缺失）
            valid_idx = (~pd.isna(X.flatten())) & (~pd.isna(y))

            if np.sum(valid_idx) < 5:  # 确保有足够样本
                neutral_values.extend([np.nan] * len(group))
                continue

            # 提取有效数据
            X_valid = X[valid_idx]
            y_valid = y[valid_idx]

            # 创建线性回归模型
            model = LinearRegression()
            model.fit(X_valid, y_valid)

            # 计算预测值
            y_pred = model.predict(X_valid)

            # 计算残差（中性化值）
            residuals = y_valid - y_pred

            # 创建全长度数组（包含缺失值）
            date_neutral = np.full(len(group), np.nan)
            date_neutral[valid_idx] = residuals

            neutral_values.extend(date_neutral)

        # 添加中性化后的因子列
        result[f'{factor}_neutral'] = neutral_values

    return result


# 5. 执行中性化处理
print("开始估值因子中性化...")
val_neutral = neutralize_factors_linear(val_data, ['PEG', 'PB', 'PE', 'PS'])

print("开始成长因子中性化...")
growth_neutral = neutralize_factors_linear(growth_data, ['NetProfit_Growth', 'NetAsset_Growth',
                                                         'Revenue_Growth', 'ROE', 'ROA'])

print("开始品质因子中性化...")
quality_neutral = neutralize_factors_linear(quality_data, ['Inventory_Turnover', 'TotalAsset_Turnover',
                                                           'CurrentAsset_Turnover', 'Capital_Fixed_Ratio'])

# 6. 保存结果
val_neutral.to_csv("中性化_消费50估值因子.csv", index=False)
growth_neutral.to_csv("中性化_消费50成长因子.csv", index=False)
quality_neutral.to_csv("中性化_消费50品质因子.csv", index=False)

print("中性化处理完成！")


# 7. 验证中性化效果
def check_neutralization(df, factor):
    """验证中性化效果：检查中性化后因子与市值的相关性"""
    # 计算相关系数
    corr_values = []

    # 按日期分组计算相关系数
    for date, group in df.groupby('trade_date'):
        valid_idx = (~group['log_mv'].isna()) & (~group[f'{factor}_neutral'].isna())
        if valid_idx.sum() > 5:
            corr = np.corrcoef(group.loc[valid_idx, 'log_mv'],
                               group.loc[valid_idx, f'{factor}_neutral'])[0, 1]
            corr_values.append(corr)

    avg_corr = np.nanmean(corr_values) if corr_values else np.nan
    print(f"因子 {factor} 中性化后与市值的平均相关系数: {avg_corr:.4f}")
    return avg_corr


print("\n中性化效果验证:")
for factor in ['PEG', 'PB', 'PE', 'PS']:
    check_neutralization(val_neutral, factor)

for factor in ['NetProfit_Growth', 'NetAsset_Growth', 'Revenue_Growth', 'ROE', 'ROA']:
    check_neutralization(growth_neutral, factor)

for factor in ['Inventory_Turnover', 'TotalAsset_Turnover', 'CurrentAsset_Turnover', 'Capital_Fixed_Ratio']:
    check_neutralization(quality_neutral, factor)