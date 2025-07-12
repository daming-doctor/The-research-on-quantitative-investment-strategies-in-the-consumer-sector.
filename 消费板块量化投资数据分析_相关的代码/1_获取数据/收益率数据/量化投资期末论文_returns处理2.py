import pandas as pd
from io import StringIO
import numpy as np
import os

# 给定的股票代码列表（确保均为6位字符串）
target_stocks = {
    '000998', '002299', '002304', '002311', '002385', '002507', '002557', '002568', '002714',
    '300146', '300498', '300765', '300957', '300999', '600132', '600298', '600299', '600519',
    '600598', '600600', '600737', '600809', '600872', '600873', '600887', '601118', '603156',
    '603288', '603345', '603369', '603589', '603605', '605499', '688363', '000568', '000596',
    '000729', '000858', '000876', '000895'
}

# 实际应用中请替换为真实文件路径
file_path_sh = 'returns(SH)预处理后数据.csv'
file_path_sz = 'returns(SZ)预处理后数据.csv'
output_path = 'returns_合并筛选后数据.csv'


# 定义通用数据读取函数
def read_and_filter(file_path, target_codes):
    """读取文件并筛选目标股票代码"""
    try:
        # 读取文件并自动处理分隔符（支持CSV/TSV）
        df = pd.read_csv(file_path, sep=None, engine='python')

        # 检查核心列是否存在
        required_cols = {'trade_date', 'stock_code', 'returns'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"文件缺少必要列: {', '.join(missing_cols)}")

        # 统一股票代码为6位字符串（去除可能的市场后缀，如.SH/.SZ）
        df['stock_code'] = df['stock_code'].astype(str).str.replace(r'\.[A-Z]+$', '', regex=True).str.zfill(6)

        # 筛选目标股票
        filtered_df = df[df['stock_code'].isin(target_codes)].copy()

        # 转换日期格式（处理无效日期）
        filtered_df['trade_date'] = pd.to_datetime(filtered_df['trade_date'], format='%Y%m%d', errors='coerce')

        # 检查日期转换是否成功
        invalid_dates = filtered_df['trade_date'].isna().sum()
        if invalid_dates > 0:
            print(f"警告: {invalid_dates} 行日期格式无法解析，已设置为NaT")

        # 移除无效日期的行
        valid_df = filtered_df.dropna(subset=['trade_date']).copy()

        return valid_df

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return pd.DataFrame()  # 返回空DataFrame


# 读取并筛选数据
print("正在读取深市数据...")
df_sz = read_and_filter(file_path_sz, target_stocks)
print(f"深市数据筛选后: {len(df_sz)} 行")

print("\n正在读取沪市数据...")
df_sh = read_and_filter(file_path_sh, target_stocks)
print(f"沪市数据筛选后: {len(df_sh)} 行")

# 合并数据
print("\n正在合并数据...")
merged_df = pd.concat([df_sz, df_sh], ignore_index=True)

# 按日期和股票代码排序
merged_df.sort_values(['trade_date', 'stock_code'], inplace=True)
merged_df.reset_index(drop=True, inplace=True)

# 数据验证
print("\n数据验证:")
print(f"总记录数: {len(merged_df)}")
print(f"时间范围: {merged_df['trade_date'].min().date()} 至 {merged_df['trade_date'].max().date()}")
print(f"包含股票数量: {merged_df['stock_code'].nunique()}")
print(f"缺失收益率: {merged_df['returns'].isna().sum()} 行")

# 保存结果
print(f"\n正在保存结果到 {output_path}...")
merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')

# 输出数据示例
print("\n数据前几行示例:")
print(merged_df.head().to_string())