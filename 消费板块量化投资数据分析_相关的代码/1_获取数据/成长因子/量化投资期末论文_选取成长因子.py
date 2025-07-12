import akshare as ak
import pandas as pd
import time
from tqdm import tqdm
import warnings
import numpy as np

warnings.filterwarnings('ignore')


# 获取消费50指数成分股
def get_consumer_50_stocks():
    consumer_50_df = ak.index_stock_cons_csindex(symbol="000932")
    stock_list = consumer_50_df["成分券代码"].tolist()

    stock_names = dict(zip(consumer_50_df["成分券代码"], consumer_50_df["成分券名称"]))

    return stock_list, stock_names


def get_financial_indicators(stock_code, stock_name, start_date, end_date):
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            df = ak.stock_financial_analysis_indicator(symbol=stock_code, start_year="2020")

            if df is None or df.empty:
                print(f"警告: {stock_code}-{stock_name} 返回空数据 (尝试 {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                continue

            df["日期"] = pd.to_datetime(df["日期"])

            mask = (df["日期"] >= start_date) & (df["日期"] <= end_date)
            filtered_df = df.loc[mask].copy()

            if filtered_df.empty:
                print(f"警告: {stock_code}-{stock_name} 在指定时间段无数据")
                return pd.DataFrame()

            filtered_df["股票代码"] = stock_code
            filtered_df["股票名称"] = stock_name

            required_columns = ['日期', '股票代码', '股票名称', '净利润增长率(%)', '净资产增长率(%)', '主营业务收入增长率(%)', '净资产报酬率(%)','资产报酬率(%)']

            available_columns = [col for col in required_columns if col in filtered_df.columns]

            result_df = filtered_df[available_columns]

            for col in set(required_columns) - set(available_columns):
                result_df[col] = float('nan')
                print(f"警告: {stock_code}-{stock_name} 缺少 {col} 列")

            return result_df

        except Exception as e:
            print(f"获取 {stock_code}-{stock_name} 数据失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(2)

        print(f"错误: {stock_code}-{stock_name} 数据获取失败，已达最大重试次数")
        return pd.DataFrame()


# 主程序
if __name__ == "__main__":
    start_date = "2023-06-06"
    end_date = "2025-06-06"

    # 获取消费50成分股列表和名称映射
    stock_list, stock_names = get_consumer_50_stocks()
    print(f"获取到 {len(stock_list)} 只消费50成分股")

    # 创建空DataFrame存储结果
    all_financial_data = pd.DataFrame()

    # 遍历股票列表获取数据
    for stock_code in tqdm(stock_list, desc="下载财务数据"):
        stock_name = stock_names.get(stock_code, "未知")
        financial_data = get_financial_indicators(stock_code, stock_name, start_date, end_date)

        if not financial_data.empty:
            all_financial_data = pd.concat([all_financial_data, financial_data], ignore_index=True)
        else:
            print(f"跳过 {stock_code}-{stock_name}")

        # 添加延迟避免请求过快
        time.sleep(2)

    # 检查是否有数据
    if all_financial_data.empty:
        print("未获取到任何财务数据，请检查参数设置")
    else:
        # 按报告日期和股票代码排序
        all_financial_data.sort_values(by=["日期", "股票代码"], inplace=True)

        all_financial_data.reset_index(drop=True, inplace=True)

        # 保存结果到CSV文件
        all_financial_data.to_csv("消费50成长因子数据.csv", index=False, encoding="utf_8_sig")

        print("\n数据处理完成!")
        print(f"总数据量: {len(all_financial_data)} 行")
        print(f"包含股票数量: {all_financial_data['股票代码'].nunique()} 只")
        print(
            f"时间范围: {all_financial_data['日期'].min().date()} 至 {all_financial_data['日期'].max().date()}")
        print("\n数据预览:")
        print(all_financial_data.head(10))

        # 列出失败的股票
        success_stocks = set(all_financial_data["股票代码"].unique())
        failed_stocks = [code for code in stock_list if code not in success_stocks]

        if failed_stocks:
            print("\n以下股票财务数据获取失败:")
            for code in failed_stocks:
                print(f"{code} - {stock_names.get(code, '未知')}")

        # 分析数据覆盖情况
        date_counts = all_financial_data.groupby("日期").size()
        print("\n各报告期数据点数量:")
        print(date_counts)