import akshare as ak
import pandas as pd
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# 获取消费50指数成分股
def get_consumer_50_stocks():
    consumer_50_df = ak.index_stock_cons_csindex(symbol="000932")
    stock_list = consumer_50_df["成分券代码"].tolist()
    stock_names = dict(zip(consumer_50_df["成分券代码"], consumer_50_df["成分券名称"]))

    return stock_list, stock_names


# 获取单只股票的估值数据
def get_stock_valuation(stock_code, stock_name, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:

            df = ak.stock_value_em(symbol=stock_code)


            df["数据日期"] = pd.to_datetime(df["数据日期"])


            mask = (df["数据日期"] >= start_date) & (df["数据日期"] <= end_date)
            filtered_df = df.loc[mask].copy()

            if filtered_df.empty:
                print(f"警告: {stock_code}-{stock_name} 在指定时间段无数据")
                return pd.DataFrame()

            filtered_df["股票代码"] = stock_code
            filtered_df["股票名称"] = stock_name

            required_columns = ['数据日期', '股票代码', '股票名称', 'PEG值', '市净率', 'PE(静)', '市销率']
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

    stock_list, stock_names = get_consumer_50_stocks()
    print(f"获取到 {len(stock_list)} 只消费50成分股")

    all_data = pd.DataFrame()

    for stock_code in tqdm(stock_list, desc="下载股票数据"):
        stock_name = stock_names.get(stock_code, "未知")
        stock_data = get_stock_valuation(stock_code, stock_name, start_date, end_date)

        if not stock_data.empty:
            all_data = pd.concat([all_data, stock_data], ignore_index=True)
        else:
            print(f"跳过 {stock_code}-{stock_name}")

        time.sleep(2)


    if all_data.empty:
        print("未获取到任何数据，请检查参数设置")
    else:
        all_data.sort_values(by=["数据日期", "股票代码"], inplace=True)
        all_data.reset_index(drop=True, inplace=True)

        all_data.to_csv("消费50价值因子数据.csv", index=False, encoding="utf_8_sig")

        print("\n数据处理完成!")
        print(f"总数据量: {len(all_data)} 行")
        print(f"包含股票数量: {all_data['股票代码'].nunique()} 只")
        print(f"时间范围: {all_data['数据日期'].min().date()} 至 {all_data['数据日期'].max().date()}")
        print("\n数据预览:")
        print(all_data.head(10))

        success_stocks = set(all_data["股票代码"].unique())
        failed_stocks = [code for code in stock_list if code not in success_stocks]

        if failed_stocks:
            print("\n以下股票数据获取失败:")
            for code in failed_stocks:
                print(f"{code} - {stock_names.get(code, '未知')}")