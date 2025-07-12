import akshare as ak
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# 获取中证消费50指数成分股
consumer_50 = ak.index_stock_cons_csindex(symbol="000932")
consumer_stocks = consumer_50["成分券代码"].tolist()
stock_names = dict(zip(consumer_50["成分券代码"], consumer_50["成分券名称"]))

# 定义时间范围
start_date = "20230606"
end_date = "20250606"

# ---------------------------- 获取成分股日线数据 ----------------------------
def get_stock_data(stock_code):
    """获取单只股票的历史日线数据（简化列）"""
    try:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        return df[["日期", "收盘", "成交量", "涨跌幅"]].rename(columns={
            "日期": "trade_date", "收盘": "close", "成交量": "volume", "涨跌幅": "pct_change"
        })
    except Exception as e:
        print(f"获取 {stock_code} 日线数据失败: {str(e)}")
        return pd.DataFrame()

print("开始获取成分股日线数据...")
all_stock_data = {}
for code in tqdm(consumer_stocks, desc="下载日线数据"):
    df = get_stock_data(code)
    if not df.empty:
        df["stock_code"] = code
        df["stock_name"] = stock_names.get(code, "未知")
        all_stock_data[code] = df
    time.sleep(0.5)  # 避免请求过快

# 合并为指数层面的数据（
index_data = pd.DataFrame()
for code, df in all_stock_data.items():
    index_data = pd.concat([index_data, df], ignore_index=True)

# 按日期聚合并重置索引
index_data = index_data.groupby("trade_date").agg({
    "close": "mean",    # 等权平均收盘价（指数点位）
    "volume": "sum",    # 总成交量（反映市场热度）
    "pct_change": "mean"# 等权平均涨跌幅（指数收益率）
}).rename(columns={
    "close": "index_close",
    "pct_change": "index_pct_change",
    "volume": "index_volume"
}).reset_index()  # 将trade_date从索引转为普通列

# ---------------------------- 计算核心技术因子 ----------------------------
print("计算核心技术因子...")

# 1. 趋势类因子
index_data["MOM_20"] = index_data["index_close"].pct_change(periods=20)  # 20日动量
index_data["MOM_60"] = index_data["index_close"].pct_change(periods=60)  # 60日动量
index_data["MA_20"] = index_data["index_close"].rolling(20).mean()       # 20日移动平均线

# 2. 波动率类因子
index_data["VOL_20"] = index_data["index_pct_change"].rolling(20).std()  # 20日波动率
index_data["VOL_60"] = index_data["index_pct_change"].rolling(60).std()  # 60日波动率

# 3. 成交量类因子
index_data["VMA_5"] = index_data["index_volume"].rolling(5).mean()       # 5日成交量均线
index_data["VOLUME_RATIO"] = index_data["index_volume"] / index_data["VMA_5"]  # 量比

# 4. 估值类因子（需先获取PE数据）
def get_pe_data(stock_code):
    """获取单只股票的PE数据（简化列）"""
    try:
        df = ak.stock_a_indicator_lg(symbol=stock_code)
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        # 筛选目标日期范围（2023-06-06 至 2025-06-06）
        start_date = pd.to_datetime("2023-06-06")
        end_date = pd.to_datetime("2025-06-06")
        df_filtered = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
        pe_col = "pe"
        if pe_col:
            return df_filtered[["trade_date", pe_col]].rename(columns={pe_col: "pe_ratio"})
        return pd.DataFrame()
    except Exception as e:
        print(f"获取 {stock_code} PE数据失败: {str(e)}")
        return pd.DataFrame()

print("开始获取成分股PE数据...")
all_pe_data = pd.DataFrame()
for code in tqdm(consumer_stocks, desc="下载PE数据"):
    pe_df = get_pe_data(code)
    if not pe_df.empty:
        pe_df["stock_code"] = code
        all_pe_data = pd.concat([all_pe_data, pe_df], ignore_index=True)
    time.sleep(0.5)

# 处理PE数据并计算指数PE及变化率
if not all_pe_data.empty:
    all_pe_data["trade_date"] = pd.to_datetime(all_pe_data["trade_date"])
    # 按日期和股票代码取最新PE（避免重复）
    all_pe_data = all_pe_data.sort_values(["trade_date", "stock_code"]).drop_duplicates(["trade_date", "stock_code"], keep="last")
    # 计算指数PE（等权平均）
    index_pe = all_pe_data.groupby("trade_date")["pe_ratio"].mean().reset_index()
    index_pe = index_pe.rename(columns={"trade_date": "trade_date", "pe_ratio": "index_pe"})
    index_pe['trade_date'] = pd.to_datetime(index_pe['trade_date'])

    # 将索引转换为列进行合并，然后再设回索引
    index_data_temp = index_data.reset_index()
    index_data_temp['trade_date'] = pd.to_datetime(index_data_temp['trade_date'])
    index_data = pd.merge(index_data_temp, index_pe, on="trade_date", how="left").set_index('trade_date')

    # 计算PE变化率（20日和60日）
    index_data["PE_CHANGE_20"] = index_data["index_pe"].pct_change(20)
    index_data["PE_CHANGE_60"] = index_data["index_pe"].pct_change(60)
else:
    print("未获取到PE数据，跳过PE相关因子")
    index_data["index_pe"] = np.nan
    index_data["PE_CHANGE_20"] = np.nan
    index_data["PE_CHANGE_60"] = np.nan

# 5. 超买超卖类因子（RSI）
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

index_data["RSI_14"] = calculate_rsi(index_data["index_close"], 14)

# 6. 趋势动量结合类因子（MACD）
def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

index_data["MACD"], index_data["MACD_SIGNAL"], index_data["MACD_HIST"] = calculate_macd(
    index_data["index_close"], fast=12, slow=26, signal=9
)

# ---------------------------- 保存结果 ----------------------------
output_file = "中证消费50指数核心因子数据.csv"
index_data.to_csv(output_file, index=False, encoding="utf_8_sig")

print(f"\n数据处理完成！结果已保存到: {output_file}")
# 获取时间范围时使用索引
print(f"时间范围: {index_data.index.min().date()} 至 {index_data.index.max().date()}")
print(f"包含核心因子: {len(index_data.columns)} 个（已剔除冗余因子）")
print("\n核心因子说明:")
core_factors = [
    "index_close", "index_pct_change", "index_volume",
    "MOM_20", "MOM_60", "MA_20",
    "VOL_20", "VOL_60",
    "VMA_5", "VOLUME_RATIO",
    "RSI_14",
    "MACD", "MACD_SIGNAL", "MACD_HIST"
]
