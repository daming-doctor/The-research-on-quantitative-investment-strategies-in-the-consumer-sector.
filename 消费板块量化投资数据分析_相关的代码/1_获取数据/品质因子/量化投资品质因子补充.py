import akshare as ak
import pandas as pd
from datetime import datetime

# 股票列表
stock_list = [
    {"code": "300765", "name": "新诺威"},
    {"code": "300957", "name": "贝泰妮"},
    {"code": "300999", "name": "金龙鱼"},
    {"code": "301498", "name": "乖宝宠物"},
    {"code": "600132", "name": "重庆啤酒"},
    {"code": "600298", "name": "安琪酵母"},
    {"code": "600299", "name": "安迪苏"},
    {"code": "600519", "name": "贵州茅台"},
    {"code": "600598", "name": "北大荒"},
    {"code": "600600", "name": "青岛啤酒"},
    {"code": "600737", "name": "中粮糖业"},
    {"code": "600809", "name": "山西汾酒"},
    {"code": "600872", "name": "中炬高新"},
    {"code": "600873", "name": "梅花生物"},
    {"code": "600887", "name": "伊利股份"},
    {"code": "601118", "name": "海南橡胶"},
    {"code": "603156", "name": "养元饮品"},
    {"code": "603288", "name": "海天味业"},
    {"code": "603345", "name": "安井食品"},
    {"code": "603369", "name": "今世缘"},
    {"code": "603589", "name": "口子窖"},
    {"code": "603605", "name": "珀莱雅"},
    {"code": "605499", "name": "东鹏饮料"},
    {"code": "688363", "name": "华熙生物"}
]

# 需要的列
required_columns = ['日期', '股票代码', '股票名称', '存货周转率(次)', '总资产周转率(次)', '流动资产周转率(次)', '资本固定化比率(%)']
# 时间范围
start_date = "2023-06-06"
end_date = "2025-06-06"

# 创建空DataFrame存储结果
all_data = pd.DataFrame(columns=required_columns)

# 遍历股票列表
for stock in stock_list:
    code = stock["code"]
    name = stock["name"]

    try:
        # 获取财务指标
        df = ak.stock_financial_analysis_indicator(symbol=code,start_year='2023')

        # 添加股票代码和名称
        df['股票代码'] = code
        df['股票名称'] = name

        # 筛选日期范围内的数据
        df['日期'] = pd.to_datetime(df['日期'])
        mask = (df['日期'] >= datetime.strptime(start_date, "%Y-%m-%d")) & \
               (df['日期'] <= datetime.strptime(end_date, "%Y-%m-%d"))
        df_filtered = df.loc[mask]

        # 筛选需要的列
        available_columns = [col for col in required_columns if col in df_filtered.columns]
        result = df_filtered[available_columns]

        # 添加到总数据
        all_data = pd.concat([all_data, result], ignore_index=True)

        print(f"成功获取 {name}({code}) 数据，记录数: {len(result)}")

    except Exception as e:
        print(f"获取 {name}({code}) 数据失败: {str(e)}")

# 保存结果到CSV
all_data.to_csv("缺失的品质因子.csv", index=False, encoding='utf_8_sig')
print("=" * 50)
print(f"数据获取完成! 共获取 {len(all_data)} 条记录")
print(f"结果已保存到: 缺失的品质因子.csv")

if not all_data.empty:
    print("\n预览数据:")
    print(all_data.head())
else:
    print("\n未获取到有效数据")