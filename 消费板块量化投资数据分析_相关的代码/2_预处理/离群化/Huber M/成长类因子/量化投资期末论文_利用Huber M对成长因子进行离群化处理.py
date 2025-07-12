import pandas as pd
from sklearn.linear_model import HuberRegressor
import numpy as np

# 加载数据
data = pd.read_csv('消费50成长因子（补齐）.csv')

# 提取特征
features = ['净利润增长率(%)', '净资产增长率(%)', '主营业务收入增长率(%)', '净资产报酬率(%)', '资产报酬率(%)']
X = data[features]

# 处理缺失值
X = X.fillna(X.median())

# 初始化 HuberRegressor
huber = HuberRegressor()

# 对每个因子分别进行拟合和预测，以处理离群值
for feature in features:
    y = X[feature].values.reshape(-1, 1)
    huber.fit(y, y)
    X[feature] = huber.predict(y)

# 将处理后的结果保存为新的 CSV 文件
path = '消费50成长因子（补齐）_huber处理.csv'
X.to_csv(path, index=False)