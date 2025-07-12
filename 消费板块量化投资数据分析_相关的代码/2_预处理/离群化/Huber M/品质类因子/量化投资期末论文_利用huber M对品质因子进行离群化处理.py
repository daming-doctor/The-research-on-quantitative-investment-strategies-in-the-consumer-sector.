import pandas as pd
from sklearn.linear_model import HuberRegressor
import numpy as np

# 加载数据
data = pd.read_csv('消费50品质因子数据（补齐）.csv')

# 查看数据基本信息
print('数据基本信息：')
data.info()

# 查看数据集行数和列数
rows, columns = data.shape

if rows < 100 and columns < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('数据全部内容信息：')
    print(data.to_csv(sep='\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('数据前几行内容信息：')
    print(data.head().to_csv(sep='\t', na_rep='nan'))

# 提取特征
features = ['存货周转率(次)', '总资产周转率(次)', '流动资产周转率(次)', '资本固定化比率(%)']
X = data[features]

# 初始化 HuberRegressor
huber = HuberRegressor()

# 对每个因子分别进行拟合和预测，以处理离群值
for feature in features:
    y = X[feature].values.reshape(-1, 1)
    huber.fit(y, y)
    X[feature] = huber.predict(y)

# 将处理后的结果保存为新的 CSV 文件
path = '消费50品质因子数据_huber处理.csv'
X.to_csv(path, index=False)