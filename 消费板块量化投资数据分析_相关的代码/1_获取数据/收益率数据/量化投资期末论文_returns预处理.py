import pandas as pd
import os

# 1. 更可靠的文件读取方式
try:
    # 尝试多种可能的文件分隔符
    df = pd.read_csv('returns(SZ).csv', sep=None, engine='python', header=0)

    # 调试输出：查看读取的列名
    print("读取的列名:", df.columns.tolist())
    print("读取的行数:", len(df))

    # 2. 处理列名：确保第一列正确命名
    # 如果第一列是未命名状态（常见情况）
    if df.columns[0].startswith('Unnamed'):
        df = df.rename(columns={df.columns[0]: 'trade_date'})

    # 3. 处理股票代码列名
    new_columns = []
    for col in df.columns:
        if col == 'trade_date':
            new_columns.append(col)
        elif '.' in col:
            # 移除.SH/.SZ后缀
            new_columns.append(col.split('.')[0])
        else:
            new_columns.append(col)

    df.columns = new_columns

    # 4. 转换成长格式
    long_df = df.melt(
        id_vars='trade_date',
        var_name='stock_code',
        value_name='returns'
    )

    # 5. 处理空值 - 更安全的做法
    if not long_df.empty:
        long_df = long_df.dropna(subset=['returns'])
        print(f"转换成功！共处理 {len(long_df)} 条记录")
        print("数据样例:")
        print(long_df.head())

        # 6. 保存结果
        long_df.to_csv('returns(SZ)预处理后数据.csv', index=False)
    else:
        print("警告：转换后数据框为空，请检查原始数据格式")
        print("原始数据样例:")
        print(df.head())

except Exception as e:
    print(f"处理出错: {str(e)}")
    print("请检查：")
    print("1. 文件路径是否正确")
    print("2. 文件内容格式是否与描述一致")
    print("3. 尝试在文本编辑器中查看文件内容")

    # 调试：打印文件前几行
    if os.path.exists('returns.txt'):
        with open('returns.txt', 'r') as f:
            print("文件前5行内容:")
            for i in range(5):
                print(f.readline().strip())