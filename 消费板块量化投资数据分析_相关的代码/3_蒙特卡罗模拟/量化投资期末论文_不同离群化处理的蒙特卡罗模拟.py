import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
from sklearn.linear_model import HuberRegressor

# 设置svg后端和字体选项
plt.rcParams["backend"] = "svg"  # 使用svg后端
plt.rcParams["svg.fonttype"] = "none"  # 直接使用系统字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
sns.set(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 8)


def load_and_prepare_data(huber_file, mad_file):
    """
    加载并准备数据
    :param huber_file: Huber处理后的文件路径
    :param mad_file: MAD处理后的文件路径
    :return: 处理后的DataFrame（Huber和MAD因子值）
    """
    # 读取Huber处理后的数据
    huber_df = pd.read_csv(huber_file, encoding='gbk')

    # 读取MAD处理后的数据，指定列名
    mad_df = pd.read_csv(mad_file, header=0,
                         names=['date', 'stock_code',
                                '存货周转率(次)','总资产周转率(次)','流动资产周转率(次)','资本固定化比率(%)'

])

    # 确保两个数据框有相同的行数
    min_rows = min(len(huber_df), len(mad_df))
    huber_df = huber_df.iloc[:min_rows]
    mad_df = mad_df.iloc[:min_rows]

    # 提取因子列（4个因子）
    factor_columns = ['存货周转率(次)','总资产周转率(次)','流动资产周转率(次)','资本固定化比率(%)'

]

    # 创建比较DataFrame
    comparison_df = pd.DataFrame()
    for i, col in enumerate(factor_columns):
        comparison_df[f'Huber_{col}'] = huber_df.iloc[:, i]
        comparison_df[f'MAD_{col}'] = mad_df[col]

    return comparison_df


def monte_carlo_comparison(comparison_df, n_simulations=1000):
    """
    执行蒙特卡洛模拟比较
    :param comparison_df: 包含Huber和MAD因子值的DataFrame
    :param n_simulations: 模拟次数
    :return: 包含结果的DataFrame
    """
    results = []
    factor_names = ['存货周转率(次)','总资产周转率(次)','流动资产周转率(次)','资本固定化比率(%)'

]

    for factor in factor_names:
        print(f"正在处理因子: {factor}...")
        huber_col = f'Huber_{factor}'
        mad_col = f'MAD_{factor}'

        # 提取有效数据（非NaN）
        valid_idx = comparison_df[[huber_col, mad_col]].notnull().all(axis=1)
        huber_vals = comparison_df.loc[valid_idx, huber_col].values
        mad_vals = comparison_df.loc[valid_idx, mad_col].values

        if len(huber_vals) < 10 or len(mad_vals) < 10:
            print(f"因子 {factor} 的有效数据不足，跳过...")
            continue

        factor_results = {
            '因子': factor,
            'Huber_IC': [],
            'MAD_IC': [],
            'IC差异': [],
            '因子值差异': []
        }

        for _ in range(n_simulations):
            # 随机生成收益数据（基于标准正态分布）
            returns = np.random.normal(0, 1, len(huber_vals))

            # 计算Huber因子的IC
            huber_ic, _ = pearsonr(huber_vals, returns)

            # 计算MAD因子的IC
            mad_ic, _ = pearsonr(mad_vals, returns)

            # 记录结果
            factor_results['Huber_IC'].append(huber_ic)
            factor_results['MAD_IC'].append(mad_ic)
            factor_results['IC差异'].append(huber_ic - mad_ic)
            factor_results['因子值差异'].append(np.mean(np.abs(huber_vals - mad_vals)))

        # 计算统计量
        results.append({
            '因子': factor,
            'Huber_IC均值': np.mean(factor_results['Huber_IC']),
            'MAD_IC均值': np.mean(factor_results['MAD_IC']),
            'IC差异均值': np.mean(factor_results['IC差异']),
            'IC差异标准差': np.std(factor_results['IC差异']),
            '因子值差异均值': np.mean(factor_results['因子值差异']),
            '因子值差异标准差': np.std(factor_results['因子值差异']),
            '有效样本数': len(huber_vals)
        })

    return pd.DataFrame(results)


def statistical_tests(comparison_df):
    """
    执行统计检验
    :param comparison_df: 包含Huber和MAD因子值的DataFrame
    :return: 包含统计检验结果的DataFrame
    """
    test_results = []
    factor_names = ['存货周转率(次)','总资产周转率(次)','流动资产周转率(次)','资本固定化比率(%)'

]

    for factor in factor_names:
        huber_col = f'Huber_{factor}'
        mad_col = f'MAD_{factor}'

        # 提取有效数据
        valid_idx = comparison_df[[huber_col, mad_col]].notnull().all(axis=1)
        huber_vals = comparison_df.loc[valid_idx, huber_col]
        mad_vals = comparison_df.loc[valid_idx, mad_col]

        if len(huber_vals) < 10 or len(mad_vals) < 10:
            continue

        # 配对t检验（因子值差异）
        t_stat, p_value = ttest_ind(huber_vals, mad_vals)

        # 相关性检验
        corr, corr_p = pearsonr(huber_vals, mad_vals)

        # 计算绝对差异的统计量
        abs_diff = np.abs(huber_vals - mad_vals)

        test_results.append({
            '因子': factor,
            't统计量': t_stat,
            'p值': p_value,
            '相关性': corr,
            '相关性p值': corr_p,
            '平均绝对差异': np.mean(abs_diff),
            '绝对差异标准差': np.std(abs_diff),
            '最大绝对差异': np.max(abs_diff),
            '最小绝对差异': np.min(abs_diff),
            '有效样本数': len(huber_vals)
        })

    return pd.DataFrame(test_results)


def plot_comparison_results(comparison_df, mc_results, stats_results):
    """
    可视化比较结果
    :param comparison_df: 包含Huber和MAD因子值的DataFrame
    :param mc_results: 蒙特卡洛模拟结果
    :param stats_results: 统计检验结果
    """
    factor_names = ['存货周转率(次)','总资产周转率(次)','流动资产周转率(次)','资本固定化比率(%)'

]

    # 1. IC比较图
    plt.figure(figsize=(14, 8))
    bar_width = 0.35
    index = np.arange(len(factor_names))

    plt.bar(index, mc_results['Huber_IC均值'], bar_width, label='Huber IC')
    plt.bar(index + bar_width, mc_results['MAD_IC均值'], bar_width, label='MAD IC')

    plt.xlabel('因子')
    plt.ylabel('平均IC值')
    plt.title('蒙特卡洛模拟：Huber vs MAD 平均IC值比较')
    plt.xticks(index + bar_width / 2, factor_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('IC_comparison.svg', format='svg')  # 保存为svg格式
    plt.show()

    # 2. 因子值差异分布图
    plt.figure(figsize=(14, 8))
    for factor in factor_names:
        huber_col = f'Huber_{factor}'
        mad_col = f'MAD_{factor}'
        diff = comparison_df[huber_col] - comparison_df[mad_col]
        sns.kdeplot(diff.dropna(), label=factor, fill=True, alpha=0.3)

    plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('因子值差异 (Huber - MAD)')
    plt.ylabel('密度')
    plt.title('因子值差异分布')
    plt.legend()
    plt.tight_layout()
    plt.savefig('factor_difference_distribution.svg', format='svg')  # 保存为svg格式
    plt.show()

    # 3. 相关性热力图
    plt.figure(figsize=(12, 10))
    corr_matrix = pd.DataFrame()

    for factor in factor_names:
        huber_col = f'Huber_{factor}'
        mad_col = f'MAD_{factor}'
        corr_matrix.loc[factor, 'Huber-MAD相关性'] = stats_results[stats_results['因子'] == factor]['相关性'].values[0]

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Huber与MAD因子值相关性')
    plt.tight_layout()
    plt.savefig('factor_correlation.svg', format='svg')  # 保存为svg格式
    plt.show()

    # 4. 因子值散点图矩阵（示例：第一个因子）
    if len(factor_names) > 0:
        plt.figure(figsize=(10, 8))
        factor = factor_names[0]
        huber_col = f'Huber_{factor}'
        mad_col = f'MAD_{factor}'

        sns.scatterplot(data=comparison_df, x=huber_col, y=mad_col, alpha=0.6)
        plt.plot([comparison_df[huber_col].min(), comparison_df[huber_col].max()],
                 [comparison_df[huber_col].min(), comparison_df[huber_col].max()],
                 'r--')
        plt.xlabel(f'Huber {factor}')
        plt.ylabel(f'MAD {factor}')
        plt.title(f'{factor} 值比较')
        plt.tight_layout()
        plt.savefig(f'scatter_{factor}.svg', format='svg')  # 保存为svg格式
        plt.show()


def main():
    huber_file = "消费50品质因子数据_huber处理_Z.csv"
    mad_file = "processed_消费50品质因子数据（补齐)_R_Z.csv"

    print("正在加载数据...")
    comparison_df = load_and_prepare_data(huber_file, mad_file)
    print(f"数据加载完成，共 {len(comparison_df)} 行数据")

    print("\n正在执行蒙特卡洛模拟...")
    mc_results = monte_carlo_comparison(comparison_df, n_simulations=1000)
    print("蒙特卡洛模拟完成!")

    print("\n正在执行统计检验...")
    stats_results = statistical_tests(comparison_df)
    print("统计检验完成!")

    print("\n正在生成结果可视化...")
    plot_comparison_results(comparison_df, mc_results, stats_results)
    print("可视化完成!")

    # 保存结果
    mc_results.to_csv("monte_carlo_results.csv", index=False, encoding='utf-8-sig')
    stats_results.to_csv("statistical_test_results.csv", index=False, encoding='utf-8-sig')
    comparison_df.to_csv("comparison_data.csv", index=False, encoding='utf-8-sig')

    print("\n结果摘要:")
    print(mc_results[['因子', 'Huber_IC均值', 'MAD_IC均值', 'IC差异均值']])
    print("\n统计检验结果:")
    print(stats_results[['因子', 't统计量', 'p值', '相关性', '平均绝对差异']])

    print("\n重要结论:")
    for _, row in mc_results.iterrows():
        factor = row['因子']
        ic_diff = row['IC差异均值']
        if ic_diff > 0:
            print(f"{factor}: Huber方法表现优于MAD方法 (IC差异: {ic_diff:.4f})")
        else:
            print(f"{factor}: MAD方法表现优于Huber方法 (IC差异: {ic_diff:.4f})")

    for _, row in stats_results.iterrows():
        factor = row['因子']
        p_value = row['p值']
        if p_value < 0.05:
            print(f"{factor}: 因子值存在显著差异 (p值: {p_value:.4f})")
        else:
            print(f"{factor}: 因子值无显著差异 (p值: {p_value:.4f})")


if __name__ == "__main__":
    main()
