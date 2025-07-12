import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge  # 改用岭回归提高稳定性
from tqdm import tqdm
import warnings
from scipy.stats import zscore  # 用于标准化处理

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文显示和美观样式
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial Unicode MS']  # 添加备用字体
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid", font="SimSun")

# ===================== 1. 数据准备 =====================
# 读取中性化因子数据
neutral_df = pd.read_csv("中性化_消费50估值因子.csv")

# 提取因子列名
factor_columns = ['PEG_neutral', 'PB_neutral', 'PE_neutral', 'PS_neutral']

# 只保留需要的列
neutral_df = neutral_df[['trade_date', 'stock_code'] + factor_columns].copy()

neutral_df[factor_columns] = neutral_df[factor_columns].fillna(neutral_df[factor_columns].mean())

# 生成正交化因子（使用PCA方法进行正交化）
def generate_orthogonal_factors(df, factor_cols):
    """使用PCA方法生成正交化因子"""
    from sklearn.decomposition import PCA

    # 标准化因子
    factors = df[factor_cols].values
    factors_standardized = zscore(factors, nan_policy='omit')

    # 使用PCA进行正交化
    pca = PCA(n_components=len(factor_cols))
    orthogonal_factors = pca.fit_transform(factors_standardized)

    # 创建正交化因子的DataFrame
    ortho_df = df[['trade_date', 'stock_code']].copy()
    for i, col in enumerate(factor_cols):
        ortho_df[f'{col}_ortho'] = orthogonal_factors[:, i]

    return ortho_df


# 生成正交化因子
ortho_df = generate_orthogonal_factors(neutral_df, factor_columns)


# 生成模拟收益率数据（假设下一期收益率）
def generate_returns(factors, n_factors=4):
    """生成模拟收益率数据"""
    np.random.seed(42)
    # 真实因子权重 - 与因子数量匹配
    true_weights = np.array([0.3, -0.2, 0.4, -0.1])[:n_factors]  # 固定权重提高稳定性

    # 生成收益率 = 因子暴露 * 因子收益 + 噪声
    returns = factors @ true_weights + np.random.normal(0, 0.2, len(factors))
    return returns


# 生成收益率数据
ortho_returns = generate_returns(ortho_df.iloc[:, 2:].values)
neutral_returns = generate_returns(neutral_df[factor_columns].values)


# ===================== 2. 蒙特卡罗模拟 =====================
def monte_carlo_backtest(factors, returns, n_simulations=1000):
    """蒙特卡罗模拟回测 - 使用岭回归提高稳定性"""
    results = []
    n_stocks = len(factors)

    for _ in tqdm(range(n_simulations), desc="蒙特卡罗模拟"):
        # 随机抽样构建投资组合
        sample_idx = np.random.choice(n_stocks, size=int(n_stocks * 0.8), replace=False)
        test_idx = np.setdiff1d(np.arange(n_stocks), sample_idx)

        if len(test_idx) < 10:  # 确保测试集足够大
            continue

        # 训练集因子暴露和收益率
        X_train = factors[sample_idx]
        y_train = returns[sample_idx]

        # 使用岭回归训练因子权重模型
        model = Ridge(alpha=1.0)  # 添加L2正则化
        model.fit(X_train, y_train)
        weights = model.coef_

        # 测试集预测
        X_test = factors[test_idx]
        predicted_returns = model.predict(X_test)
        actual_returns = returns[test_idx]

        # 构建多空组合
        sorted_idx = np.argsort(predicted_returns)
        n_test = len(test_idx)
        long_idx = sorted_idx[-5:]  # 前5名做多
        short_idx = sorted_idx[:5]  # 后5名做空

        # 计算组合收益率
        long_return = actual_returns[long_idx].mean()
        short_return = actual_returns[short_idx].mean()
        portfolio_return = long_return - short_return

        results.append({
            "long_return": long_return,
            "short_return": short_return,
            "portfolio_return": portfolio_return,
            "weights": weights.copy()
        })

    return pd.DataFrame(results)


# 对两种因子处理方法进行模拟
print("估值正交化中性化因子模拟中...")
ortho_results = monte_carlo_backtest(ortho_df.iloc[:, 2:].values, ortho_returns, n_simulations=1000)

print("估值线性回归中性化因子模拟中...")
neutral_results = monte_carlo_backtest(neutral_df[factor_columns].values, neutral_returns, n_simulations=1000)


# ===================== 3. 结果分析 =====================
# 计算关键指标
def calculate_metrics(results, name):
    """计算回测指标"""
    portfolio_returns = results["portfolio_return"]
    if len(portfolio_returns) == 0:
        return {
            "Method": name,
            "平均收益": 0,
            "波动率": 0,
            "夏普比率": 0,

        }


    return {
        "Method": name,
        "平均收益": portfolio_returns.mean(),
        "波动率": portfolio_returns.std(),
        "夏普比率": portfolio_returns.mean() / (portfolio_returns.std() + 1e-8),

    }


# 生成指标对比
metrics_df = pd.DataFrame([
    calculate_metrics(ortho_results, "矩阵分解正交化因子"),
    calculate_metrics(neutral_results, "线性回归因子")
])


# ===================== 4. 结果可视化 =====================
def plot_comparison_revised(ortho_results, neutral_results, metrics_df):
    """可视化比较结果 - 移除指定的两个指标"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle("因子处理方法对比分析", fontsize=16, fontweight='bold')

    # 1. 因子权重稳定性
    ax = axes[0]
    ortho_weights = np.vstack(ortho_results["weights"])
    neutral_weights = np.vstack(neutral_results["weights"])

    # 处理极端值
    ortho_weights = np.nan_to_num(ortho_weights, nan=0, posinf=0, neginf=0)
    neutral_weights = np.nan_to_num(neutral_weights, nan=0, posinf=0, neginf=0)

    # 限制权重范围
    ortho_weights = np.clip(ortho_weights, -1, 1)
    neutral_weights = np.clip(neutral_weights, -1, 1)

    factor_names = ['PEG', 'PB', 'PE', 'PS']

    # 计算权重波动性
    ortho_std = ortho_weights.std(axis=0)
    neutral_std = neutral_weights.std(axis=0)

    # 绘制波动性对比
    x = np.arange(len(factor_names))
    width = 0.35
    rects1 = ax.bar(x - width / 2, ortho_std, width, label='矩阵分解正交化', color='skyblue')
    rects2 = ax.bar(x + width / 2, neutral_std, width, label='线性回归', color='salmon')

    ax.set_ylabel('权重波动率')
    ax.set_title('因子权重稳定性对比')
    ax.set_xticks(x)
    ax.set_xticklabels(factor_names)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # 2. 指标对比表格
    ax = axes[1]
    table_data = metrics_df.set_index("Method").T

    # 创建美观的表格
    cell_text = []
    for row in range(len(table_data)):
        row_values = []
        for val in table_data.iloc[row]:
            # 根据指标类型格式化
            if "夏普" in table_data.index[row]:
                row_values.append(f"{val:.4f}")
            else:
                row_values.append(f"{val:.4f}")
        cell_text.append(row_values)

    # 创建带样式的表格
    table = ax.table(cellText=cell_text,
                     rowLabels=table_data.index,
                     colLabels=table_data.columns,
                     loc="center",
                     cellLoc='center',
                     colColours=['#d9edf7'] * len(table_data.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # 设置标题
    ax.set_title("性能指标对比", pad=20)
    ax.axis("off")

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    try:
        plt.savefig("估值因子处理方法对比.png", dpi=300, bbox_inches='tight')
        print("图像已成功保存至: 估值因子处理方法对比.png")
    except Exception as e:
        print(f"保存图像时出错: {e}")

    plt.show()


# 执行可视化 (使用修改后的函数)
plot_comparison_revised(ortho_results, neutral_results, metrics_df)

# 返回关键指标
print("=" * 50)
print("因子处理方法性能对比:")
print("=" * 50)
print(metrics_df)

# 保存结果到CSV
try:
    ortho_results.to_csv("估值矩阵分解正交化中性化因子模拟结果.csv", index=False)
    neutral_results.to_csv("估值线性回归中性化因子模拟结果.csv", index=False)
    metrics_df.to_csv("估值因子处理方法对比指标.csv", index=False)
    print("结果已成功保存至CSV文件")
except Exception as e:
    print(f"保存结果时出错: {e}")