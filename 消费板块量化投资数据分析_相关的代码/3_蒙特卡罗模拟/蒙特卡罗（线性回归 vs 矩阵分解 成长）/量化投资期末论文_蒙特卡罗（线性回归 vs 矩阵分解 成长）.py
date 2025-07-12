import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from tqdm import tqdm
import warnings
from scipy.stats import zscore
from sklearn.decomposition import PCA

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文显示和美观样式
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid", font="SimSun")

# ===================== 1. 数据准备 =====================
# 读取中性化因子数据
neutral_df = pd.read_csv("中性化_消费50成长因子.csv")

# 提取因子列名
factor_columns = ['NetProfit_Growth_neutral', 'NetAsset_Growth_neutral', 'Revenue_Growth_neutral', 'ROE_neutral',
                  'ROA_neutral']

# 只保留需要的列
neutral_df = neutral_df[['trade_date', 'stock_code'] + factor_columns].copy()
neutral_df[factor_columns] = neutral_df[factor_columns].fillna(neutral_df[factor_columns].mean())


# 生成正交化因子
def generate_orthogonal_factors(df, factor_cols):
    """使用PCA方法生成正交化因子"""
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


# ===================== 2. 蒙特卡罗模拟 - 仅关注因子稳定性 =====================
def monte_carlo_factor_stability(factors, n_simulations=1000):
    """蒙特卡罗模拟 - 评估因子权重稳定性"""
    results = []
    n_stocks = len(factors)

    for _ in tqdm(range(n_simulations), desc="蒙特卡罗模拟"):
        # 随机抽样
        sample_idx = np.random.choice(n_stocks, size=int(n_stocks * 0.8), replace=False)

        # 使用岭回归估计因子权重
        model = Ridge(alpha=1.0)

        # 随机生成模拟因子收益
        random_returns = np.random.normal(0, 1, len(sample_idx))

        # 训练模型
        X_train = factors[sample_idx]
        y_train = random_returns
        model.fit(X_train, y_train)

        results.append({
            "weights": model.coef_.copy()
        })

    return pd.DataFrame(results)


# 对两种因子处理方法进行模拟
print("成长正交化中性化因子稳定性分析中...")
ortho_results = monte_carlo_factor_stability(ortho_df.iloc[:, 2:].values, n_simulations=1000)

print("成长线性回归中性化因子稳定性分析中...")
neutral_results = monte_carlo_factor_stability(neutral_df[factor_columns].values, n_simulations=1000)


# ===================== 3. 结果分析 =====================
def calculate_stability_metrics(results, name):
    """计算因子权重稳定性指标"""
    weights = np.vstack(results["weights"])

    # 处理极端值
    weights = np.nan_to_num(weights, nan=0, posinf=0, neginf=0)
    weights = np.clip(weights, -1, 1)

    # 计算稳定性指标
    weight_stds = weights.std(axis=0)
    weight_abs_mean = np.abs(weights).mean(axis=0)

    return {
        "Method": name,
        "权重平均绝对值": weight_abs_mean.mean(),
        "权重波动率": weight_stds.mean(),
        "权重稳定性指数": 1 / (weight_stds.mean() + 1e-8)  # 波动率越低，稳定性越高
    }


# 生成指标对比
metrics_df = pd.DataFrame([
    calculate_stability_metrics(ortho_results, "矩阵分解正交化因子"),
    calculate_stability_metrics(neutral_results, "线性回归因子")
])


# ===================== 4. 结果可视化 =====================
def plot_factor_stability_comparison(ortho_results, neutral_results, metrics_df):
    """可视化因子稳定性对比"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle("因子处理方法稳定性对比分析", fontsize=16, fontweight='bold')

    # 1. 因子权重分布
    ax = axes[0]

    # 提取权重数据
    ortho_weights = np.vstack(ortho_results["weights"])
    neutral_weights = np.vstack(neutral_results["weights"])

    # 处理极端值
    ortho_weights = np.nan_to_num(ortho_weights, nan=0, posinf=0, neginf=0)
    neutral_weights = np.nan_to_num(neutral_weights, nan=0, posinf=0, neginf=0)
    ortho_weights = np.clip(ortho_weights, -1, 1)
    neutral_weights = np.clip(neutral_weights, -1, 1)

    # 绘制核密度估计图
    sns.kdeplot(ortho_weights.flatten(), ax=ax, label='矩阵分解正交化', fill=True, alpha=0.5, color='skyblue')
    sns.kdeplot(neutral_weights.flatten(), ax=ax, label='线性回归', fill=True, alpha=0.5, color='salmon')

    ax.set_title('因子权重分布对比')
    ax.set_xlabel('因子权重')
    ax.set_ylabel('密度')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # 2. 稳定性指标对比
    ax = axes[1]

    # 准备表格数据
    table_data = metrics_df.set_index("Method")[["权重波动率", "权重稳定性指数"]].T
    cell_text = [
        [f"{val:.4f}" for val in table_data.iloc[0]],
        [f"{val:.4f}" for val in table_data.iloc[1]]
    ]

    # 创建表格
    table = ax.table(cellText=cell_text,
                     rowLabels=table_data.index,
                     colLabels=table_data.columns,
                     loc="center",
                     cellLoc='center',
                     colColours=['#d9edf7'] * len(table_data.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title("稳定性指标对比", pad=20)
    ax.axis("off")

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    try:
        plt.savefig("成长因子稳定性对比分析.png", dpi=300, bbox_inches='tight')
        print("图像已成功保存至: 成长因子稳定性对比分析.png")
    except Exception as e:
        print(f"保存图像时出错: {e}")

    plt.show()


# 执行可视化
plot_factor_stability_comparison(ortho_results, neutral_results, metrics_df)

# 返回关键指标
print("=" * 50)
print("因子处理方法稳定性对比:")
print("=" * 50)
print(metrics_df)

# 保存结果到CSV
try:
    ortho_results.to_csv("成长矩阵分解正交化因子稳定性结果.csv", index=False)
    neutral_results.to_csv("成长线性回归中性化因子稳定性结果.csv", index=False)
    metrics_df.to_csv("成长因子稳定性对比指标.csv", index=False)
    print("结果已成功保存至CSV文件")
except Exception as e:
    print(f"保存结果时出错: {e}")