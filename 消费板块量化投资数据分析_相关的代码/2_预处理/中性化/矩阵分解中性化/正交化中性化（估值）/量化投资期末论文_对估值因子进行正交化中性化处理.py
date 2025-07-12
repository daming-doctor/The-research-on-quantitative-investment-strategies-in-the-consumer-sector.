import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
def orthogonalize_factors(factors):
    """
    使用QR分解对因子矩阵进行正交化处理

    参数:
    factors -- 原始因子矩阵 (n_samples, n_factors)

    返回:
    Q -- 正交化后的因子矩阵
    R -- 上三角矩阵
    """
    # 1. 标准化因子数据
    scaler = StandardScaler()
    factors_std = scaler.fit_transform(factors)

    # 2. 执行QR分解
    Q, R = np.linalg.qr(factors_std, mode='reduced')

    # 3. 将正交因子重新标准化为均值0方差1
    Q = StandardScaler().fit_transform(Q)

    return Q, R


def visualize_results(original, orthogonalized, factor_names):
    """
    可视化正交化前后的因子分布和相关性

    参数:
    original -- 原始因子矩阵
    orthogonalized -- 正交化后的因子矩阵
    factor_names -- 因子名称列表
    """
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, name in enumerate(factor_names):
        sns.histplot(original[:, i], ax=axes[0, 0], kde=True, label=name, alpha=0.6)
    axes[0, 0].set_title('原始因子分布')
    axes[0, 0].legend()

    for i, name in enumerate(factor_names):
        sns.histplot(orthogonalized[:, i], ax=axes[0, 1], kde=True, label=name, alpha=0.6)
    axes[0, 1].set_title('正交化后因子分布')
    axes[0, 1].legend()

    orig_corr = np.corrcoef(original.T)
    sns.heatmap(orig_corr, annot=True, fmt=".2f", ax=axes[1, 0],
                cmap='coolwarm', xticklabels=factor_names, yticklabels=factor_names)
    axes[1, 0].set_title('原始因子相关性')

    ortho_corr = np.corrcoef(orthogonalized.T)
    sns.heatmap(ortho_corr, annot=True, fmt=".2f", ax=axes[1, 1],
                cmap='coolwarm', xticklabels=factor_names, yticklabels=factor_names)
    axes[1, 1].set_title('正交化后因子相关性')

    plt.tight_layout()
    plt.savefig('因子正交化结果.png', dpi=300)
    plt.show()


def main():
    file_path = r"C:\Users\86133\Desktop\各种资料\python\量化\使用HUBER M离群化处理的因子标准化\消费50估值因子_huber处理_Z.csv"
    df = pd.read_csv(file_path, header=1,names=['PEG值','市净率','PE(静)','市销率'],encoding='gbk')

    # 获取因子名称和矩阵
    factor_names = df.columns.tolist()
    original_factors = df.values

    # 执行正交化
    orthogonalized_factors, R_matrix = orthogonalize_factors(original_factors)

    # 打印结果摘要
    print("=" * 60)
    print("因子正交化处理结果摘要")
    print("=" * 60)
    print(f"原始因子矩阵形状: {original_factors.shape}")
    print(f"正交化后因子矩阵形状: {orthogonalized_factors.shape}")
    print("\nQR分解中的R矩阵:")
    print(np.round(R_matrix, 4))

    # 计算正交化前后的相关性
    orig_corr = np.corrcoef(original_factors.T)
    ortho_corr = np.corrcoef(orthogonalized_factors.T)

    print("\n原始因子平均相关性: {:.4f}".format(np.mean(np.abs(orig_corr - np.eye(orig_corr.shape[0])))))
    print("正交化后因子平均相关性: {:.4f}".format(np.mean(np.abs(ortho_corr - np.eye(ortho_corr.shape[0])))))

    # 保存结果
    result_df = pd.DataFrame(orthogonalized_factors, columns=[f"{name}_正交化" for name in factor_names])
    result_df.to_csv("消费50估值因子_正交化处理.csv", index=False)
    print("\n正交化结果已保存至: 消费50估值因子_正交化处理.csv")

    # 可视化结果
    visualize_results(original_factors, orthogonalized_factors, factor_names)

if __name__ == "__main__":
    main()