import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.linalg import qr

plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 1. 数据加载和预处理
def load_and_preprocess(file_path):
    """加载数据并进行预处理"""
    df = pd.read_csv(file_path,
                     header=1,
                     names=['存货周转率(次)','总资产周转率(次)','流动资产周转率(次)','资本固定化比率(%)'],
                     encoding='gbk')

    # 检查数据
    print("原始数据形状:", df.shape)
    print("\n前5行数据:")
    print(df.head())

    # 检查缺失值
    print("\n缺失值统计:")
    print(df.isnull().sum())

    # 转换数据为矩阵
    data_matrix = df.values

    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_matrix)

    return scaled_data, df.columns.tolist(), df


# 2. 矩阵分解方法（包含正交化处理）
def perform_decompositions(data, n_components=2):
    """执行多种矩阵分解方法，包括正交化处理"""
    results = {}

    # PCA分解
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    results['PCA'] = {
        'components': pca.components_,
        'explained_variance': pca.explained_variance_ratio_,
        'result': pca_result
    }

    # SVD分解
    svd = TruncatedSVD(n_components=n_components)
    svd_result = svd.fit_transform(data)
    results['SVD'] = {
        'components': svd.components_,
        'explained_variance': svd.explained_variance_ratio_,
        'singular_values': svd.singular_values_,
        'result': svd_result
    }

    # NMF分解 (需要非负数据)
    non_neg_data = data - np.min(data) + 1e-6
    nmf = NMF(n_components=n_components, init='random', random_state=42)
    nmf_result = nmf.fit_transform(non_neg_data)
    results['NMF'] = {
        'components': nmf.components_,
        'reconstruction_error': nmf.reconstruction_err_,
        'result': nmf_result
    }

    # 正交化处理 (QR分解)
    # 中心化处理
    centered_data = data - np.mean(data, axis=0)
    Q, R = qr(centered_data, mode='economic')

    # 确保对角线元素为正（保持结果一致性）
    sign_adjust = np.sign(np.diag(R))
    sign_adjust[sign_adjust == 0] = 1
    Q = Q * sign_adjust

    # 取前两个正交因子
    orthogonal_result = Q[:, :n_components]

    results['QR'] = {
        'components': Q.T[:n_components],
        'result': orthogonal_result,
        'R_matrix': R
    }

    return results


# 3. 结果可视化
def visualize_results(results, feature_names):
    """可视化各种分解方法的结果"""
    plt.figure(figsize=(18, 15))

    # PCA可视化
    plt.subplot(3, 3, 1)
    pca_result = results['PCA']['result']
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
    plt.title('PCA投影')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    # PCA载荷图
    plt.subplot(3, 3, 4)
    components = results['PCA']['components']
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, components[0, i], components[1, i],
                  head_width=0.05, head_length=0.05, fc='k', ec='k')
        plt.text(components[0, i] * 1.15, components[1, i] * 1.15, feature,
                 fontsize=9, ha='center', va='center')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid()
    plt.title('PCA因子载荷')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    # SVD可视化
    plt.subplot(3, 3, 2)
    svd_result = results['SVD']['result']
    plt.scatter(svd_result[:, 0], svd_result[:, 1], alpha=0.6, c='green')
    plt.title('SVD投影')
    plt.xlabel('SV1')
    plt.ylabel('SV2')

    # NMF可视化
    plt.subplot(3, 3, 3)
    nmf_result = results['NMF']['result']
    plt.scatter(nmf_result[:, 0], nmf_result[:, 1], alpha=0.6, c='purple')
    plt.title('NMF投影')
    plt.xlabel('成分1')
    plt.ylabel('成分2')

    # 方差解释率
    plt.subplot(3, 3, 5)
    methods = ['PCA', 'SVD']
    for method in methods:
        explained_variance = results[method]['explained_variance']
        plt.plot(np.cumsum(explained_variance), 'o-', label=method)
    plt.title('累计解释方差')
    plt.xlabel('主成分数量')
    plt.ylabel('累计解释方差')
    plt.legend()
    plt.grid()

    # t-SNE可视化
    plt.subplot(3, 3, 6)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(data)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6, c='orange')
    plt.title('t-SNE投影')

    # QR正交化结果可视化
    plt.subplot(3, 3, 7)
    qr_result = results['QR']['result']
    plt.scatter(qr_result[:, 0], qr_result[:, 1], alpha=0.6, c='blue')
    plt.title('QR正交化结果')
    plt.xlabel('正交因子1')
    plt.ylabel('正交因子2')

    # 正交化因子相关性验证
    plt.subplot(3, 3, 8)
    # 计算正交因子的相关系数矩阵
    qr_df = pd.DataFrame(results['QR']['result'], columns=['正交因子1', '正交因子2'])
    corr_matrix = qr_df.corr().values

    # 绘制相关系数矩阵热力图
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('正交因子相关系数矩阵')
    plt.xticks([0, 1], ['正交因子1', '正交因子2'])
    plt.yticks([0, 1], ['正交因子1', '正交因子2'])

    # 添加相关系数值
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            plt.text(j, i, f'{corr_matrix[i, j]:.4f}',
                     ha='center', va='center', color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig('matrix_decomposition_results.png', dpi=300)
    plt.show()

    # 验证正交性
    Q = results['QR']['result']
    QTQ = np.dot(Q.T, Q)
    identity = np.eye(Q.shape[1])

    # 检查是否接近单位矩阵
    is_orthogonal = np.allclose(QTQ, identity, atol=1e-8)
    print(f"\n正交性验证: {'成功' if is_orthogonal else '失败'}")
    print("Q^T * Q 矩阵:")
    print(QTQ.round(4))


# 4. 聚类分析
def cluster_analysis(results):
    """使用正交化因子进行聚类分析"""
    # 使用QR正交化结果进行聚类
    qr_result = results['QR']['result']

    # 使用肘部法则确定最佳聚类数
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(qr_result)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, 'o-')
    plt.title('肘部法则确定最优聚类数')
    plt.xlabel('聚类数量')
    plt.ylabel('组内平方和(WCSS)')
    plt.grid()
    plt.savefig('elbow_method.png', dpi=300)
    plt.show()

    # 根据肘部法则选择聚类数
    n_clusters = 3  # 根据肘部图选择
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(qr_result)

    # 可视化聚类结果
    plt.figure(figsize=(10, 8))
    plt.scatter(qr_result[:, 0], qr_result[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=300, c='red', marker='X', label='聚类中心')
    plt.title(f'基于正交因子的K-means聚类 (k={n_clusters})')
    plt.xlabel('正交因子1')
    plt.ylabel('正交因子2')
    plt.legend()
    plt.colorbar(label='聚类')
    plt.savefig('orthogonal_clustering.png', dpi=300)
    plt.show()

    return clusters


# 5. 重建误差分析
def reconstruction_error_analysis(data, results):
    """分析重建误差并识别异常点"""
    # 计算PCA重建误差
    pca = PCA(n_components=2)
    pca.fit(data)
    pca_reconstructed = pca.inverse_transform(pca.transform(data))
    pca_error = np.mean((data - pca_reconstructed) ** 2, axis=1)

    # 计算QR正交化重建误差
    # 对于正交化，重建为: X ≈ Q * R
    Q = results['QR']['result']
    # 使用完整的R矩阵（但只取前两列对应的部分）
    R_full = results['QR']['R_matrix']
    # 重建数据
    qr_reconstructed = np.dot(Q, R_full[:2, :])
    qr_error = np.mean((data - qr_reconstructed) ** 2, axis=1)

    # 可视化重建误差
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(results['QR']['result'][:, 0], results['QR']['result'][:, 1],
                c=qr_error, cmap='viridis', alpha=0.7)
    plt.colorbar(label='重建误差')
    plt.title('QR正交化重建误差')
    plt.xlabel('正交因子1')
    plt.ylabel('正交因子2')

    plt.subplot(1, 2, 2)
    sorted_indices = np.argsort(qr_error)
    plt.scatter(range(len(qr_error)), qr_error[sorted_indices], marker='o')
    plt.title('排序后的QR重建误差')
    plt.xlabel('样本索引')
    plt.ylabel('误差')
    plt.grid()

    plt.tight_layout()
    plt.savefig('reconstruction_errors.png', dpi=300)
    plt.show()

    # 识别异常点（误差最大的样本）
    anomaly_threshold = np.percentile(qr_error, 95)
    anomalies = np.where(qr_error > anomaly_threshold)[0]

    print(f"\n检测到 {len(anomalies)} 个异常样本（重建误差 > 95%分位数）:")
    print(anomalies)

    return anomalies


# 主程序
if __name__ == "__main__":
    # 文件路径 - 替换为您的实际文件路径
    file_path = r"消费50品质因子数据_huber处理_Z.csv"

    # 1. 加载和预处理数据
    data, feature_names, original_df = load_and_preprocess(file_path)

    # 2. 执行矩阵分解（包含正交化处理）
    decomposition_results = perform_decompositions(data, n_components=2)

    # 3. 打印分解结果
    print("\n=== PCA 结果 ===")
    print(f"解释方差比例: {decomposition_results['PCA']['explained_variance']}")
    print(f"累计解释方差: {sum(decomposition_results['PCA']['explained_variance']):.4f}")

    print("\n=== SVD 结果 ===")
    print(f"解释方差比例: {decomposition_results['SVD']['explained_variance']}")
    print(f"奇异值: {decomposition_results['SVD']['singular_values']}")

    print("\n=== NMF 结果 ===")
    print(f"重建误差: {decomposition_results['NMF']['reconstruction_error']:.4f}")

    print("\n=== QR 正交化结果 ===")
    print(f"正交因子形状: {decomposition_results['QR']['result'].shape}")
    print(f"R矩阵形状: {decomposition_results['QR']['R_matrix'].shape}")

    # 4. 可视化结果
    visualize_results(decomposition_results, feature_names)

    # 5. 聚类分析
    clusters = cluster_analysis(decomposition_results)

    # 将聚类结果添加到原始数据
    original_df['聚类'] = clusters
    print("\n聚类结果添加到原始数据:")
    print(original_df.head())

    # 6. 重建误差分析
    anomalies = reconstruction_error_analysis(data, decomposition_results)

    # 标记异常点
    original_df['异常点'] = 0
    original_df.loc[anomalies, '异常点'] = 1
    print("\n异常点标记:")
    print(original_df.loc[original_df['异常点'] == 1].head())

    # 保存结果
    original_df.to_csv('消费50品质因子_正交化处理结果.csv', index=False, encoding='gbk')
    print("\n结果已保存为: '消费50品质因子_正交化处理结果.csv'")

    print("\n分析完成！所有结果已保存。")
