import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


# 1. 数据加载和预处理
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path,
                     header=1,
                     names=['净利润增长率(%)', '净资产增长率(%)', '主营业务收入增长率(%)', '净资产报酬率(%)',
                            '资产报酬率(%)'],
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

    return scaled_data, df.columns.tolist(), df  # 返回原始DataFrame


# 2. 矩阵分解方法
def perform_decompositions(data, n_components=2):
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
    # 将数据转换为非负
    non_neg_data = data - np.min(data) + 1e-6
    nmf = NMF(n_components=n_components, init='random', random_state=42)
    nmf_result = nmf.fit_transform(non_neg_data)
    results['NMF'] = {
        'components': nmf.components_,
        'reconstruction_error': nmf.reconstruction_err_,
        'result': nmf_result
    }

    return results


# 3. 结果可视化
def visualize_results(results, feature_names):
    plt.figure(figsize=(18, 12))

    # PCA可视化
    plt.subplot(2, 3, 1)
    pca_result = results['PCA']['result']
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
    plt.title('PCA Projection')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    # PCA载荷图
    plt.subplot(2, 3, 4)
    components = results['PCA']['components']
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, components[0, i], components[1, i],
                  head_width=0.05, head_length=0.05, fc='k', ec='k')
        plt.text(components[0, i] * 1.15, components[1, i] * 1.15, feature,
                 fontsize=9, ha='center', va='center')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid()
    plt.title('PCA Component Loadings')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    # SVD可视化
    plt.subplot(2, 3, 2)
    svd_result = results['SVD']['result']
    plt.scatter(svd_result[:, 0], svd_result[:, 1], alpha=0.6, c='green')
    plt.title('SVD Projection')
    plt.xlabel('SV1')
    plt.ylabel('SV2')

    # NMF可视化
    plt.subplot(2, 3, 3)
    nmf_result = results['NMF']['result']
    plt.scatter(nmf_result[:, 0], nmf_result[:, 1], alpha=0.6, c='purple')
    plt.title('NMF Projection')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # 方差解释率
    plt.subplot(2, 3, 5)
    methods = ['PCA', 'SVD']
    for method in methods:
        explained_variance = results[method]['explained_variance']
        plt.plot(np.cumsum(explained_variance), 'o-', label=method)
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    plt.grid()

    # t-SNE可视化
    plt.subplot(2, 3, 6)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(data)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6, c='orange')
    plt.title('t-SNE Projection')

    plt.tight_layout()
    plt.savefig('matrix_decomposition_results.png', dpi=300)
    plt.show()


# 4. 聚类分析
def cluster_analysis(results):
    # 使用PCA结果进行聚类
    pca_result = results['PCA']['result']

    # 使用肘部法则确定最佳聚类数
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(pca_result)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, 'o-')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS (Within-Cluster Sum of Square)')
    plt.grid()
    plt.savefig('elbow_method.png', dpi=300)
    plt.show()

    # 根据肘部法则选择聚类数（这里假设为3）
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(pca_result)

    # 可视化聚类结果
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=300, c='red', marker='X', label='Centroids')
    plt.title(f'K-means Clustering (k={n_clusters}) on PCA Components')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.colorbar(label='Cluster')
    plt.savefig('pca_clustering.png', dpi=300)
    plt.show()

    return clusters


# 5. 重建误差分析
def reconstruction_error_analysis(data, results):
    # 计算PCA重建误差
    pca = PCA(n_components=2)
    pca.fit(data)
    pca_reconstructed = pca.inverse_transform(pca.transform(data))
    pca_error = np.mean((data - pca_reconstructed) ** 2, axis=1)

    # 计算SVD重建误差
    svd = TruncatedSVD(n_components=2)
    svd.fit(data)
    svd_reconstructed = svd.inverse_transform(svd.transform(data))
    svd_error = np.mean((data - svd_reconstructed) ** 2, axis=1)

    # 可视化重建误差
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(results['PCA']['result'][:, 0], results['PCA']['result'][:, 1],
                c=pca_error, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Reconstruction Error')
    plt.title('PCA Reconstruction Error')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.subplot(1, 2, 2)
    plt.scatter(range(len(pca_error)), np.sort(pca_error), marker='o')
    plt.title('Sorted PCA Reconstruction Errors')
    plt.xlabel('Sample Index')
    plt.ylabel('Error')
    plt.grid()

    plt.tight_layout()
    plt.savefig('reconstruction_errors.png', dpi=300)
    plt.show()

    # 识别异常点（误差最大的样本）
    anomaly_threshold = np.percentile(pca_error, 95)
    anomalies = np.where(pca_error > anomaly_threshold)[0]

    print(f"\n检测到 {len(anomalies)} 个异常样本（重建误差 > 95%分位数）:")
    print(anomalies)

    return anomalies


# 主程序
if __name__ == "__main__":
    # 文件路径 - 替换为您的实际文件路径
    file_path = r"C:\Users\86133\Desktop\各种资料\python\量化\使用HUBER M离群化处理的因子标准化\消费50成长因子（补齐）_huber处理_Z.csv"

    # 1. 加载和预处理数据
    data, feature_names, original_df = load_and_preprocess(file_path)

    # 2. 执行矩阵分解
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

    # 4. 可视化结果
    visualize_results(decomposition_results, feature_names)

    # 5. 聚类分析
    clusters = cluster_analysis(decomposition_results)

    # 6. 重建误差分析
    anomalies = reconstruction_error_analysis(data, decomposition_results)

    # 7. 保存中性化处理后的数据（新增部分）
    # 创建包含PCA成分的DataFrame
    pca_df = pd.DataFrame(decomposition_results['PCA']['result'],
                          columns=[f'PC{i + 1}' for i in range(decomposition_results['PCA']['result'].shape[1])])

    # 添加聚类结果
    pca_df['Cluster'] = clusters

    # 添加异常标记
    pca_df['Anomaly'] = 0
    pca_df.loc[anomalies, 'Anomaly'] = 1

    # 合并原始数据和处理结果
    result_df = pd.concat([original_df, pca_df], axis=1)

    # 保存结果
    output_path = file_path.replace('.csv', '_neutralized.csv')
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n中性化处理后的数据已保存至: {output_path}")

    print("\n分析完成！结果已保存为图像文件和CSV文件。")