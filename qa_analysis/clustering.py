"""
聚类分析模块

提供K-Means聚类分析功能，包括：
- 最优聚类数确定（肘部法则）
- 聚类执行
- 聚类特征分析
- 可视化生成
"""

from typing import Dict, Tuple, List
from collections import Counter
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# 常量定义
DEFAULT_VARIANCE_THRESHOLD = 0.8
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_INIT = 10
DEFAULT_TSNE_PERPLEXITY = 30

# 特征列定义
FEATURE_COLUMNS = [
    'qa_turns', 'is_multi_turn', 'total_time_minutes', 'avg_qa_time_minutes',
    'total_question_chars', 'avg_question_length',
    'if_non_class', 'is_video_unit', 'is_discussion_unit', 'is_graphic_unit', 'is_ai_task', 'is_confusion_entry',
    'avg_hours_to_assignment', 'avg_hours_since_release',
    'course_progress_ratio', 'calendar_week_since_2025_0217',
    'hours_to_next_class', 'hours_from_last_class', 'is_copy_paste', 'copy_keywords_count',
    'is_exam_week', 'day_period', 'is_weekend',
    'is_in_class_time', 'question_type_why'
]

# 二值特征列（不需要标准化）
BINARY_COLUMNS = [
    'is_multi_turn', 'if_non_class', 'is_copy_paste', 'is_video_unit', 
    'is_discussion_unit', 'is_graphic_unit', 'is_ai_task', 'is_confusion_entry',
    'is_exam_week', 'is_weekend', 'is_first_tier', 'is_in_class_time', 'question_type_why',
]

# 为了向后兼容，保留全局变量
feature_columns = FEATURE_COLUMNS
binary_cols = BINARY_COLUMNS


def get_cluster_color_map(n_clusters: int) -> Dict[int, tuple]:
    """
    生成聚类颜色映射
    
    Args:
        n_clusters: 聚类数量
    
    Returns:
        dict: 聚类ID到颜色的映射字典
    """
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    return {i: cmap(i) for i in range(n_clusters)}


def find_optimal_components_for_clustering(pca, variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD) -> Tuple[int, float]:
    """
    根据方差解释比例确定聚类所需的主成分数量
    
    Args:
        pca: 已拟合的PCA对象
        variance_threshold: 方差解释率阈值
    
    Returns:
        tuple: (所需主成分数量, 实际方差解释率)
    """
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_needed = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    if cumulative_variance[-1] < variance_threshold:
        n_components_needed = len(cumulative_variance)
        actual_variance = cumulative_variance[-1]
        print(f"⚠️ 所有主成分仅能解释 {actual_variance:.3f} ({actual_variance*100:.1f}%) 的方差")
    else:
        actual_variance = cumulative_variance[n_components_needed-1]
        print(f"✅ 前 {n_components_needed} 个主成分解释了 {actual_variance:.3f} ({actual_variance*100:.1f}%) 的方差")
    
    print(f"\n=== 主成分选择分析 ===")
    print(f"目标方差解释率: {variance_threshold*100:.0f}%")
    print(f"所需主成分数量: {n_components_needed}")
    
    print("\n各主成分贡献:")
    for i in range(min(n_components_needed + 2, len(pca.explained_variance_ratio_))):
        individual = pca.explained_variance_ratio_[i]
        cumulative = cumulative_variance[i]
        marker = "★" if i < n_components_needed else "  "
        status = "(已选)" if i < n_components_needed else "(未选)"
        print(f"{marker} PC{i+1}: {individual:.4f} | 累计: {cumulative:.4f} ({cumulative*100:.1f}%) {status}")
    
    return n_components_needed, actual_variance


def find_elbow_second_derivative(inertias: List[float], K_range: range) -> int:
    """通过二阶导数检测肘部点（检测最大曲率）"""
    if len(inertias) < 3:
        return K_range[0]
    first_diff = np.diff(inertias)
    second_diff = np.diff(first_diff)
    elbow_idx = np.argmax(second_diff) + 2
    return K_range[min(elbow_idx, len(K_range)-1)]


def find_elbow_distance_method(inertias: List[float], K_range: range) -> int:
    """通过点到直线距离法检测肘部点（测量到首尾连线的最大垂直距离）"""
    if len(inertias) < 3:
        return K_range[0]

    x = np.array(K_range)
    y = np.array(inertias)
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    start_point = np.array([x_norm[0], y_norm[0]])
    end_point = np.array([x_norm[-1], y_norm[-1]])

    distances = []
    for i in range(len(x_norm)):
        point = np.array([x_norm[i], y_norm[i]])
        line_vec = end_point - start_point
        point_vec = point - start_point
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            distance = 0
        else:
            cross_product = np.abs(np.cross(line_vec, point_vec))
            distance = cross_product / line_len
        distances.append(distance)

    elbow_idx = np.argmax(distances)
    return K_range[elbow_idx]


def find_elbow_slope_method(inertias: List[float], K_range: range) -> int:
    """通过斜率变化检测肘部点"""
    if len(inertias) < 3:
        return K_range[0]

    slopes = []
    for i in range(len(inertias) - 1):
        slope = (inertias[i+1] - inertias[i]) / (K_range[i+1] - K_range[i])
        slopes.append(abs(slope))

    slope_changes = []
    for i in range(len(slopes) - 1):
        change = abs(slopes[i+1] - slopes[i])
        slope_changes.append(change)

    if not slope_changes:
        return K_range[0]

    elbow_idx = np.argmax(slope_changes) + 1
    return K_range[min(elbow_idx, len(K_range)-1)]


def find_optimal_clusters_elbow_only(X_pca, pca, max_k=10, variance_threshold=0.8, output_folder=None):
    """
    使用肘部法则结合自适应主成分选择确定最优聚类数
    
    Args:
        X_pca: PCA降维后的数据
        pca: 已拟合的PCA对象
        max_k: 最大聚类数
        variance_threshold: 方差解释率阈值
        output_folder: 输出文件夹路径
    
    Returns:
        tuple: (最优K值, 用于聚类的特征矩阵)
    """
    # 步骤1: 确定主成分数量
    n_components, actual_variance = find_optimal_components_for_clustering(pca, variance_threshold)
    X_cluster = X_pca[:, :n_components]

    print(f"\n=== Clustering Configuration ===")
    print(f"Number of principal components used: {n_components}")
    print(f"Cluster feature space shape: {X_cluster.shape}")
    print(f"Variance retained: {actual_variance:.3f} ({actual_variance*100:.1f}%)")

    # 步骤2: 计算不同K值的惯性
    inertias = []
    K_range = range(2, min(max_k + 1, len(X_cluster)))

    print(f"\n=== Using Elbow Method (with {n_components} principal components) ===")

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=DEFAULT_RANDOM_STATE, n_init=DEFAULT_N_INIT)
        cluster_labels = kmeans.fit_predict(X_cluster)
        inertias.append(kmeans.inertia_)
        print(f"K={k}: Inertia={kmeans.inertia_:.2f}")

    # 步骤3: 应用三种肘部检测方法
    elbow_k_method1 = find_elbow_second_derivative(inertias, K_range)
    elbow_k_method2 = find_elbow_distance_method(inertias, K_range)
    elbow_k_method3 = find_elbow_slope_method(inertias, K_range)

    print(f"\n=== Elbow Detection Results ===")
    print(f"Method 1 (Second Derivative): K = {elbow_k_method1}")
    print(f"Method 2 (Distance Method): K = {elbow_k_method2}")
    print(f"Method 3 (Slope Change): K = {elbow_k_method3}")

    # 步骤4: 选择最终K值
    count = Counter([elbow_k_method1, elbow_k_method2, elbow_k_method3])
    if len(count.most_common(1)) > 0 and count.most_common(1)[0][1] > 1:
        final_k = count.most_common(1)[0][0]
        decision_reason = f"Consensus: {count.most_common(1)[0][1]}/3 methods agree"
        print(f"Final Selection (Consensus): K = {final_k}")
    else:
        final_k = int(np.median([elbow_k_method1, elbow_k_method2, elbow_k_method3]))
        decision_reason = "Median-based decision (no consensus)"
        print(f"Final Selection (Median): K = {final_k}")

    # 步骤5: 可视化
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])

    # (1) PCA方差解释图
    ax1 = fig.add_subplot(gs[0, 0])
    pc_range = range(1, len(pca.explained_variance_ratio_) + 1)
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)

    ax1.bar(pc_range, pca.explained_variance_ratio_, alpha=0.6, label='Individual')
    ax1.plot(pc_range, cumulative_var, 'ro-', linewidth=2, label='Cumulative')
    ax1.axhline(y=variance_threshold, color='red', linestyle='--', alpha=0.7,
                label=f'Target {variance_threshold*100:.0f}%')
    ax1.axvline(x=n_components, color='green', linestyle='--', alpha=0.7,
                label=f'Selected PC={n_components}')
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title(f'PCA Explained Variance\n({n_components} PCs cover {actual_variance*100:.1f}%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) 主肘部法则图
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.plot(K_range, inertias, 'bo-', linewidth=3, markersize=10, label='Inertia Curve')
    ax2.axvline(x=final_k, color='red', linestyle='--', linewidth=3, label=f'Selected K={final_k}')
    method_colors = ['orange', 'green', 'purple']
    method_names = ['Second Derivative', 'Distance Method', 'Slope Change']
    method_results = [elbow_k_method1, elbow_k_method2, elbow_k_method3]
    for method_k, color, name in zip(method_results, method_colors, method_names):
        ax2.axvline(x=method_k, color=color, linestyle=':', alpha=0.8, linewidth=2,
                    label=f'{name} (K={method_k})')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Within-Cluster Sum of Squares (Inertia)')
    ax2.set_title(f'Elbow Method Analysis\n(using {n_components} principal components)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (3) 一阶导数
    if len(inertias) > 1:
        ax3 = fig.add_subplot(gs[1, 0])
        first_diff = np.diff(inertias)
        ax3.plot(K_range[1:], first_diff, 'go-', linewidth=2, markersize=6)
        ax3.set_xlabel('Number of Clusters (K)')
        ax3.set_ylabel('First Derivative')
        ax3.set_title('Rate of Change in Inertia')
        ax3.grid(True, alpha=0.3)

    # (4) 二阶导数
    if len(inertias) > 2:
        ax4 = fig.add_subplot(gs[1, 1])
        first_diff = np.diff(inertias)
        second_diff = np.diff(first_diff)
        ax4.plot(K_range[2:], second_diff, 'ro-', linewidth=2, markersize=6)
        ax4.axvline(x=elbow_k_method1, color='red', linestyle='--', alpha=0.7,
                    label=f'Max Curvature (K={elbow_k_method1})')
        ax4.set_xlabel('Number of Clusters (K)')
        ax4.set_ylabel('Second Derivative')
        ax4.set_title('Curvature Change (2nd Derivative)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # (5) 距离法可视化
    if len(inertias) >= 3:
        ax5 = fig.add_subplot(gs[1, 2])
        x = np.array(K_range)
        y = np.array(inertias)
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())
        start_point = np.array([x_norm[0], y_norm[0]])
        end_point = np.array([x_norm[-1], y_norm[-1]])
        distances = []
        for i in range(len(x_norm)):
            point = np.array([x_norm[i], y_norm[i]])
            line_vec = end_point - start_point
            point_vec = point - start_point
            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                distance = 0
            else:
                cross_product = np.abs(np.cross(line_vec, point_vec))
                distance = cross_product / line_len
            distances.append(distance)
        ax5.plot(K_range, distances, 'mo-', linewidth=2, markersize=6)
        ax5.axvline(x=elbow_k_method2, color='magenta', linestyle='--', alpha=0.7,
                    label=f'Max Distance (K={elbow_k_method2})')
        ax5.set_xlabel('Number of Clusters (K)')
        ax5.set_ylabel('Distance to Line')
        ax5.set_title('Point-to-Line Distance (Distance Method)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # (6) 摘要文本
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    summary_text = f"""
    PCA Summary:
    • Total number of components: {len(pca.explained_variance_ratio_)}
    • Target variance threshold: {variance_threshold*100:.0f}%
    • Selected number of components: {n_components}
    • Actual explained variance: {actual_variance*100:.1f}%

    Elbow Method Results:
    • Second Derivative Method: K = {elbow_k_method1}
    • Distance Method: K = {elbow_k_method2}
    • Slope Change Method: K = {elbow_k_method3}
    • Final Selection: K = {final_k} ({decision_reason})

    Clustering Configuration:
    • Feature space: {n_components}-dimensional PCA space
    • Clustering algorithm: K-Means
    • Final number of clusters: {final_k}
    """
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))

    plt.suptitle(
        f'Adaptive PCA + Elbow Method Analysis\n'
        f'({n_components} PCs explaining {actual_variance*100:.1f}% variance → K={final_k} clusters)',
        fontsize=16, y=0.95
    )
    plt.tight_layout()

    if output_folder:
        plt.savefig(os.path.join(output_folder, 'adaptive_elbow_method_analysis.png'),
                    dpi=300, bbox_inches='tight')

    # 步骤6: 详细分析结果
    print(f"\n=== Detailed Analysis Results ===")
    print(f"PCA Configuration:")
    print(f"  - Target variance threshold: {variance_threshold*100:.0f}%")
    print(f"  - Selected components: {n_components}")
    print(f"  - Actual variance explained: {actual_variance*100:.1f}%")
    print(f"  - Information retained: {actual_variance:.3f}")
    print(f"  - Information lost: {1-actual_variance:.3f}")

    print(f"\nElbow Method Results:")
    print("Inertia values for each K:")
    for k, inertia in zip(K_range, inertias):
        print(f"  K={k}: {inertia:.2f}")

    if len(inertias) > 1:
        print("\nInertia reduction analysis:")
        first_diff = np.diff(inertias)
        for i, (k, diff) in enumerate(zip(K_range[1:], first_diff)):
            reduction_pct = abs(diff) / inertias[i] * 100
            print(f"  K={K_range[i]}→K={k}: -{abs(diff):.2f} ({reduction_pct:.1f}%)")

    print(f"\nFinal Configuration:")
    print(f"  - Principal Components: {n_components} (explaining {actual_variance*100:.1f}% variance)")
    print(f"  - Optimal Clusters: K = {final_k}")
    print(f"  - Decision Method: {decision_reason}")

    return final_k, X_cluster


def perform_clustering(X_cluster, n_clusters, features_df, output_folder, extra_k_list=(2, 3, 4)):
    """
    执行聚类分析并保存多个K值的聚类结果
    
    Args:
        X_cluster: 用于聚类的特征矩阵
        n_clusters: 主聚类数量
        features_df: 原始特征DataFrame
        output_folder: 输出文件夹路径
        extra_k_list: 额外要计算的K值列表
    
    Returns:
        tuple: (主聚类标签, 包含聚类标签的DataFrame, 主KMeans对象)
    """
    n_samples = X_cluster.shape[0]
    os.makedirs(output_folder, exist_ok=True)

    # 主聚类（最优K）
    kmeans = KMeans(n_clusters=n_clusters, random_state=DEFAULT_RANDOM_STATE, n_init=DEFAULT_N_INIT)
    cluster_labels = kmeans.fit_predict(X_cluster)

    features_df_clustered = features_df.copy()
    features_df_clustered['cluster'] = cluster_labels

    # 聚类统计
    print(f"\n=== Clustering Results Statistics (K={n_clusters}) ===")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")

    # 额外的多个K值聚类
    if extra_k_list:
        print("\n=== Additional clusterings (K in {0}) and saving to files ===".format(list(extra_k_list)))
        for k in extra_k_list:
            col_name = f'cluster_k{k}'
            if k == n_clusters:
                features_df_clustered[col_name] = cluster_labels
                out_csv = os.path.join(output_folder, f'clustered_features_k{k}.csv')
                df_to_save = features_df.copy()
                df_to_save[col_name] = cluster_labels
                df_to_save.to_csv(out_csv, index=False, encoding='utf-8-sig')
                print(f"  K={k}: reused main clustering -> saved '{out_csv}'")
                continue

            if k > n_samples:
                print(f"  K={k}: skipped (K > n_samples={n_samples})")
                continue

            try:
                kmeans_k = KMeans(n_clusters=k, random_state=DEFAULT_RANDOM_STATE, n_init=DEFAULT_N_INIT)
                labels_k = kmeans_k.fit_predict(X_cluster)
                features_df_clustered[col_name] = labels_k

                counts_k = pd.Series(labels_k).value_counts().sort_index()
                dist_str = ", ".join([f"{cid}:{cnt}" for cid, cnt in counts_k.items()])
                print(f"  K={k}: added column '{col_name}' | counts -> {dist_str}")

                out_csv = os.path.join(output_folder, f'clustered_features_k{k}.csv')
                df_to_save = features_df.copy()
                df_to_save[col_name] = labels_k
                df_to_save.to_csv(out_csv, index=False, encoding='utf-8-sig')
                print(f"  K={k}: saved to '{out_csv}'")
            except Exception as e:
                print(f"  K={k}: failed -> {e}")

    return cluster_labels, features_df_clustered, kmeans


def analyze_cluster_characteristics(features_df_clustered, feature_columns, output_folder):
    """
    分析并可视化各聚类的特征
    
    Args:
        features_df_clustered: 包含聚类标签的DataFrame
        feature_columns: 特征列名列表
        output_folder: 输出文件夹路径
    
    Returns:
        pd.DataFrame: 各聚类的特征均值
    """
    print(f"\n=== Cluster Characteristics Analysis ===")

    cluster_means = features_df_clustered.groupby('cluster')[feature_columns].mean()
    print("\nMean values for each cluster:")
    print(cluster_means.round(3))

    cluster_stats = features_df_clustered.groupby('cluster')[feature_columns].agg(['mean', 'std', 'median'])
    cluster_stats.to_csv(os.path.join(output_folder, 'cluster_statistics.csv'), encoding='utf-8-sig')
    print(f"✅ Saved detailed cluster statistics to: {output_folder}/cluster_statistics.csv")

    # 为每个特征创建单独的柱状图
    for feature in feature_columns:
        plt.figure(figsize=(8, 6))
        cluster_means[feature].plot(kind='bar', color='skyblue', alpha=0.7, edgecolor='black')

        plt.title(f'{feature} by Cluster')
        plt.xlabel('Cluster ID')
        plt.ylabel(f'Mean {feature}')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=0)

        safe_feature_name = feature.replace('/', '_').replace(' ', '_')
        filename = f'cluster_characteristic_{safe_feature_name}.png'
        filepath = os.path.join(output_folder, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✅ Saved {len(feature_columns)} individual feature plots to: {output_folder}")

    return cluster_means


def comprehensive_clustering_visualization_all_k(X_scaled, X_pca, features_df_clustered, 
                                                 feature_columns, cluster_cols_list, 
                                                 output_folder):
    """
    为多个聚类结果生成综合可视化
    
    Args:
        X_scaled: 标准化后的特征矩阵
        X_pca: PCA降维后的数据
        features_df_clustered: 包含聚类标签的DataFrame
        feature_columns: 特征列名列表
        cluster_cols_list: 聚类列名列表
        output_folder: 输出文件夹路径
    """
    os.makedirs(output_folder, exist_ok=True)

    for cluster_col in cluster_cols_list:
        cluster_labels = features_df_clustered[cluster_col].values
        n_clusters = len(np.unique(cluster_labels))
        colors_dict = get_cluster_color_map(n_clusters)
        colors = [colors_dict[label] for label in cluster_labels]

        print(f"Generating visualizations for {cluster_col} (K={n_clusters})...")

        # === 1. PCA可视化 ===
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors,
                   s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.title(f'PCA Clustering Visualization ({cluster_col})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # 添加质心
        for i in range(n_clusters):
            cluster_points = X_pca[cluster_labels == i]
            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                plt.scatter(centroid[0], centroid[1], c='red', s=200, marker='x', linewidth=3)
                plt.annotate(f'C{i}', (centroid[0], centroid[1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'pca_{cluster_col}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # === 2. t-SNE可视化 ===
        plt.figure(figsize=(8, 6))
        tsne = TSNE(n_components=2, random_state=DEFAULT_RANDOM_STATE, 
                   perplexity=min(DEFAULT_TSNE_PERPLEXITY, len(X_scaled)-1))
        X_tsne = tsne.fit_transform(X_scaled)
        colors_tsne = [colors_dict[label] for label in cluster_labels]
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors_tsne,
                   s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.title(f't-SNE Clustering Visualization ({cluster_col})')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'tsne_{cluster_col}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # === 3. 聚类大小分布 ===
        plt.figure(figsize=(8, 6))
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        bars = plt.bar(range(n_clusters), cluster_counts.values,
                      color=[colors_dict[i] for i in range(n_clusters)],
                      alpha=0.7, edgecolor='black')
        plt.title(f'Cluster Size Distribution ({cluster_col})')
        plt.xlabel('Cluster Label')
        plt.ylabel('Number of Samples')
        plt.xticks(range(n_clusters))
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'cluster_size_{cluster_col}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # === 4. QA轮次分布 ===
        plt.figure(figsize=(8, 6))
        qa_data, qa_labels, qa_stats = [], [], []
        for i in range(n_clusters):
            cluster_qa = features_df_clustered[features_df_clustered[cluster_col]==i]['qa_turns']
            if len(cluster_qa) > 0:
                orig_mean = cluster_qa.mean()
                orig_std = cluster_qa.std()
                qa_min, qa_max = cluster_qa.min(), cluster_qa.max()
                if qa_max > qa_min:
                    cluster_qa_norm = (cluster_qa - qa_min) / (qa_max - qa_min)
                else:
                    cluster_qa_norm = pd.Series(0.5, index=cluster_qa.index)
                qa_data.append(cluster_qa_norm)
                qa_labels.append(f'C{i}')
                qa_stats.append((orig_mean, orig_std))

        if qa_data:
            box_plot = plt.boxplot(qa_data, labels=qa_labels, patch_artist=True)
            for patch, i in zip(box_plot['boxes'], range(len(qa_data))):
                patch.set_facecolor(colors_dict[i])
                patch.set_alpha(0.7)
            for i, (mean_val, std_val) in enumerate(qa_stats, start=1):
                plt.text(i, 1.05, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.title(f'Normalized QA Turns Distribution ({cluster_col})')
        plt.xlabel('Cluster Label')
        plt.ylabel('Normalized QA Turns (0–1)')
        plt.ylim(-0.05, 1.15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'qa_turns_{cluster_col}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # === 5. 聚类特征热力图 ===
        out_file = os.path.join(output_folder, f'feature_heatmap_{cluster_col}.png')
        plot_cluster_feature_heatmap(features_df_clustered, feature_columns, cluster_col, out_file)

        # === 6. 课程进度分布 ===
        plt.figure(figsize=(8, 6))
        for i in range(n_clusters):
            cluster_weeks = features_df_clustered[features_df_clustered[cluster_col]==i]['course_progress_ratio']
            cluster_weeks = cluster_weeks[cluster_weeks>0]
            if len(cluster_weeks)>0:
                plt.hist(cluster_weeks, bins=20, alpha=0.7, 
                       label=f'C{i}', color=colors_dict[i])
        plt.legend()
        plt.title(f'Course Progress Distribution ({cluster_col})')
        plt.xlabel('Course Progress Ratio')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'course_progress_{cluster_col}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✅ Saved all clustering visualizations for columns: {cluster_cols_list} in {output_folder}")


def plot_cluster_feature_heatmap(features_df_clustered, feature_columns, cluster_col, output_file):
    """
    绘制聚类特征热力图
    
    Args:
        features_df_clustered: 包含聚类标签的DataFrame
        feature_columns: 特征列名列表
        cluster_col: 聚类标签列名
        output_file: 输出文件路径
    """
    cluster_means = features_df_clustered.groupby(cluster_col)[feature_columns].mean()
    n_clusters = cluster_means.shape[0]

    cluster_means_norm = cluster_means.copy()

    # 只标准化非二值特征
    numeric_cols = [col for col in feature_columns if col not in BINARY_COLUMNS]
    if numeric_cols:
        cluster_means_norm[numeric_cols] = (
            cluster_means[numeric_cols] - cluster_means[numeric_cols].mean()
        ) / cluster_means[numeric_cols].std()

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    im = plt.imshow(cluster_means_norm.T, cmap='RdBu_r', aspect='auto')
    plt.title(f'Cluster Feature Heatmap ({cluster_col})')
    plt.xlabel('Cluster Label')
    plt.ylabel('Features')
    plt.xticks(range(n_clusters), [f'C{i}' for i in range(n_clusters)])
    plt.yticks(range(len(feature_columns)), feature_columns, fontsize=8)
    plt.colorbar(im, label='Normalized Value')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
