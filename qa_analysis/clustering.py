import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os, seaborn as sns
from sklearn.manifold import TSNE

def get_cluster_color_map(n_clusters):
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    return {i: cmap(i) for i in range(n_clusters)}

# Select numeric features
feature_columns = [
        'qa_turns', 'is_multi_turn', 'total_time_minutes', 'avg_qa_time_minutes',
        'total_question_chars', 'avg_question_length',
        'if_non_class', 'avg_hours_to_assignment', 'avg_hours_since_release',
        'course_progress_ratio', 'calendar_week_since_2025_0217',
        'hours_to_next_class', 'hours_from_last_class', 'has_copy_keywords', 'copy_keywords_count',
        'is_exam_week','day_period','is_weekend',
        'is_in_class_time','question_type_why_how'
    ]
    
    # Binary columns (not standardized)
binary_cols = [
        'is_multi_turn', 'if_non_class', 'has_copy_keywords',
        'is_exam_week', 'is_weekend', 'is_first_tier', 'is_in_class_time', 'question_type_why_how',
    ]

def find_optimal_components_for_clustering(pca, variance_threshold=0.8):
    """
    根据方差解释比例确定聚类所需的主成分数量
    """
    # 计算累计解释方差比
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # 找到达到阈值的最少主成分数
    n_components_needed = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # 如果没有达到阈值，使用所有主成分
    if cumulative_variance[-1] < variance_threshold:
        n_components_needed = len(cumulative_variance)
        actual_variance = cumulative_variance[-1]
        print(f"⚠️ 所有主成分仅能解释 {actual_variance:.3f} ({actual_variance*100:.1f}%) 的方差")
    else:
        actual_variance = cumulative_variance[n_components_needed-1]
        print(f"✅ 前 {n_components_needed} 个主成分解释了 {actual_variance:.3f} ({actual_variance*100:.1f}%) 的方差")
    
    # 显示详细分析
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

def find_optimal_clusters_elbow_only(X_pca, pca, max_k=10, variance_threshold=0.8, output_folder=None):
    """
    Use the elbow method with adaptive principal component selection to determine the optimal number of clusters.
    """

    # Step 1: Determine the number of principal components based on explained variance ratio
    n_components, actual_variance = find_optimal_components_for_clustering(pca, variance_threshold)
    X_cluster = X_pca[:, :n_components]

    print(f"\n=== Clustering Configuration ===")
    print(f"Number of principal components used: {n_components}")
    print(f"Cluster feature space shape: {X_cluster.shape}")
    print(f"Variance retained: {actual_variance:.3f} ({actual_variance*100:.1f}%)")

    inertias = []
    K_range = range(2, min(max_k + 1, len(X_cluster)))

    print(f"\n=== Using Elbow Method (with {n_components} principal components) ===")

    # Compute inertia (within-cluster sum of squares) for different K values
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_cluster)
        inertias.append(kmeans.inertia_)
        print(f"K={k}: Inertia={kmeans.inertia_:.2f}")

    # ---------------------------------------------------------------------
    # Elbow point detection methods
    # ---------------------------------------------------------------------
    def find_elbow_second_derivative(inertias, K_range):
        """Find the elbow by detecting the largest curvature via the second derivative."""
        if len(inertias) < 3:
            return K_range[0]
        first_diff = np.diff(inertias)
        second_diff = np.diff(first_diff)
        elbow_idx = np.argmax(second_diff) + 2
        return K_range[min(elbow_idx, len(K_range)-1)]

    def find_elbow_distance_method(inertias, K_range):
        """Find the elbow by measuring the maximum perpendicular distance to the line connecting first and last points."""
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

    def find_elbow_slope_method(inertias, K_range):
        """Find the elbow by detecting the largest change in slope."""
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

    # Apply all three elbow detection methods
    elbow_k_method1 = find_elbow_second_derivative(inertias, K_range)
    elbow_k_method2 = find_elbow_distance_method(inertias, K_range)
    elbow_k_method3 = find_elbow_slope_method(inertias, K_range)

    print(f"\n=== Elbow Detection Results ===")
    print(f"Method 1 (Second Derivative): K = {elbow_k_method1}")
    print(f"Method 2 (Distance Method): K = {elbow_k_method2}")
    print(f"Method 3 (Slope Change): K = {elbow_k_method3}")

    # Select final K value (based on majority vote or median)
    from collections import Counter
    count = Counter([elbow_k_method1, elbow_k_method2, elbow_k_method3])
    if len(count.most_common(1)) > 0 and count.most_common(1)[0][1] > 1:
        final_k = count.most_common(1)[0][0]
        decision_reason = f"Consensus: {count.most_common(1)[0][1]}/3 methods agree"
        print(f"Final Selection (Consensus): K = {final_k}")
    else:
        final_k = int(np.median([elbow_k_method1, elbow_k_method2, elbow_k_method3]))
        decision_reason = "Median-based decision (no consensus)"
        print(f"Final Selection (Median): K = {final_k}")

    # ---------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])

    # (1) PCA explained variance plot
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

    # (2) Main Elbow Method Plot
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

    # (3) First Derivative
    if len(inertias) > 1:
        ax3 = fig.add_subplot(gs[1, 0])
        first_diff = np.diff(inertias)
        ax3.plot(K_range[1:], first_diff, 'go-', linewidth=2, markersize=6)
        ax3.set_xlabel('Number of Clusters (K)')
        ax3.set_ylabel('First Derivative')
        ax3.set_title('Rate of Change in Inertia')
        ax3.grid(True, alpha=0.3)

    # (4) Second Derivative
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

    # (5) Distance Method Visualization
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

    # (6) Summary Text
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

    # Detailed text summary in console
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
    """执行聚类分析并就地保存K=2/3/4的聚类结果到文件
    - 主聚类：使用 n_clusters（来自肘部法或其他选择），列名为 'cluster'
    - 额外聚类：默认再计算 K=2、3、4 的聚类，列名为 'cluster_k2'、'cluster_k3'、'cluster_k4'
    - 将每个额外K的结果保存为: clustered_features_k{k}.csv
    - 返回值保持不变：cluster_labels(最优K)、features_df_clustered、kmeans(最优K)
    """
    n_samples = X_cluster.shape[0]
    os.makedirs(output_folder, exist_ok=True)

    # 主聚类（最优K）
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster)

    # 添加到原始数据副本
    features_df_clustered = features_df.copy()
    features_df_clustered['cluster'] = cluster_labels

    # 聚类统计
    print(f"\n=== Clustering Results Statistics (K={n_clusters}) ===")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")

    # 额外的多个K（默认2/3/4）并就地保存
    if extra_k_list:
        print("\n=== Additional clusterings (K in {0}) and saving to files ===".format(list(extra_k_list)))
        for k in extra_k_list:
            col_name = f'cluster_k{k}'
            if k == n_clusters:
                # 与主聚类相同的K，直接复用
                features_df_clustered[col_name] = cluster_labels
                # 同时保存一个专门的文件，便于对比
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
                kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_k = kmeans_k.fit_predict(X_cluster)
                features_df_clustered[col_name] = labels_k

                # 简要统计并保存
                counts_k = pd.Series(labels_k).value_counts().sort_index()
                dist_str = ", ".join([f"{cid}:{cnt}" for cid, cnt in counts_k.items()])
                print(f"  K={k}: added column '{col_name}' | counts -> {dist_str}")

                # 保存该K的CSV（就地保存）
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
    Analyze and visualize feature characteristics for each cluster.
    - Computes and saves descriptive statistics.
    - Creates and saves individual bar plots (one per feature) showing cluster-wise means.
    """

    print(f"\n=== Cluster Characteristics Analysis ===")

    # Compute mean feature values for each cluster
    cluster_means = features_df_clustered.groupby('cluster')[feature_columns].mean()
    print("\nMean values for each cluster:")
    print(cluster_means.round(3))

    # Save detailed statistics (mean, std, median)
    cluster_stats = features_df_clustered.groupby('cluster')[feature_columns].agg(['mean', 'std', 'median'])
    cluster_stats.to_csv(os.path.join(output_folder, 'cluster_statistics.csv'), encoding='utf-8-sig')
    print(f"✅ Saved detailed cluster statistics to: {output_folder}/cluster_statistics.csv")

    # Number of clusters
    n_clusters = len(cluster_means)

    # Create and save individual plots for each feature
    for feature in feature_columns:
        plt.figure(figsize=(8, 6))
        cluster_means[feature].plot(kind='bar', color='skyblue', alpha=0.7, edgecolor='black')

        plt.title(f'{feature} by Cluster')
        plt.xlabel('Cluster ID')
        plt.ylabel(f'Mean {feature}')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=0)

        # Save each subplot as a separate image
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
    Comprehensive Clustering Visualization for multiple clusterings (main + K=2/3/4)
    - Ensures consistent cluster colors across PCA, t-SNE, distribution plots.
    """
    os.makedirs(output_folder, exist_ok=True)

    for cluster_col in cluster_cols_list:
        cluster_labels = features_df_clustered[cluster_col].values
        n_clusters = len(np.unique(cluster_labels))
        colors_dict = get_cluster_color_map(n_clusters)
        colors = [colors_dict[label] for label in cluster_labels]

        print(f"Generating visualizations for {cluster_col} (K={n_clusters})...")

        # === 1. PCA Visualization ===
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors,
                              s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.title(f'PCA Clustering Visualization ({cluster_col})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # Add centroids
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

        # === 2. t-SNE Visualization ===
        plt.figure(figsize=(8, 6))
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
        X_tsne = tsne.fit_transform(X_scaled)
        colors_tsne = [colors_dict[label] for label in cluster_labels]
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors_tsne,
                              s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.title(f't-SNE Clustering Visualization ({cluster_col})')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'tsne_{cluster_col}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # === 3. Cluster Size Distribution ===
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

        # === 4. QA Turns Distribution ===
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

        # === 5. Cluster Feature Heatmap ===
        out_file = os.path.join(output_folder, f'feature_heatmap_{cluster_col}.png')
        plot_cluster_feature_heatmap(features_df_clustered, feature_columns, cluster_col, out_file)

        # === 6. Course Progress Distribution ===
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
    Plot cluster feature heatmap for a specific clustering column.
    - features_df_clustered: dataframe with features and cluster labels
    - feature_columns: list of feature names to include
    - cluster_col: column name that contains cluster labels (e.g., 'cluster', 'cluster_k2')
    - output_file: path to save the heatmap
    """
    # 使用全局 binary_cols
    global binary_cols

    # 计算各 cluster 的均值
    cluster_means = features_df_clustered.groupby(cluster_col)[feature_columns].mean()
    n_clusters = cluster_means.shape[0]

    # 拷贝一份以免修改原数据
    cluster_means_norm = cluster_means.copy()

    # 只标准化非 binary 特征
    numeric_cols = [col for col in feature_columns if col not in binary_cols]
    cluster_means_norm[numeric_cols] = (
        cluster_means[numeric_cols] - cluster_means[numeric_cols].mean()
    ) / cluster_means[numeric_cols].std()

    # binary 特征保持原值（0~1）
    # 画热力图
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
