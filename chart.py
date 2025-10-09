import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_cluster_feature_differences(clustered_features_file, output_folder=None):
    """绘制14个特征在不同聚类间的平均值差异"""
    
    # 读取聚类结果数据
    df = pd.read_csv(clustered_features_file, encoding='utf-8-sig')
    
    # 定义14个特征
    feature_columns = [
        'qa_turns', 'total_time_minutes', 'avg_qa_time_minutes',
        'total_question_chars', 'avg_question_length',
        'is_courseware_entry', 'is_discussion_entry', 'is_ai_task_entry', 'is_exercise_entry',
        'non_class_ratio', 'avg_days_to_assignment', 'avg_teaching_week',
        'hours_to_next_class', 'hours_from_last_class'
    ]
    
    # 检查聚类列是否存在
    if 'cluster' not in df.columns:
        print("错误：数据中没有'cluster'列")
        return
    
    # 获取聚类数量
    n_clusters = df['cluster'].nunique()
    clusters = sorted(df['cluster'].unique())
    
    print(f"发现 {n_clusters} 个聚类: {clusters}")
    print(f"每个聚类的样本数: {df['cluster'].value_counts().sort_index().values}")
    
    # 计算每个聚类的特征均值
    cluster_means = df.groupby('cluster')[feature_columns].mean()
    
    # 创建14个子图（4行4列，最后2个空）
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    # fig.suptitle(f'Feature Differences Across {n_clusters} Clusters', fontsize=16, y=0.95)
    
    # 扁平化axes数组便于索引
    axes_flat = axes.flatten()
    
    # 为每个聚类定义颜色
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    cluster_colors = {cluster: colors[i] for i, cluster in enumerate(clusters)}
    
    # 为每个特征绘制子图
    for i, feature in enumerate(feature_columns):
        ax = axes_flat[i]
        
        # 准备数据
        feature_means = [cluster_means.loc[cluster, feature] for cluster in clusters]
        x_positions = range(len(clusters))
        
        # 绘制柱状图
        bars = ax.bar(x_positions, feature_means, 
                     color=[cluster_colors[c] for c in clusters],
                     alpha=0.7, 
                     edgecolor='black', 
                     linewidth=0.5)
        
        # 在柱子上添加数值标签
        for j, (bar, value) in enumerate(zip(bars, feature_means)):
            height = bar.get_height()
            # 根据数值大小调整标签格式
            if abs(value) < 0.01:
                label_text = f'{value:.3f}'
            elif abs(value) < 1:
                label_text = f'{value:.2f}'
            elif abs(value) < 100:
                label_text = f'{value:.1f}'
            else:
                label_text = f'{value:.0f}'
            
            ax.text(bar.get_x() + bar.get_width()/2., height + max(feature_means)*0.02,
                   label_text, ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # 设置子图样式
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Cluster', fontsize=10)
        ax.set_ylabel('Mean Value', fontsize=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'C{c}' for c in clusters])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 设置y轴范围，确保标签不被切掉
        y_max = max(feature_means) * 1.15
        y_min = min(feature_means) - abs(max(feature_means)) * 0.05
        ax.set_ylim(y_min, y_max)
        
        # 如果是二元特征，特殊处理
        binary_features = ['is_courseware_entry', 'is_discussion_entry', 
                          'is_ai_task_entry', 'is_exercise_entry']
        if feature in binary_features:
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(['0%', '50%', '100%'])
        
        # 如果是比例特征，特殊处理
        if feature == 'non_class_ratio':
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # 隐藏最后两个空子图
    for i in range(len(feature_columns), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # 在最后一个空子图位置添加图例和统计信息
    legend_ax = axes_flat[-1]
    legend_ax.set_visible(True)
    legend_ax.axis('off')
    
    # 创建聚类图例
    legend_elements = []
    for cluster in clusters:
        from matplotlib.patches import Patch
        legend_elements.append(Patch(facecolor=cluster_colors[cluster], 
                                   edgecolor='black', 
                                   label=f'Cluster {cluster}'))
    
    legend_ax.legend(handles=legend_elements, loc='center', fontsize=12, title='Clusters')
    
    # 添加统计信息
    stats_text = f"Total Samples: {len(df)}\n"
    for cluster in clusters:
        count = len(df[df['cluster'] == cluster])
        percentage = count / len(df) * 100
        stats_text += f"Cluster {cluster}: {count} ({percentage:.1f}%)\n"
    
    legend_ax.text(0.5, 0.3, stats_text, transform=legend_ax.transAxes,
                  ha='center', va='center', fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # 为总标题留出空间
    
    # 保存图片
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'cluster_feature_differences.png'), 
                    dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return cluster_means

def plot_cluster_feature_heatmap(clustered_features_file, output_folder=None):
    """绘制聚类特征热力图（补充可视化）"""
    
    df = pd.read_csv(clustered_features_file, encoding='utf-8-sig')
    
    feature_columns = [
        'qa_turns', 'total_time_minutes', 'avg_qa_time_minutes',
        'total_question_chars', 'avg_question_length',
        'is_courseware_entry', 'is_discussion_entry', 'is_ai_task_entry', 'is_exercise_entry',
        'non_class_ratio', 'avg_days_to_assignment', 'avg_teaching_week',
        'hours_to_next_class', 'hours_from_last_class'
    ]
    
    # 计算聚类均值
    cluster_means = df.groupby('cluster')[feature_columns].mean()
    
    # 标准化均值用于热力图显示
    cluster_means_normalized = (cluster_means - cluster_means.mean()) / cluster_means.std()
    
    # 绘制热力图
    plt.figure(figsize=(12, 8))
    
    heatmap = sns.heatmap(
        cluster_means_normalized.T,  # 转置：特征为行，聚类为列
        annot=True,                  # 显示数值
        cmap='RdBu_r',              # 红蓝色图
        center=0,                   # 0为白色
        fmt='.2f',                  # 数值格式
        cbar_kws={'label': 'Standardized Mean Value'},
        xticklabels=[f'Cluster {c}' for c in cluster_means.index],
        yticklabels=feature_columns
    )
    
    plt.title('Cluster Feature Profile Heatmap\n(Standardized Mean Values)', fontsize=14, pad=20)
    plt.xlabel('Clusters', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # 旋转y轴标签以便阅读
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'cluster_feature_heatmap.png'), 
                    dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_cluster_differences(clustered_features_file):
    """分析聚类间的特征差异"""
    
    df = pd.read_csv(clustered_features_file, encoding='utf-8-sig')
    
    feature_columns = [
        'qa_turns', 'total_time_minutes', 'avg_qa_time_minutes',
        'total_question_chars', 'avg_question_length',
        'is_courseware_entry', 'is_discussion_entry', 'is_ai_task_entry', 'is_exercise_entry',
        'non_class_ratio', 'avg_days_to_assignment', 'avg_teaching_week',
        'hours_to_next_class', 'hours_from_last_class'
    ]
    
    cluster_means = df.groupby('cluster')[feature_columns].mean()
    
    print("=== 聚类特征差异分析 ===")
    
    # 为每个聚类提供描述性标签
    cluster_descriptions = {}
    
    for cluster_id in cluster_means.index:
        cluster_data = cluster_means.loc[cluster_id]
        
        # 找出该聚类的突出特征（高于平均值1个标准差以上）
        overall_means = df[feature_columns].mean()
        overall_stds = df[feature_columns].std()
        
        high_features = []
        low_features = []
        
        for feature in feature_columns:
            z_score = (cluster_data[feature] - overall_means[feature]) / overall_stds[feature]
            
            if z_score > 1:  # 明显高于平均
                high_features.append((feature, z_score, cluster_data[feature]))
            elif z_score < -1:  # 明显低于平均
                low_features.append((feature, z_score, cluster_data[feature]))
        
        # 排序
        high_features.sort(key=lambda x: x[1], reverse=True)
        low_features.sort(key=lambda x: x[1])
        
        print(f"\n聚类 {cluster_id} 特征画像：")
        print(f"样本数量: {len(df[df['cluster'] == cluster_id])}")
        
        if high_features:
            print("突出特征（高于平均）：")
            for feature, z_score, mean_val in high_features[:3]:  # 显示前3个
                print(f"  ↑ {feature}: {mean_val:.2f} (Z={z_score:.2f})")
        
        if low_features:
            print("显著特征（低于平均）：")
            for feature, z_score, mean_val in low_features[:3]:  # 显示前3个
                print(f"  ↓ {feature}: {mean_val:.2f} (Z={z_score:.2f})")
        
        # 生成聚类描述
        cluster_descriptions[cluster_id] = generate_cluster_description(
            high_features, low_features)
        print(f"聚类描述: {cluster_descriptions[cluster_id]}")
    
    return cluster_descriptions

def generate_cluster_description(high_features, low_features):
    """基于特征生成聚类描述"""
    
    # 特征类别定义
    engagement_features = ['qa_turns', 'total_time_minutes', 'total_question_chars']
    quality_features = ['avg_question_length', 'avg_qa_time_minutes']
    autonomy_features = ['non_class_ratio', 'is_courseware_entry', 'is_discussion_entry']
    timing_features = ['hours_to_next_class', 'hours_from_last_class', 'avg_teaching_week']
    
    # 分析特征模式
    high_feature_names = [f[0] for f in high_features]
    low_feature_names = [f[0] for f in low_features]
    
    description_parts = []
    
    # 检查投入度
    if any(f in high_feature_names for f in engagement_features):
        description_parts.append("高投入度")
    elif any(f in low_feature_names for f in engagement_features):
        description_parts.append("低投入度")
    
    # 检查学习质量
    if any(f in high_feature_names for f in quality_features):
        description_parts.append("深度学习")
    elif any(f in low_feature_names for f in quality_features):
        description_parts.append("快速学习")
    
    # 检查自主性
    if any(f in high_feature_names for f in autonomy_features):
        description_parts.append("自主导向")
    elif any(f in low_feature_names for f in autonomy_features):
        description_parts.append("课堂导向")
    
    # 检查时间模式
    if 'hours_to_next_class' in low_feature_names or 'hours_from_last_class' in low_feature_names:
        description_parts.append("课程同步")
    elif 'hours_to_next_class' in high_feature_names or 'hours_from_last_class' in high_feature_names:
        description_parts.append("时间灵活")
    
    return " + ".join(description_parts) if description_parts else "平衡型"

def plot_cluster_comparison_radar(clustered_features_file, output_folder=None):
    """绘制聚类特征雷达图比较"""
    
    df = pd.read_csv(clustered_features_file, encoding='utf-8-sig')
    
    feature_columns = [
        'qa_turns', 'total_time_minutes', 'avg_qa_time_minutes',
        'total_question_chars', 'avg_question_length',
        'is_courseware_entry', 'is_discussion_entry', 'is_ai_task_entry', 'is_exercise_entry',
        'non_class_ratio', 'avg_days_to_assignment', 'avg_teaching_week',
        'hours_to_next_class', 'hours_from_last_class'
    ]
    
    # 计算聚类均值并标准化到[0,1]
    cluster_means = df.groupby('cluster')[feature_columns].mean()
    
    # 标准化到[0,1]范围用于雷达图
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        columns=feature_columns,
        index=cluster_means.index
    )
    
    # 设置雷达图
    angles = np.linspace(0, 2 * np.pi, len(feature_columns), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合圆形
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # 为每个聚类绘制雷达图
    colors = plt.cm.Set1(np.linspace(0, 1, len(cluster_means_scaled)))
    
    for i, (cluster_id, row) in enumerate(cluster_means_scaled.iterrows()):
        values = row.values
        values = np.concatenate((values, [values[0]]))  # 闭合图形
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=f'Cluster {cluster_id}', color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace('_', '\n') for f in feature_columns], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)
    
    plt.title('Cluster Feature Profile Comparison\n(Radar Chart)', 
              fontsize=14, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'cluster_radar_comparison.png'), 
                    dpi=300, bbox_inches='tight')
    
    plt.show()

# 主函数：综合分析
def comprehensive_cluster_analysis(clustered_features_file, output_folder=None):
    """综合聚类分析"""
    
    print("=== 开始聚类特征差异分析 ===")
    
    # 1. 绘制14个特征的柱状图对比
    print("\n1. 绘制特征差异柱状图...")
    cluster_means = plot_cluster_feature_differences(clustered_features_file, output_folder)
    
    # 2. 绘制特征热力图
    print("\n2. 绘制特征热力图...")
    plot_cluster_feature_heatmap(clustered_features_file, output_folder)
    
    # 3. 绘制雷达图比较
    print("\n3. 绘制雷达图比较...")
    plot_cluster_comparison_radar(clustered_features_file, output_folder)
    
    # 4. 详细分析输出
    print("\n4. 生成聚类描述...")
    cluster_descriptions = analyze_cluster_differences(clustered_features_file)
    
    return cluster_means, cluster_descriptions

# 使用方法
if __name__ == "__main__":
    # 设置文件路径
    clustered_features_file = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/split/clustering_results/clustered_features.csv"
    output_folder = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/split/clustering_results1"
    os.makedirs(output_folder, exist_ok=True)
    # 执行综合分析
    cluster_means, descriptions = comprehensive_cluster_analysis(
        clustered_features_file, output_folder)
    
    print("\n=== 聚类总结 ===")
    for cluster_id, description in descriptions.items():
        print(f"聚类 {cluster_id}: {description}")