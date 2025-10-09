import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
import platform
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
# Mac最简单中文字体配置（无需清理缓存）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
print("Mac中文字体配置完成！")
# 设置seaborn样式
import seaborn as sns
sns.set_style("whitegrid")
import json

def load_reference_data(class_time_file, homework_file, class_schedule_file):
    """加载参考数据（新增课堂时间数据）"""
    df_class = pd.read_csv(class_time_file, encoding='utf-8-sig')
    df_homework = pd.read_csv(homework_file, encoding='utf-8-sig')
    df_schedule = pd.read_csv(class_schedule_file, encoding='utf-8-sig')  # 新增
    
    # 转换时间格式
    df_class['开始时间'] = pd.to_datetime(df_class['开始时间'])
    df_class['结束时间'] = pd.to_datetime(df_class['结束时间'])
    df_homework['发布时间'] = pd.to_datetime(df_homework['发布时间'], errors='coerce')
    df_homework['提交截止时间'] = pd.to_datetime(df_homework['提交截止时间'], errors='coerce')
    
    # 转换课堂时间格式
    df_schedule['开课时间'] = pd.to_datetime(df_schedule['开课时间'], errors='coerce')
    df_schedule['结课时间'] = pd.to_datetime(df_schedule['结课时间'], errors='coerce')
    
    return df_class, df_homework, df_schedule

def get_time_to_next_class(qa_time, class_id, df_schedule):
    """计算对话开始时间与接下来最近一次上课开始时间的间隔（小时）"""
    # 根据教学班ID筛选课程
    class_schedule = df_schedule[df_schedule['教学班ID'] == class_id]
    
    if class_schedule.empty:
        return np.inf  # 找不到对应课程，返回无穷大
    
    # 获取有效的开课时间
    start_times = pd.to_datetime(class_schedule['开课时间'], errors='coerce').dropna()
    
    if start_times.empty:
        return np.inf
    
    # 找到在QA时间之后的所有开课时间
    future_classes = start_times[start_times > qa_time]
    
    if not future_classes.empty:
        # 找到最近的一次开课时间
        next_class = future_classes.min()
        hours_diff = (next_class - qa_time).total_seconds() / 3600  # 转换为小时
        return max(hours_diff, 0)  # 确保非负
    else:
        return np.inf  # 没有未来的课程

def get_time_from_last_class(qa_time, class_id, df_schedule):
    """计算对话开始时间与之前最近一次上课结束时间的间隔（小时）"""
    # 根据教学班ID筛选课程
    class_schedule = df_schedule[df_schedule['教学班ID'] == class_id]
    
    if class_schedule.empty:
        return np.inf  # 找不到对应课程，返回无穷大
    
    # 获取有效的结课时间
    end_times = pd.to_datetime(class_schedule['结课时间'], errors='coerce').dropna()
    
    if end_times.empty:
        return np.inf
    
    # 找到在QA时间之前的所有结课时间
    past_classes = end_times[end_times < qa_time]
    
    if not past_classes.empty:
        # 找到最近的一次结课时间
        last_class = past_classes.max()
        hours_diff = (qa_time - last_class).total_seconds() / 3600  # 转换为小时
        return max(hours_diff, 0)  # 确保非负
    else:
        return np.inf  # 没有之前的课程

def replace_inf_with_reasonable_value(series, multiplier=1.5):
    """将无穷大值替换为合理的有限值"""
    if series.empty:
        return series
    
    # 获取非无穷大的值
    finite_values = series[np.isfinite(series)]
    
    if finite_values.empty:
        # 如果所有值都是无穷大，使用一个默认大值
        replacement_value = 168.0  # 一周的小时数作为默认值
    else:
        # 使用有限值的最大值的1.5倍作为替换值
        replacement_value = finite_values.max() * multiplier
    
    # 替换无穷大值
    series_cleaned = series.replace([np.inf, -np.inf], replacement_value)
    
    return series_cleaned

def get_teaching_week(qa_time, class_id, df_class):
    """计算QA发生在第几个教学周；仅当 qa_time 落在[开课时间, 结课时间]内才返回有效周数，否则返回 -1"""
    class_info = df_class[df_class['教学班ID'] == class_id]
    if class_info.empty:
        return -1

    start_time = pd.to_datetime(class_info['开始时间'].iloc[0], errors='coerce')
    end_time = pd.to_datetime(class_info['结束时间'].iloc[0], errors='coerce')

    qa_time = pd.to_datetime(qa_time, errors='coerce')

    if pd.isna(start_time) or pd.isna(end_time) or pd.isna(qa_time):
        return -1

    if not (start_time <= qa_time <= end_time):
        return -1

    days_diff = (qa_time - start_time).days
    week_num = max(1, (days_diff // 7) + 1)
    return week_num

def get_hours_to_next_assignment(qa_time, class_id, df_homework):
    """计算距离下一次作业截止的小时数（最大720小时）"""
    class_homework = df_homework[df_homework['教学班ID'] == class_id]
    if class_homework.empty:
        return 720  # 默认值（30天）

    deadline_times = pd.to_datetime(class_homework['提交截止时间'], errors='coerce').dropna()
    if deadline_times.empty:
        return 720

    qa_ts = pd.to_datetime(qa_time, errors='coerce')
    if pd.isna(qa_ts):
        return 720

    future_deadlines = deadline_times[deadline_times > qa_ts]
    if not future_deadlines.empty:
        next_deadline = future_deadlines.min()
        hours_diff = (next_deadline - qa_ts).total_seconds() / 3600
        return min(hours_diff, 720)  # 最大30天=720小时
    else:
        return 720  # 后面没有作业就返回720小时


def get_hours_since_last_assignment_release(qa_time, class_id, df_homework):
    """计算距离最近一次作业发布的小时数（从最近一次发布到 qa_time 的间隔，最大720小时）"""
    class_homework = df_homework[df_homework['教学班ID'] == class_id]
    if class_homework.empty:
        return 720  # 默认值（30天）

    release_times = pd.to_datetime(class_homework['发布时间'], errors='coerce').dropna()
    if release_times.empty:
        return 720

    qa_ts = pd.to_datetime(qa_time, errors='coerce')
    if pd.isna(qa_ts):
        return 720

    past_releases = release_times[release_times <= qa_ts]
    if not past_releases.empty:
        last_release = past_releases.max()
        hours_diff = (qa_ts - last_release).total_seconds() / 3600
        return min(hours_diff, 720)  # 最大30天=720小时
    else:
        return 720  # 如果 qa_time 之前没有发布记录，返回720小时

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
    使用自适应主成分数量的肘部法则找到最优聚类数
    """
    
    # 根据方差解释率确定主成分数量
    n_components, actual_variance = find_optimal_components_for_clustering(pca, variance_threshold)
    
    # 使用选定数量的主成分进行聚类
    X_cluster = X_pca[:, :n_components]
    
    print(f"\n=== 聚类配置 ===")
    print(f"使用主成分数量: {n_components}")
    print(f"聚类空间维度: {X_cluster.shape}")
    print(f"保留信息比例: {actual_variance:.3f} ({actual_variance*100:.1f}%)")
    
    inertias = []
    K_range = range(2, min(max_k + 1, len(X_cluster)))
    
    print(f"\n=== Using Elbow Method with {n_components} Principal Components ===")
    
    # 计算不同K值下的inertia
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_cluster)
        inertias.append(kmeans.inertia_)
        print(f"K={k}: Inertia={kmeans.inertia_:.2f}")
    
    # 三种肘部检测方法保持不变
    def find_elbow_second_derivative(inertias, K_range):
        if len(inertias) < 3:
            return K_range[0]
        first_diff = np.diff(inertias)
        second_diff = np.diff(first_diff)
        elbow_idx = np.argmax(second_diff) + 2
        return K_range[min(elbow_idx, len(K_range)-1)]
    
    def find_elbow_distance_method(inertias, K_range):
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
    
    # 使用三种方法检测肘部点
    elbow_k_method1 = find_elbow_second_derivative(inertias, K_range)
    elbow_k_method2 = find_elbow_distance_method(inertias, K_range)
    elbow_k_method3 = find_elbow_slope_method(inertias, K_range)
    
    print(f"\n=== Elbow Point Detection Results ===")
    print(f"Method 1 (Second Derivative): K = {elbow_k_method1}")
    print(f"Method 2 (Distance Method): K = {elbow_k_method2}")
    print(f"Method 3 (Slope Change): K = {elbow_k_method3}")
    
    # 选择最终的K值
    elbow_results = [elbow_k_method1, elbow_k_method2, elbow_k_method3]
    
    from collections import Counter
    count = Counter(elbow_results)
    if len(count.most_common(1)) > 0 and count.most_common(1)[0][1] > 1:
        final_k = count.most_common(1)[0][0]
        print(f"Final Selection (Mode): K = {final_k}")
        decision_reason = f"共识选择：{count.most_common(1)[0][1]}/3 方法支持"
    else:
        final_k = int(np.median(elbow_results))
        print(f"Final Selection (Median): K = {final_k}")
        decision_reason = "中位数选择：三种方法无共识"
    
    # 绘制增强的分析图（包含主成分信息）
    fig = plt.figure(figsize=(20, 12))
    
    # 创建网格布局
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    # 1. 主成分方差解释图
    ax1 = fig.add_subplot(gs[0, 0])
    pc_range = range(1, len(pca.explained_variance_ratio_) + 1)
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    
    ax1.bar(pc_range, pca.explained_variance_ratio_, alpha=0.6, label='Individual')
    ax1.plot(pc_range, cumulative_var, 'ro-', linewidth=2, label='Cumulative')
    ax1.axhline(y=variance_threshold, color='red', linestyle='--', alpha=0.7, 
                label=f'{variance_threshold*100:.0f}% Threshold')
    ax1.axvline(x=n_components, color='green', linestyle='--', alpha=0.7,
                label=f'Selected: PC{n_components}')
    
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title(f'PCA Analysis\n({n_components} PCs for {actual_variance*100:.1f}% variance)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 主要肘部法则图
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.plot(K_range, inertias, 'bo-', linewidth=3, markersize=10, label='Inertia Curve')
    ax2.axvline(x=final_k, color='red', linestyle='--', linewidth=3, 
                label=f'Selected K={final_k}')
    
    # 标记三种方法的结果
    method_colors = ['orange', 'green', 'purple']
    method_names = ['Second Derivative', 'Distance Method', 'Slope Change']
    method_results = [elbow_k_method1, elbow_k_method2, elbow_k_method3]
    
    for method_k, color, name in zip(method_results, method_colors, method_names):
        ax2.axvline(x=method_k, color=color, linestyle=':', alpha=0.8, linewidth=2,
                    label=f'{name} (K={method_k})')
    
    ax2.set_xlabel('Number of Clusters K')
    ax2.set_ylabel('Within-Cluster Sum of Squares (Inertia)')
    ax2.set_title(f'Elbow Method Analysis\n(Using {n_components} Principal Components)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 一阶差分图
    if len(inertias) > 1:
        ax3 = fig.add_subplot(gs[1, 0])
        first_diff = np.diff(inertias)
        ax3.plot(K_range[1:], first_diff, 'go-', linewidth=2, markersize=6)
        ax3.set_xlabel('Number of Clusters K')
        ax3.set_ylabel('First Derivative')
        ax3.set_title('Rate of Change in Inertia')
        ax3.grid(True, alpha=0.3)
    
    # 4. 二阶差分图
    if len(inertias) > 2:
        ax4 = fig.add_subplot(gs[1, 1])
        first_diff = np.diff(inertias)
        second_diff = np.diff(first_diff)
        ax4.plot(K_range[2:], second_diff, 'ro-', linewidth=2, markersize=6)
        ax4.axvline(x=elbow_k_method1, color='red', linestyle='--', alpha=0.7,
                    label=f'Max (K={elbow_k_method1})')
        ax4.set_xlabel('Number of Clusters K')
        ax4.set_ylabel('Second Derivative')
        ax4.set_title('Curvature Change')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. 距离分析图
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
                    label=f'Max (K={elbow_k_method2})')
        ax5.set_xlabel('Number of Clusters K')
        ax5.set_ylabel('Distance to Line')
        ax5.set_title('Point-to-Line Distance')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. 方法对比总结
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # 创建总结文本
    summary_text = f"""
    主成分分析总结:
    • 总特征数: {len(pca.explained_variance_ratio_)}
    • 目标方差解释率: {variance_threshold*100:.0f}%
    • 实际选择主成分数: {n_components}
    • 实际方差解释率: {actual_variance*100:.1f}%
    
    肘部法则结果:
    • 二阶差分法: K = {elbow_k_method1}
    • 距离法: K = {elbow_k_method2}  
    • 斜率变化法: K = {elbow_k_method3}
    • 最终选择: K = {final_k} ({decision_reason})
    
    聚类配置:
    • 特征空间: {n_components} 维主成分空间
    • 聚类算法: K-means
    • 聚类数量: {final_k}
    """
    
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=12,
            verticalalignment='center', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    
    plt.suptitle(f'Adaptive PCA + Elbow Method Analysis\n'
                f'({n_components} PCs explaining {actual_variance*100:.1f}% variance → K={final_k} clusters)', 
                fontsize=16, y=0.95)
    
    plt.tight_layout()
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'adaptive_elbow_method_analysis.png'), 
                    dpi=300, bbox_inches='tight')
    #plt.show()
    
    # 输出详细的分析结果
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

def extract_all_features(dialog_folder, class_time_file, homework_file, class_schedule_file):
    """提取所有对话文件的特征（新增课堂时间特征）"""
    print("Loading reference data...")
    try:
        df_class, df_homework, df_schedule = load_reference_data(
            class_time_file, homework_file, class_schedule_file)
    except Exception as e:
        print(f"Failed to load reference data: {e}")
        df_class, df_homework, df_schedule = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    print(f"Searching for dialog files in: {dialog_folder}")
    
    # 修复：递归查找所有CSV文件
    csv_files = []
    
    # 方法1：使用glob递归查找
    patterns = [
        os.path.join(dialog_folder, "*.csv"),              # 直接在根目录
        os.path.join(dialog_folder, "*", "*.csv"),         # 一级子目录
        os.path.join(dialog_folder, "*", "*", "*.csv"),    # 二级子目录
        os.path.join(dialog_folder, "**", "*.csv"),        # 递归查找
    ]
    
    for pattern in patterns:
        found_files = glob.glob(pattern, recursive=True)
        csv_files.extend(found_files)
    
    # 去重
    csv_files = list(set(csv_files))
    
    # 过滤掉明显不是对话文件的CSV（如feature、cluster、result等）
    dialog_files = []
    exclude_keywords = ['feature', 'cluster', 'result', 'statistic', 'analysis', 'pca']
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path).lower()
        if not any(keyword in file_name for keyword in exclude_keywords):
            dialog_files.append(file_path)
    
    print(f"Found {len(dialog_files)} potential dialog CSV files")
    
    # 如果还是没找到文件，显示文件夹结构帮助调试
    if not dialog_files:
        print("No CSV files found! Folder structure:")
        for root, dirs, files in os.walk(dialog_folder):
            level = root.replace(dialog_folder, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # 只显示前5个文件
                if file.endswith('.csv'):
                    print(f'{subindent}{file}')
            if len([f for f in files if f.endswith('.csv')]) > 5:
                csv_count = len([f for f in files if f.endswith('.csv')])
                print(f'{subindent}... {csv_count-5} more CSV files')
        return pd.DataFrame()
    
    # 验证文件是否为有效的对话文件
    print("Validating file formats...")
    valid_files = []
    
    for file_path in dialog_files[:10]:  # 先验证前10个文件
        try:
            df_test = pd.read_csv(file_path, encoding='utf-8-sig', nrows=3)
            required_columns = ['提问时间', '提问内容', 'AI回复']
            
            if all(col in df_test.columns for col in required_columns):
                valid_files.append(file_path)
                print(f"✓ Valid file: {os.path.basename(file_path)}")
            else:
                print(f"✗ Invalid file (missing required columns): {os.path.basename(file_path)}")
                print(f"  File columns: {df_test.columns.tolist()}")
        except Exception as e:
            print(f"✗ Failed to read: {os.path.basename(file_path)} - {e}")
    
    # 如果前10个都有效，假设其他的也有效
    if len(valid_files) == min(10, len(dialog_files)):
        valid_files = dialog_files
        print(f"First 10 files are valid, assuming all {len(dialog_files)} files are valid")
    else:
        print("Validating all files...")
        valid_files = []
        for file_path in dialog_files:
            try:
                df_test = pd.read_csv(file_path, encoding='utf-8-sig', nrows=3)
                required_columns = ['提问时间', '提问内容', 'AI回复']
                if all(col in df_test.columns for col in required_columns):
                    valid_files.append(file_path)
            except:
                continue
    
    print(f"Final number of valid files: {len(valid_files)}")
    
    if not valid_files:
        print("Error: No valid dialog files found!")
        return pd.DataFrame()
    
    # 提取特征
    features_list = []
    failed_count = 0
    stats = {'total': 0, 'processed': 0, 'failed': 0, 'out_of_range': 0}  # 初始化

    for i, file_path in enumerate(valid_files):
        if i % 100 == 0:
            print(f"Processing progress: {i+1}/{len(valid_files)}")
        
        features = extract_features_from_dialog(file_path, df_class, df_homework, df_schedule, stats=stats)
        if features is not None:
            features_list.append(features)
        else:
            failed_count += 1
            if failed_count <= 5:
                print(f"  Feature extraction failed: {os.path.basename(file_path)}")
    
    print(f"Successfully extracted features from {len(features_list)} dialogs")
    print(f"Failed on {failed_count} files")

    # 计算并保存 stats 到 homework_file 路径下的 JSON
    total = int(stats.get('total', 0))
    processed = int(stats.get('processed', 0))
    out_of_range = int(stats.get('out_of_range', 0))
    failed_other = int(stats.get('failed', 0))

    stats_to_save = {
        "valid_files_found": int(len(valid_files)),
        "total_dialog_calls": total,                 # 传入 extract_features_from_dialog 的调用次数
        "processed_dialogs": processed,              # 成功返回 features 的对话数
        "out_of_range_dialogs": out_of_range,        # 因超出教学周被筛掉的对话数
        "other_failed_dialogs": failed_other,        # 其他原因失败的对话数
        "successful_feature_rows": int(len(features_list)),
        "failed_count_in_loop": int(failed_count),   # 循环中的失败计数（用于打印）
    }
    if total > 0:
        stats_to_save.update({
            "success_ratio": processed / total,
            "out_of_range_ratio": out_of_range / total,
            "fail_ratio": (out_of_range + failed_other) / total
        })

    stats_dir = os.path.dirname(os.path.abspath(homework_file))
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, "dialog_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_to_save, f, ensure_ascii=False, indent=2)
    print(f"Stats saved to: {stats_path}")

    if not features_list:
        print("Warning: No features were successfully extracted!")
        return pd.DataFrame()

    # 转换为DataFrame
    features_df = pd.DataFrame(features_list)

    # 处理新增的时间特征中的无穷大值
    print("Processing infinite values in class time features...")
    time_features = ['hours_to_next_class', 'hours_from_last_class']
    for feature in time_features:
        if feature in features_df.columns:
            inf_count = np.isinf(features_df[feature]).sum()
            if inf_count > 0:
                print(f"Found {inf_count} infinite values in {feature}")
                features_df[feature] = replace_inf_with_reasonable_value(
                    features_df[feature], multiplier=1.5)
                print(f"Replaced with max finite value * 1.5 = {features_df[feature].max():.2f}")

    return features_df

def extract_features_from_dialog(file_path, df_class, df_homework, df_schedule, stats=None):
    """从单个对话文件中提取特征
       - 超出教学周范围的对话直接筛掉（返回 None）
       - 新增变量：course_progress_ratio, calendar_week_since_2025_0217
       - 新增变量：is_multi_turn（对话轮次是否大于1，True/False）
       - 可选 stats 计数 out_of_range 比例
    """
    try:
        if stats is not None:
            stats['total'] = stats.get('total', 0) + 1
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            if stats is not None:
                stats['failed'] = stats.get('failed', 0) + 1
            return None
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        if df.empty:
            print(f"File is empty: {os.path.basename(file_path)}")
            if stats is not None:
                stats['failed'] = stats.get('failed', 0) + 1
            return None
        required_columns = ['提问时间', '提问内容', 'AI回复']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"File missing required columns {missing_columns}: {os.path.basename(file_path)}")
            if stats is not None:
                stats['failed'] = stats.get('failed', 0) + 1
            return None
        df.fillna("", inplace=True)
        file_name = os.path.basename(file_path)
        # 获取教学班ID
        if '教学班ID' in df.columns:
            class_id = df["教学班ID"].iloc[0]
        else:
            print(f"Warning: File missing class ID column: {file_name}")
            if stats is not None:
                stats['failed'] = stats.get('failed', 0) + 1
            return None
        # 转换提问时间
        try:
            df["提问时间"] = pd.to_datetime(df["提问时间"], errors='coerce')
        except Exception as e:
            print(f"Time conversion failed: {file_name} - {e}")
            if stats is not None:
                stats['failed'] = stats.get('failed', 0) + 1
            return None
        if df["提问时间"].isna().any():
            print(f"Invalid QA times in: {file_name}")
            if stats is not None:
                stats['failed'] = stats.get('failed', 0) + 1
            return None
        # 课程信息与区间校验
        class_info = df_class[df_class['教学班ID'] == class_id]
        if class_info.empty:
            print(f"No class info for class_id={class_id}: {file_name}")
            if stats is not None:
                stats['failed'] = stats.get('failed', 0) + 1
            return None
        course_start = pd.to_datetime(class_info['开始时间'].iloc[0], errors='coerce')
        course_end = pd.to_datetime(class_info['结束时间'].iloc[0], errors='coerce')
        if pd.isna(course_start) or pd.isna(course_end):
            print(f"Invalid course start/end time for class_id={class_id}: {file_name}")
            if stats is not None:
                stats['failed'] = stats.get('failed', 0) + 1
            return None
        qa_min_time = df["提问时间"].min()
        qa_max_time = df["提问时间"].max()
        # 若任一 QA 时间不在课程区间内，筛掉该对话
        if (qa_min_time < course_start) or (qa_max_time > course_end):
            print(f"Dialog outside course window [{course_start}, {course_end}] -> skip: {file_name}")
            if stats is not None:
                stats['out_of_range'] = stats.get('out_of_range', 0) + 1
            return None
        # 1. 对话中有几轮问答
        qa_turns = len(df)
        # 新增：对话轮次是否大于1（True/False）
        is_multi_turn = qa_turns > 1
        # 2. 对话总共所花时间（分钟）
        if qa_turns > 1:
            total_time = (df["提问时间"].max() - df["提问时间"].min()).total_seconds() / 60
            total_time = max(0, total_time)
        else:
            total_time = 0
        # 3. 对话中每个问答的平均时间（分钟）
        if qa_turns > 1 and total_time > 0:
            avg_qa_time = total_time / (qa_turns - 1)
        else:
            avg_qa_time = 0
        # 4. 对话中学生提问的总文字数
        question_lengths = df["提问内容"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        total_question_chars = int(question_lengths.sum())
        # 5. 对话中每个问题的平均文字数
        avg_question_length = float(question_lengths.mean()) if len(question_lengths) > 0 else 0.0
        # 6-10. 提问入口特征
        if '提问入口' in df.columns:
            is_courseware = int("课件不懂" in df["提问入口"].values)
            is_discussion = int("讨论单元" in df["提问入口"].values)
            is_ai_task = int("课堂AI任务" in df["提问入口"].values)
            is_exercise = int("习题不懂" in df["提问入口"].values)
            if_non_class = int((df["提问入口"] != "班级").any()) if qa_turns > 0 else 0
        else:
            print(f"Warning: File missing entry point column: {file_name}")
            is_courseware = is_discussion = is_ai_task = is_exercise = 0
            if_non_class = 0
        # 11. QA与下一次最接近的作业的时间关系（小时）
        hours_to_assignment_list = []
        for qa_time in df["提问时间"]:
            hours = get_hours_to_next_assignment(qa_time, class_id, df_homework)
            hours_to_assignment_list.append(hours)
        avg_hours_to_assignment = float(np.mean(hours_to_assignment_list)) if hours_to_assignment_list else 720.0
        # 11.5 QA与最近一次作业发布的小时数
        hours_since_release_list = []
        for qa_time in df["提问时间"]:
            hours = get_hours_since_last_assignment_release(qa_time, class_id, df_homework)
            hours_since_release_list.append(hours)
        avg_hours_since_release = float(np.mean(hours_since_release_list)) if hours_since_release_list else 720.0
        # 12. 对话位于教学进度的比例（0-1）
        total_weeks = max(1, ((course_end - course_start).days // 7) + 1)
        progress_values = []
        for t in df["提问时间"]:
            wk = get_teaching_week(t, class_id, df_class)
            if wk > 0:
                progress_values.append(wk / total_weeks)
        course_progress_ratio = float(np.mean(progress_values)) if progress_values else 0.0
        # 12.5 对话发生的自然周（以 2025-02-17 为第1周的起点，取对话最早一次提问所在周）
        anchor = pd.Timestamp('2025-02-17')  # 周一
        qa_start_time = df["提问时间"].min()
        calendar_week_since_2025_0217 = int(((qa_start_time.normalize() - anchor.normalize()).days // 7) + 1)
        # 13. 距离下次课开始的时间间隔（小时）
        hours_to_next_class_list = []
        for qa_time in df["提问时间"]:
            hours = get_time_to_next_class(qa_time, class_id, df_schedule)
            hours_to_next_class_list.append(hours)
        avg_hours_to_next_class = float(np.mean(hours_to_next_class_list)) if hours_to_next_class_list else float('inf')
        # 14. 距离上次课结束的时间间隔（小时）
        hours_from_last_class_list = []
        for qa_time in df["提问时间"]:
            hours = get_time_from_last_class(qa_time, class_id, df_schedule)
            hours_from_last_class_list.append(hours)
        avg_hours_from_last_class = float(np.mean(hours_from_last_class_list)) if hours_from_last_class_list else float('inf')
        features = {
            "file_name": file_name,
            "class_id": class_id,
            "qa_turns": int(qa_turns),
            "is_multi_turn": bool(is_multi_turn),  # 新增特征：对话轮次是否大于1
            "total_time_minutes": float(total_time),
            "avg_qa_time_minutes": float(avg_qa_time),
            "total_question_chars": int(total_question_chars),
            "avg_question_length": float(avg_question_length),
            "is_courseware_entry": int(is_courseware),
            "is_discussion_entry": int(is_discussion),
            "is_ai_task_entry": int(is_ai_task),
            "is_exercise_entry": int(is_exercise),
            "if_non_class": int(if_non_class),
            "avg_hours_to_assignment": float(avg_hours_to_assignment),
            "avg_hours_since_release": float(avg_hours_since_release),
            "course_progress_ratio": float(course_progress_ratio),
            "calendar_week_since_2025_0217": int(calendar_week_since_2025_0217),
            "hours_to_next_class": float(avg_hours_to_next_class),
            "hours_from_last_class": float(avg_hours_from_last_class),
        }
        # 数值健壮性处理（保留 inf 的两项不校验）
        for key, value in features.items():
            if key not in ['file_name', 'class_id', 'hours_to_next_class', 'hours_from_last_class']:
                if not np.isfinite(value) and not isinstance(value, (bool, str, int)):
                    features[key] = 0.0
        if stats is not None:
            stats['processed'] = stats.get('processed', 0) + 1
        return features
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        if stats is not None:
            stats['failed'] = stats.get('failed', 0) + 1
        return None

def perform_pca_analysis(features_df, output_folder):
    """执行PCA分析（更新特征列表；0/1二元变量不做标准化）"""
    # 选择数值特征
    feature_columns = [
        'qa_turns', 'is_multi_turn', 'total_time_minutes', 'avg_qa_time_minutes',
        'total_question_chars', 'avg_question_length',
        'is_courseware_entry', 'is_discussion_entry', 'is_ai_task_entry', 'is_exercise_entry',
        'if_non_class', 'avg_hours_to_assignment', 'avg_hours_since_release',
        'course_progress_ratio', 'calendar_week_since_2025_0217',
        'hours_to_next_class', 'hours_from_last_class'
    ]
    
    # 定义不进行标准化的二元特征列
    binary_cols = [
        'is_multi_turn', 'is_courseware_entry', 'is_discussion_entry',
        'is_ai_task_entry', 'is_exercise_entry', 'if_non_class'
    ]
    # 实际存在的数据列交集
    binary_cols_present = [c for c in binary_cols if c in feature_columns and c in features_df.columns]
    continuous_cols = [c for c in feature_columns if c not in binary_cols_present]

    X = features_df[feature_columns].copy()
    
    # 处理缺失值和异常值
    print("Processing missing values and outliers...")
    print(f"Original data shape: {X.shape}")
    print(f"Missing values count:\n{X.isnull().sum()}")
    
    # 先填充缺失值（对所有列使用中位数，二元列中位数仍为0或1，不影响后续）
    X = X.fillna(X.median(numeric_only=True))
    
    # 特殊处理新增的时间特征中的无穷大值
    print("Processing infinite values in class time features...")
    time_features = ['hours_to_next_class', 'hours_from_last_class']
    for feature in time_features:
        if feature in X.columns:
            inf_count = np.isinf(X[feature]).sum()
            if inf_count > 0:
                print(f"Column {feature} found {inf_count} infinite values")
                finite_values = X[feature][np.isfinite(X[feature])]
                if not finite_values.empty:
                    max_finite = finite_values.max()
                    replacement_value = max_finite * 1.5
                    print(f"  Max finite value: {max_finite:.2f} hours")
                    print(f"  Replacement value: {replacement_value:.2f} hours")
                else:
                    replacement_value = 168.0  # 一周（7*24小时）
                    print(f"  No finite values found, using default: {replacement_value:.2f} hours")
                X.loc[np.isinf(X[feature]), feature] = replacement_value
    
    # 处理其他列的无穷大值
    for col in X.columns:
        if col not in time_features:
            inf_mask = np.isinf(X[col])
            if inf_mask.any():
                print(f"Column {col} found {inf_mask.sum()} infinite values")
                median_val = X[col][~inf_mask].median()
                X.loc[inf_mask, col] = median_val
    
    # 检查是否有常数列（仅对需要标准化的连续列处理）
    if continuous_cols:
        constant_cols = X[continuous_cols].columns[X[continuous_cols].var() == 0].tolist()
        if constant_cols:
            print(f"Found constant continuous columns (adding tiny noise to avoid zero variance): {constant_cols}")
            for col in constant_cols:
                X[col] += np.random.normal(0, 1e-8, len(X))
    else:
        print("No continuous columns to standardize. Only binary/indicator features present.")

    # 标准化：仅对连续特征列进行标准化，二元列保持原始0/1值
    print("Performing standardization on continuous features only...")
    scaler = StandardScaler()
    if continuous_cols:
        X_cont_scaled = scaler.fit_transform(X[continuous_cols])
        X_scaled_df = pd.DataFrame(X_cont_scaled, columns=continuous_cols, index=X.index)
    else:
        # 若没有连续列，创建空DataFrame并保留 scaler 未使用的状态
        X_scaled_df = pd.DataFrame(index=X.index)

    # 将二元列直接拼接（不做标准化）
    for col in binary_cols_present:
        X_scaled_df[col] = X[col].astype(float)  # 转成浮点以便与标准化后的矩阵共同用于PCA

    # 重新排列列顺序，保持与 feature_columns 一致
    X_scaled_df = X_scaled_df[feature_columns]
    X_scaled = X_scaled_df.values

    # 检查标准化后的数据
    if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
        print("Warning: Still have outliers after standardization")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 显示新增时间特征统计信息
    print(f"\n=== Class Time Features Statistics ===")
    for feature in time_features:
        if feature in features_df.columns:
            feature_data = features_df[feature]
            finite_data = feature_data[np.isfinite(feature_data)]
            inf_count = np.isinf(feature_data).sum()
            print(f"{feature}:")
            print(f"  Total samples: {len(feature_data)}")
            print(f"  Infinite values: {inf_count} ({inf_count/len(feature_data)*100:.1f}%)")
            if not finite_data.empty:
                print(f"  Finite values - Min: {finite_data.min():.2f}h, Max: {finite_data.max():.2f}h, Mean: {finite_data.mean():.2f}h")
            print()
    
    # PCA分析
    print("Performing PCA...")
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # 打印PCA结果
    print("\n=== PCA Analysis Results ===")
    print("Explained variance ratio for each principal component:")
    for i, ratio in enumerate(pca.explained_variance_ratio_[:6]):
        print(f"PC{i+1}: {ratio:.4f}")
    print(f"\nCumulative explained variance ratio for first 3 components: {pca.explained_variance_ratio_[:3].sum():.4f}")
    
    # 特征重要性分析
    print("\nFeature loadings in first 3 principal components:")
    components_df = pd.DataFrame(
        pca.components_[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=feature_columns
    )
    print(components_df.round(3))
    
    # 可视化PCA结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 解释方差比
    axes[0,0].bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
    axes[0,0].set_xlabel('Principal Components')
    axes[0,0].set_ylabel('Explained Variance Ratio')
    axes[0,0].set_title('Explained Variance Ratio by Component')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 累计解释方差比
    axes[0,1].plot(range(1, len(pca.explained_variance_ratio_)+1), 
                   np.cumsum(pca.explained_variance_ratio_), 'bo-')
    axes[0,1].axhline(y=0.8, color='r', linestyle='--', label='80%')
    axes[0,1].set_xlabel('Number of Components')
    axes[0,1].set_ylabel('Cumulative Explained Variance Ratio')
    axes[0,1].set_title('Cumulative Explained Variance Ratio')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 特征载荷热力图
    sns.heatmap(components_df.T, annot=True, cmap='RdBu_r', center=0, 
                ax=axes[1,0], cbar_kws={'label': 'Loading'})
    axes[1,0].set_title('Feature Loadings in Principal Components')
    
    # 4. 2D PCA散点图
    axes[1,1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=30)
    axes[1,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[1,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    axes[1,1].set_title('PCA 2D Projection')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pca_analysis.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    
    return X_scaled, X_pca, pca, scaler, feature_columns

def find_optimal_clusters(X_pca, max_k=10, output_folder=None):
    """使用肘部法则找到最优聚类数"""
    # 使用前几个主成分进行聚类
    n_components = min(5, X_pca.shape[1])
    X_cluster = X_pca[:, :n_components]
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, min(max_k + 1, len(X_cluster)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_cluster)
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(X_cluster, cluster_labels)
        silhouette_scores.append(sil_score)
        print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
    
    # 绘制肘部图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 肘部法则
    axes[0].plot(K_range, inertias, 'bo-')
    axes[0].set_xlabel('Number of Clusters K')
    axes[0].set_ylabel('Within-Cluster Sum of Squares (Inertia)')
    axes[0].set_title('Elbow Method')
    axes[0].grid(True, alpha=0.3)
    
    # 轮廓系数
    axes[1].plot(K_range, silhouette_scores, 'ro-')
    axes[1].set_xlabel('Number of Clusters K')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'optimal_clusters.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    
    # 推荐最优K值
    best_k_silhouette = K_range[np.argmax(silhouette_scores)]
    print(f"\nRecommended number of clusters (based on silhouette score): K = {best_k_silhouette}")
    
    return best_k_silhouette, X_cluster

def perform_clustering(X_cluster, n_clusters, features_df, output_folder):
    """执行聚类分析"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster)
    
    # 添加聚类标签到原始数据
    features_df_clustered = features_df.copy()
    features_df_clustered['cluster'] = cluster_labels
    
    # 聚类统计
    print(f"\n=== Clustering Results Statistics ===")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
    
    return cluster_labels, features_df_clustered, kmeans

def analyze_cluster_characteristics(features_df_clustered, feature_columns, output_folder):
    """分析各簇的特征"""
    print(f"\n=== Cluster Characteristics Analysis ===")
    
    # 计算各簇的特征均值
    cluster_means = features_df_clustered.groupby('cluster')[feature_columns].mean()
    print("\nMean values for each cluster:")
    print(cluster_means.round(3))
    
    # 保存详细统计
    cluster_stats = features_df_clustered.groupby('cluster')[feature_columns].agg(['mean', 'std', 'median'])
    cluster_stats.to_csv(os.path.join(output_folder, 'cluster_statistics.csv'), encoding='utf-8-sig')
    
    # 可视化簇特征
    n_clusters = len(cluster_means)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_columns):
        if i < len(axes):
            cluster_means[feature].plot(kind='bar', ax=axes[i], color='skyblue', alpha=0.7)
            axes[i].set_title(f'{feature}')
            axes[i].set_xlabel('Cluster ID')
            axes[i].tick_params(axis='x', rotation=0)
            axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(feature_columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cluster_characteristics.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    
    return cluster_means

def comprehensive_clustering_visualization(X_scaled, X_pca, cluster_labels, features_df_clustered, 
                                        feature_columns, n_clusters, output_folder):
    """综合聚类可视化"""
    fig = plt.figure(figsize=(20, 16))
    
    # 1. PCA 2D可视化
    plt.subplot(2, 3, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                         cmap='tab10', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.title(f'PCA Clustering Visualization (K={n_clusters})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter, label='Cluster Label')
    
    # 添加簇中心
    for i in range(n_clusters):
        cluster_points = X_pca[cluster_labels == i]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            plt.scatter(centroid[0], centroid[1], c='red', s=200, marker='x', linewidth=3)
            plt.annotate(f'C{i}', (centroid[0], centroid[1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    # 2. t-SNE可视化
    plt.subplot(2, 3, 2)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
    X_tsne = tsne.fit_transform(X_scaled)
    
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, 
                         cmap='tab10', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.title('t-SNE Clustering Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter, label='Cluster Label')
    
    # 3. 簇大小分布
    plt.subplot(2, 3, 3)
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    bars = plt.bar(range(n_clusters), cluster_counts.values, 
                   color=plt.cm.tab10(np.arange(n_clusters)), alpha=0.7, edgecolor='black')
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Samples')
    plt.xticks(range(n_clusters))
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 4. QA轮次分布
    plt.subplot(2, 3, 4)
    qa_data = []
    qa_labels = []
    for i in range(n_clusters):
        cluster_qa = features_df_clustered[features_df_clustered['cluster'] == i]['qa_turns']
        if len(cluster_qa) > 0:
            qa_data.append(cluster_qa)
            qa_labels.append(f'C{i}')
    
    if qa_data:
        box_plot = plt.boxplot(qa_data, labels=qa_labels, patch_artist=True)
        colors = plt.cm.tab10(np.arange(len(qa_data)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    plt.title('QA Turns Distribution')
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of QA Turns')
    plt.grid(True, alpha=0.3)
    
    # 5. 特征重要性热力图
    plt.subplot(2, 3, 5)
    cluster_means = features_df_clustered.groupby('cluster')[feature_columns].mean()
    
    # 标准化以便比较
    cluster_means_norm = (cluster_means - cluster_means.mean()) / cluster_means.std()
    
    im = plt.imshow(cluster_means_norm.T, cmap='RdBu_r', aspect='auto')
    plt.title('Cluster Features Heatmap (Normalized)')
    plt.xlabel('Cluster Label')
    plt.ylabel('Features')
    plt.xticks(range(n_clusters), [f'C{i}' for i in range(n_clusters)])
    plt.yticks(range(len(feature_columns)), feature_columns, fontsize=8)
    plt.colorbar(im, label='Normalized Value')
    
    # 6. 教学周分布
    plt.subplot(2, 3, 6)
    week_data = []
    week_labels = []
    for i in range(n_clusters):
        cluster_weeks = features_df_clustered[features_df_clustered['cluster'] == i]['course_progress_ratio']
        cluster_weeks = cluster_weeks[cluster_weeks > 0]  # 过滤无效值
        if len(cluster_weeks) > 0:
            week_data.append(cluster_weeks)
            week_labels.append(f'C{i}')
    
    if week_data:
        plt.hist(week_data, bins=20, alpha=0.7, label=week_labels)
        plt.legend()
    
    plt.title('Teaching Week Distribution')
    plt.xlabel('Teaching Week')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'comprehensive_clustering_visualization.png'), 
                dpi=300, bbox_inches='tight')
    #plt.show()

def debug_infinite_values(features_df):
    """调试无穷大值（更新版）"""
    print("=== Checking Infinite Values ===")
    
    # 更新特征列表（新增两个时间特征）
    feature_columns = [
        'qa_turns', 'is_multi_turn','total_time_minutes', 'avg_qa_time_minutes',
        'total_question_chars', 'avg_question_length',
        'is_courseware_entry', 'is_discussion_entry', 'is_ai_task_entry', 'is_exercise_entry',
        'if_non_class', 'avg_hours_to_assignment', 'avg_hours_since_release', 'course_progress_ratio', 'calendar_week_since_2025_0217',
        'hours_to_next_class', 'hours_from_last_class'  # 新增
    ]
    
    print("Available columns in features_df:")
    print(features_df.columns.tolist())
    
    for col in feature_columns:
        if col in features_df.columns:
            inf_count = np.isinf(features_df[col]).sum()
            nan_count = np.isnan(features_df[col]).sum()
            
            if inf_count > 0 or nan_count > 0:
                print(f"Column '{col}': {inf_count} infinite values, {nan_count} NaN values")
                
                if inf_count > 0:
                    inf_indices = features_df[np.isinf(features_df[col])].index
                    print(f"  Infinite values in rows: {inf_indices.tolist()[:5]}...")
                    if 'file_name' in features_df.columns:
                        print(f"  Corresponding files: {features_df.loc[inf_indices[:3], 'file_name'].tolist()}")
            
            # 显示基本统计（排除无穷大值）
            finite_data = features_df[col][np.isfinite(features_df[col])]
            if not finite_data.empty:
                print(f"  {col} (finite only): min={finite_data.min():.2f}, max={finite_data.max():.2f}, mean={finite_data.mean():.2f}")
            else:
                print(f"  {col}: All values are infinite or NaN")
    
    return

import argparse
import sys

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='聚类分析工具')
    parser.add_argument('dialog_folder', 
                       help='对话文件夹路径')
    parser.add_argument('--class_time_file', 
                       default="/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/class_time_range_merged.csv",
                       help='课程时间文件路径')
    parser.add_argument('--homework_file',
                       default="/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/题目明细.csv", 
                       help='作业文件路径')
    parser.add_argument('--class_schedule_file',
                       default="/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/课堂开结课时间.csv",
                       help='课堂时间表文件路径')
    parser.add_argument('--max_k', type=int, default=10,
                       help='最大聚类数')
    parser.add_argument('--variance_threshold', type=float, default=0.8,
                       help='PCA方差阈值')
    
    return parser.parse_args()

def main():
    """主函数（命令行版本）"""
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 配置路径
    dialog_folder = args.dialog_folder
    class_time_file = args.class_time_file
    homework_file = args.homework_file
    class_schedule_file = args.class_schedule_file
    
    # 自动生成输出文件夹
    output_folder = os.path.join(dialog_folder, "clustering_results")
    
    # 验证输入文件夹
    if not os.path.exists(dialog_folder):
        print(f"❌ Error: Dialog folder does not exist: {dialog_folder}")
        sys.exit(1)
    
    print("=== Clustering Analysis Configuration ===")
    print(f"Dialog folder: {dialog_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Class time file: {class_time_file}")
    print(f"Homework file: {homework_file}")
    print(f"Class schedule file: {class_schedule_file}")
    print(f"Max K: {args.max_k}")
    print(f"Variance threshold: {args.variance_threshold}")
    
    # 检查其他必要文件
    missing_files = []
    for file_path, file_desc in [
        (class_time_file, "Class time file"),
        (homework_file, "Homework file"),
        (class_schedule_file, "Class schedule file")
    ]:
        if not os.path.exists(file_path):
            missing_files.append(f"{file_desc}: {file_path}")
    
    if missing_files:
        print("\n⚠️ Warning: Missing reference files:")
        for missing in missing_files:
            print(f"  - {missing}")
        print("Analysis will continue with available data...\n")
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    print("=== Starting Feature Extraction and Clustering Analysis ===")
    
    try:
        # 1. 特征提取
        print("\n1. Extracting features...")
        features_df = extract_all_features(dialog_folder, class_time_file, homework_file, class_schedule_file)
        
        # 检查是否成功提取特征
        if features_df.empty:
            print("❌ Error: No features extracted, terminating program")
            return None, None
        
        print(f"✅ Successfully extracted features from {len(features_df)} dialogs")
        
        # 调试步骤
        print("\n=== Debug: Checking Extracted Features ===")
        debug_infinite_values(features_df)
        
        # 保存特征
        features_file = os.path.join(output_folder, 'extracted_features.csv')
        features_df.to_csv(features_file, index=False, encoding='utf-8-sig')
        print(f"Features saved to: {features_file}")
        
        # 显示描述性统计
        print("\n=== Feature Descriptive Statistics ===")
        feature_columns = [col for col in features_df.columns if col not in ['file_name', 'class_id']]
        if feature_columns and len(features_df) > 0:
            print(features_df[feature_columns].describe())
        
        # 2. PCA分析
        print("\n2. Performing PCA analysis...")
        X_scaled, X_pca, pca, scaler, feature_cols = perform_pca_analysis(features_df, output_folder)
        
        # 3. 使用肘部法则寻找最优聚类数
        print("\n3. Finding optimal number of clusters using elbow method...")
        optimal_k, X_cluster = find_optimal_clusters_elbow_only(X_pca, pca, max_k=10, output_folder=output_folder)
        
        # 4. 执行聚类
        print(f"\n4. Performing clustering analysis (K={optimal_k})...")
        cluster_labels, features_df_clustered, kmeans = perform_clustering(
            X_cluster, optimal_k, features_df, output_folder)
        
        # 保存聚类结果
        clustered_file = os.path.join(output_folder, 'clustered_features.csv')
        features_df_clustered.to_csv(clustered_file, index=False, encoding='utf-8-sig')
        print(f"Clustered features saved to: {clustered_file}")
        
        # 5. 分析簇特征
        print("\n5. Analyzing cluster characteristics...")
        cluster_means = analyze_cluster_characteristics(features_df_clustered, feature_cols, output_folder)
        
        # 6. 综合可视化
        print("\n6. Generating visualizations...")
        comprehensive_clustering_visualization(
            X_scaled, X_pca, cluster_labels, features_df_clustered, 
            feature_cols, optimal_k, output_folder)
        
        # 输出最终配置摘要
        print(f"\n=== Analysis Complete! ===")
        print(f"📁 Results saved to: {output_folder}")
        print(f"📊 Configuration:")
        print(f"   - Input folder: {dialog_folder}")
        print(f"   - Total samples: {len(features_df_clustered):,}")
        print(f"   - Features: {len(feature_cols)}")
        print(f"   - Clusters: {optimal_k}")
        
        # 保存配置信息
        config_info = {
            'dialog_folder': dialog_folder,
            'output_folder': output_folder,
            'total_samples': len(features_df_clustered),
            'features_count': len(feature_cols),
            'optimal_clusters': optimal_k,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        config_file = os.path.join(output_folder, 'analysis_config.json')
        import json
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Configuration saved to: {config_file}")
        
        return features_df_clustered, cluster_means
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    features_df_clustered, cluster_means = main()