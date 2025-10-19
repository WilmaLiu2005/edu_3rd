"""
命令行接口模块

提供QA对话聚类分析的完整工作流，包括：
- 特征提取
- PCA降维分析
- 聚类分析
- 结果可视化
"""

import os
import sys
import json
from typing import Optional, Tuple, List
import pandas as pd

from .config import setup_plotting
from .features import extract_all_features
from .pca_utils import perform_pca_analysis
from .clustering import (
    find_optimal_clusters_elbow_only,
    perform_clustering,
    analyze_cluster_characteristics,
    comprehensive_clustering_visualization_all_k,
    plot_cluster_feature_heatmap
)
from .feature_utils import debug_infinite_values

# 常量定义
DEFAULT_MAX_K = 10
DEFAULT_VARIANCE_THRESHOLD = 0.8
PREFERRED_FILE_EXTENSIONS = ('.csv', '.txt')
DEFAULT_CLUSTER_COLUMNS = ['cluster', 'cluster_k2', 'cluster_k3', 'cluster_k4']


def find_file_recursively(
    root_dir: str,
    file_name: str,
    prefer_exts: Tuple[str, ...] = PREFERRED_FILE_EXTENSIONS,
    case_insensitive: bool = True
) -> Optional[str]:
    """
    在指定目录下递归查找文件
    
    Args:
        root_dir: 根目录路径
        file_name: 要查找的文件名（可带或不带后缀）
        prefer_exts: 优先匹配的文件扩展名顺序
        case_insensitive: 是否大小写不敏感
    
    Returns:
        str: 找到的文件完整路径，未找到返回None
    """
    if not file_name or not isinstance(file_name, str):
        return None

    # 检查是否为绝对路径或已存在的相对路径
    if os.path.isabs(file_name) and os.path.exists(file_name):
        return file_name
    
    direct_path = os.path.join(root_dir, file_name)
    if os.path.exists(direct_path):
        return direct_path

    # 分离文件名和扩展名
    target = os.path.basename(file_name)
    tbase, text = os.path.splitext(target)
    tname_cmp = target.lower() if case_insensitive else target
    tbase_cmp = tbase.lower() if case_insensitive else tbase

    best_path = None
    best_rank = float('inf')

    # 递归搜索
    for root, _, files in os.walk(root_dir):
        for f in files:
            f_cmp = f.lower() if case_insensitive else f
            
            if text:  # 目标文件名包含扩展名
                if f_cmp == tname_cmp:
                    return os.path.join(root, f)
            else:  # 目标文件名不含扩展名，按优先级匹配
                fbase, fext = os.path.splitext(f)
                fbase_cmp = fbase.lower() if case_insensitive else fbase
                
                if fbase_cmp == tbase_cmp:
                    rank = prefer_exts.index(fext.lower()) if fext.lower() in prefer_exts else len(prefer_exts) + 1
                    if rank < best_rank:
                        best_rank = rank
                        best_path = os.path.join(root, f)
                        if best_rank == 0:
                            return best_path

    return best_path


def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='学堂在线QA对话聚类分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 必需参数
    parser.add_argument(
        '--dialog_folder',
        required=True,
        help='对话文件所在文件夹路径'
    )
    parser.add_argument(
        '--class_time_file',
        required=True,
        help='课程时间信息文件路径（包含课程开始和结束时间）'
    )
    parser.add_argument(
        '--homework_file',
        required=True,
        help='作业信息文件路径'
    )
    parser.add_argument(
        '--class_schedule_file',
        required=True,
        help='课程表文件路径（包含开课时间）'
    )
    
    # 可选参数
    parser.add_argument(
        '--school_info_file',
        required=False,
        default='',
        help='学校基本信息文件路径（可选）'
    )
    parser.add_argument(
        '--class_info_file',
        required=False,
        default='',
        help='班级信息文件路径（可选）'
    )
    parser.add_argument(
        '--max_k',
        type=int,
        default=DEFAULT_MAX_K,
        help=f'最大聚类数（默认：{DEFAULT_MAX_K}）'
    )
    parser.add_argument(
        '--variance_threshold',
        type=float,
        default=DEFAULT_VARIANCE_THRESHOLD,
        help=f'PCA方差阈值（默认：{DEFAULT_VARIANCE_THRESHOLD}）'
    )
    
    return parser.parse_args()


def validate_input_files(
    dialog_folder: str,
    class_time_file: str,
    homework_file: str,
    class_schedule_file: str
) -> List[str]:
    """
    验证输入文件是否存在
    
    Args:
        dialog_folder: 对话文件夹路径
        class_time_file: 课程时间文件路径
        homework_file: 作业文件路径
        class_schedule_file: 课程表文件路径
    
    Returns:
        list: 缺失的文件列表
    """
    missing_files = []
    
    if not os.path.exists(dialog_folder):
        missing_files.append(f"Dialog folder: {dialog_folder}")
    
    for file_path, file_desc in [
        (class_time_file, "Class time file"),
        (homework_file, "Homework file"),
        (class_schedule_file, "Class schedule file")
    ]:
        if not os.path.exists(file_path):
            missing_files.append(f"{file_desc}: {file_path}")
    
    return missing_files


def organize_dialogs_by_cluster(
    features_df: pd.DataFrame,
    dialog_folder: str,
    output_folder: str,
    cluster_col: str = "cluster",
) -> None:
    """
    根据聚类结果组织对话文件。
    会跳过目标路径与源路径相同的文件。
    """
    import os
    import shutil

    if cluster_col not in features_df.columns:
        print(f"⚠️ '{cluster_col}' column not found — skipping file organization")
        return

    print(f"\n📁 Organizing dialogs by '{cluster_col}'...")

    for cluster_id, group_df in features_df.groupby(cluster_col):
        cluster_folder = os.path.join(output_folder, f"cluster_{cluster_id}")
        os.makedirs(cluster_folder, exist_ok=True)
        print(f"  📦 Creating folder: cluster_{cluster_id} ({len(group_df)} dialogs)")

        for _, row in group_df.iterrows():
            file_name = row.get("file_name")
            if not file_name:
                continue

            src_path = find_file_recursively(
                dialog_folder,
                file_name,
                prefer_exts=PREFERRED_FILE_EXTENSIONS,
                case_insensitive=True,
            )

            if not src_path or not os.path.exists(src_path):
                print(f"⚠️ Source file not found: {file_name}")
                continue

            dst_path = os.path.join(cluster_folder, os.path.basename(src_path))

            # 🚫 跳过源文件与目标文件相同的情况
            try:
                if os.path.samefile(src_path, dst_path):
                    # samefile 在不同系统下兼容判断 inode/路径是否一致
                    print(f"⚠️ Skipped (same file): {file_name}")
                    continue
            except FileNotFoundError:
                # os.path.samefile 要求文件存在，万一目标还没创建就忽略
                pass

            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"⚠️ Failed to copy {file_name}: {e}")


def save_analysis_config(
    output_folder: str,
    dialog_folder: str,
    total_samples: int,
    features_count: int,
    optimal_clusters: int
) -> None:
    """
    保存分析配置信息
    
    Args:
        output_folder: 输出文件夹路径
        dialog_folder: 输入对话文件夹路径
        total_samples: 样本总数
        features_count: 特征数量
        optimal_clusters: 最优聚类数
    """
    config_info = {
        'dialog_folder': dialog_folder,
        'output_folder': output_folder,
        'total_samples': total_samples,
        'features_count': features_count,
        'optimal_clusters': optimal_clusters,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    config_file = os.path.join(output_folder, 'analysis_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Configuration saved to: {config_file}")


def main() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    主函数：执行完整的聚类分析流程
    
    Returns:
        tuple: (聚类后的特征DataFrame, 簇特征均值DataFrame)
    """
    setup_plotting()
    args = parse_arguments()

    # 配置路径
    dialog_folder = args.dialog_folder
    output_folder = os.path.join(dialog_folder, "clustering_results")
    
    # 验证输入文件
    if not os.path.exists(dialog_folder):
        print(f"❌ Error: Dialog folder does not exist: {dialog_folder}")
        sys.exit(1)
    
    # 打印配置信息
    print("=" * 50)
    print("Clustering Analysis Configuration")
    print("=" * 50)
    print(f"Dialog folder:       {dialog_folder}")
    print(f"Output folder:       {output_folder}")
    print(f"Class time file:     {args.class_time_file}")
    print(f"Homework file:       {args.homework_file}")
    print(f"Class schedule file: {args.class_schedule_file}")
    print(f"Class info file:     {args.class_info_file or 'Not provided'}")
    print(f"School info file:    {args.school_info_file or 'Not provided'}")
    print(f"Max K:               {args.max_k}")
    print(f"Variance threshold:  {args.variance_threshold}")
    print("=" * 50)
    
    # 检查必要文件
    missing_files = validate_input_files(
        dialog_folder,
        args.class_time_file,
        args.homework_file,
        args.class_schedule_file
    )
    
    if missing_files:
        print("\n⚠️ Warning: Missing reference files:")
        for missing in missing_files:
            print(f"  - {missing}")
        print("Analysis will continue with available data...\n")
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("Starting Feature Extraction and Clustering Analysis")
    print("=" * 50)
    
    try:
        # 步骤1: 特征提取
        print("\n[1/7] Extracting features...")
        features_df = extract_all_features(
            dialog_folder,
            args.class_time_file,
            args.homework_file,
            args.class_schedule_file,
            school_info_file=args.school_info_file,
            class_info_file=args.class_info_file
        )
        
        if features_df.empty:
            print("❌ Error: No features extracted, terminating program")
            return None, None
        
        print(f"✅ Successfully extracted features from {len(features_df)} dialogs")
        
        # 调试和保存特征
        print("\n[Debug] Checking extracted features...")
        debug_infinite_values(features_df)
        
        features_file = os.path.join(output_folder, 'extracted_features.csv')
        features_df.to_csv(features_file, index=False, encoding='utf-8-sig')
        print(f"💾 Features saved to: {features_file}")
        
        # 显示描述性统计
        feature_columns = [col for col in features_df.columns 
                          if col not in ['file_name', 'class_id']]
        if feature_columns and len(features_df) > 0:
            print("\n📊 Feature Descriptive Statistics:")
            print(features_df[feature_columns].describe())
        
        # 步骤2: PCA分析
        print("\n[2/7] Performing PCA analysis...")
        X_scaled, X_pca, pca, scaler, feature_cols = perform_pca_analysis(
            features_df, output_folder
        )
        
        # 步骤3: 寻找最优聚类数
        print("\n[3/7] Finding optimal number of clusters (elbow method)...")
        optimal_k, X_cluster = find_optimal_clusters_elbow_only(
            X_pca, pca, max_k=args.max_k, output_folder=output_folder
        )
        
        # 步骤4: 执行聚类
        print(f"\n[4/7] Performing clustering analysis (K={optimal_k})...")
        cluster_labels, features_df_clustered, kmeans = perform_clustering(
            X_cluster, optimal_k, features_df, output_folder
        )
        
        clustered_file = os.path.join(output_folder, 'clustered_features.csv')
        features_df_clustered.to_csv(clustered_file, index=False, encoding='utf-8-sig')
        print(f"💾 Clustered features saved to: {clustered_file}")
        
        # 步骤5: 分析簇特征
        print("\n[5/7] Analyzing cluster characteristics...")
        cluster_means = analyze_cluster_characteristics(
            features_df_clustered, feature_cols, output_folder
        )
        
        # 步骤6: 生成可视化
        print("\n[6/7] Generating visualizations...")
        comprehensive_clustering_visualization_all_k(
            X_scaled, X_pca, features_df_clustered, feature_columns,
            DEFAULT_CLUSTER_COLUMNS, output_folder
        )
        
        # 步骤7: 组织对话文件
        print("\n[7/7] Organizing dialogs by cluster...")
        organize_dialogs_by_cluster(
            features_df_clustered, dialog_folder, output_folder
        )
        
        # 保存配置信息
        save_analysis_config(
            output_folder, dialog_folder, len(features_df_clustered),
            len(feature_cols), optimal_k
        )
        
        # 输出最终摘要
        print("\n" + "=" * 50)
        print("Analysis Complete!")
        print("=" * 50)
        print(f"📁 Results saved to:  {output_folder}")
        print(f"📊 Total samples:     {len(features_df_clustered):,}")
        print(f"📈 Features:          {len(feature_cols)}")
        print(f"🎯 Clusters:          {optimal_k}")
        print("=" * 50)
        
        return features_df_clustered, cluster_means
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == '__main__':
    main()