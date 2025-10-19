import os, sys, json
import pandas as pd
from .config import setup_plotting
from .features import extract_all_features
from .pca_utils import perform_pca_analysis
from .clustering import find_optimal_clusters_elbow_only, perform_clustering, analyze_cluster_characteristics, comprehensive_clustering_visualization_all_k, plot_cluster_feature_heatmap
from .feature_utils import debug_infinite_values

def find_file_recursively(root_dir, file_name, prefer_exts=('.csv', '.txt'), case_insensitive=True):
    """
    在 root_dir 下递归查找 file_name。
    - file_name 可以带或不带后缀
    - 优先匹配 prefer_exts 顺序的后缀
    - 大小写不敏感匹配（可配置）
    """
    if not file_name or not isinstance(file_name, str):
        return None

    # 1) 直接传入绝对路径或相对路径已存在
    if os.path.isabs(file_name) and os.path.exists(file_name):
        return file_name
    direct = os.path.join(root_dir, file_name)
    if os.path.exists(direct):
        return direct

    target = os.path.basename(file_name)
    tbase, text = os.path.splitext(target)
    tname_cmp = target.lower() if case_insensitive else target
    tbase_cmp = tbase.lower() if case_insensitive else tbase

    best_path = None
    best_rank = float('inf')

    for r, _, files in os.walk(root_dir):
        for f in files:
            f_cmp = f.lower() if case_insensitive else f
            if text:  # 目标包含后缀：精确匹配文件名
                if f_cmp == tname_cmp:
                    return os.path.join(r, f)
            else:     # 目标不含后缀：按基名匹配并根据 prefer_exts 选优
                fbase, fext = os.path.splitext(f)
                fbase_cmp = fbase.lower() if case_insensitive else fbase
                if fbase_cmp == tbase_cmp:
                    rank = prefer_exts.index(fext.lower()) if fext.lower() in prefer_exts else len(prefer_exts) + 1
                    if rank < best_rank:
                        best_rank = rank
                        best_path = os.path.join(r, f)
                        if best_rank == 0:  # 找到首选后缀，可提前返回
                            return best_path

    return best_path

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dialog_folder', required=True)
    parser.add_argument('--class_time_file', required=True) # 整个课程的开始和结束时间
    parser.add_argument('--homework_file', required=True)
    parser.add_argument('--class_schedule_file', required=True)# 开课时间
    parser.add_argument('--school_info_file', required=False, help="Path to school basic info CSV (optional)") # 学校信息
    parser.add_argument('--class_info_file', required=False, help="Path to class info CSV (optional)") # 班级信息
    parser.add_argument('--max_k', type=int, default=10)
    parser.add_argument('--variance_threshold', type=float, default=0.8)
    return parser.parse_args()

def main():
    setup_plotting()
    args = parse_arguments()

    # 配置路径
    dialog_folder = args.dialog_folder
    class_time_file = args.class_time_file
    homework_file = args.homework_file
    class_schedule_file = args.class_schedule_file
    school_info_file = args.school_info_file if args.school_info_file else ""
    class_info_file = args.class_info_file if args.class_info_file else ""
    
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
    print(f"Class info file: {class_info_file}")
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
    # 定义要可视化的聚类列
    cluster_cols_list = ['cluster', 'cluster_k2', 'cluster_k3', 'cluster_k4']
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
        features_df = extract_all_features(
            dialog_folder, 
            class_time_file, 
            homework_file, 
            class_schedule_file, 
            school_info_file=school_info_file,
            class_info_file=class_info_file  # ✅ 新增传入
        )
        
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
        
        # 7. 根据聚类结果分类保存原始对话文件
        print("\n7. Organizing dialogs by cluster...")
        cluster_col = 'cluster'  # 主要聚类标签列

        if cluster_col in features_df_clustered.columns:
            for cluster_id, group_df in features_df_clustered.groupby(cluster_col):
                cluster_folder = os.path.join(output_folder, f"cluster_{cluster_id}")
                os.makedirs(cluster_folder, exist_ok=True)

                # 统计信息
                print(f"📁 Creating folder: cluster_{cluster_id} ({len(group_df)} dialogs)")

                # 遍历每个样本，将原始对话文件复制过去
                for _, row in group_df.iterrows():
                    file_name = row.get('file_name')
                    if not file_name:
                        continue

                    # 确定源文件路径
                    src_path = src_path = find_file_recursively(
                        dialog_folder,
                        file_name,
                        prefer_exts=('.csv', '.txt'),  # 需要严格只要 csv 就改为 ('.csv',)
                        case_insensitive=True
                    )

                    if src_path and os.path.exists(src_path):
                        dst_path = os.path.join(cluster_folder, os.path.basename(src_path))
                        try:
                            import shutil
                            shutil.copy2(src_path, dst_path)
                        except Exception as e:
                            print(f"⚠️ Failed to copy {file_name} -> cluster_{cluster_id}: {e}")
                    else:
                        print(f"⚠️ Source file not found for {file_name}")
        else:
            print("⚠️ 'cluster' column not found in clustered features — skipping file organization step.")
        
        # 6. 综合可视化
        print("\n6. Generating visualizations...")
        comprehensive_clustering_visualization_all_k(
            X_scaled, X_pca, features_df_clustered, feature_columns,
            cluster_cols_list, output_folder
        )

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