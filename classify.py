import os
import shutil
import pandas as pd

def classify_from_cluster_csv(base_dir, cluster_csv, output_dir="classified"):
    """
    根据已有的聚类结果文件（clustered_features.csv），
    从 base_dir 中复制对应 csv 到分类文件夹下（按 cluster、cluster_k2、cluster_k3、cluster_k4）
    """
    # 读取聚类结果表
    df = pd.read_csv(cluster_csv)

    # 确保输出路径存在
    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        file_name = row["file_name"]
        src_path = None

        # 递归查找文件
        for root, _, files in os.walk(base_dir):
            if file_name in files:
                src_path = os.path.join(root, file_name)
                break

        if src_path is None:
            print(f"⚠️ 未找到文件: {file_name}")
            continue

        # 遍历4个聚类列
        for col in ["cluster", "cluster_k2", "cluster_k3", "cluster_k4"]:
            if col in row and not pd.isna(row[col]):
                cluster_value = str(row[col])
                dest_dir = os.path.join(output_dir, col, cluster_value)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(src_path, os.path.join(dest_dir, file_name))

    print(f"✅ 分类完成！结果已保存到 {output_dir}")


# === 示例用法 ===
classify_from_cluster_csv(
    base_dir="/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/split",
    cluster_csv="/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/split/clustering_results/clustered_features.csv",
    output_dir="/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/clustering_results/classified"
)
