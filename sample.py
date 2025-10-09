import pandas as pd
import os
import random
import shutil

def sample_and_copy_cluster_files(csv_file_path, search_directory, output_dir="cluster_conversation_samples", sample_size=50):
    """
    简化版：为每个聚类随机取样50个文件并复制
    """
    # 读取聚类结果
    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
    
    # 确保cluster是整数类型
    df['cluster'] = df['cluster'].astype(int)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子保证可重复性
    random.seed(42)
    
    print(f"=== 开始聚类文件采样 ===")
    print(f"目标：每个聚类最多 {sample_size} 个样本文件")
    print(f"搜索目录: {search_directory}")
    print(f"输出目录: {output_dir}")
    
    # 建立文件名到路径的映射（一次性搜索）
    print("建立文件映射...")
    filename_to_path = {}
    
    for root, dirs, files in os.walk(search_directory):
        for file in files:
            if file.endswith('.csv'):
                # 排除一些明显不是对话的文件
                if not any(keyword in file.lower() for keyword in 
                          ['feature', 'cluster', 'result', 'analysis', 'summary']):
                    filename_to_path[file] = os.path.join(root, file)
    
    print(f"找到 {len(filename_to_path)} 个潜在对话文件")
    
    # 处理每个聚类
    clusters = sorted(df['cluster'].unique())
    total_copied = 0
    
    for cluster_id in clusters:
        print(f"\n处理聚类 {cluster_id}:")
        
        # 获取该聚类的所有文件
        cluster_data = df[df['cluster'] == cluster_id].copy()
        cluster_files = cluster_data['file_name'].tolist()
        
        print(f"  聚类总文件数: {len(cluster_files)}")
        
        # 找到实际存在的文件
        available_files = []
        for filename in cluster_files:
            if filename in filename_to_path:
                available_files.append({
                    'filename': filename,
                    'source_path': filename_to_path[filename],
                    'cluster_info': cluster_data[cluster_data['file_name'] == filename].iloc[0].to_dict()
                })
        
        print(f"  找到存在的文件: {len(available_files)}")
        
        # 随机采样
        if len(available_files) > sample_size:
            sampled_files = random.sample(available_files, sample_size)
            print(f"  随机采样: {sample_size} 个文件")
        else:
            sampled_files = available_files
            print(f"  使用全部: {len(sampled_files)} 个文件")
        
        if not sampled_files:
            print(f"  ⚠️ 聚类 {cluster_id} 没有找到任何文件")
            continue
        
        # 创建聚类文件夹
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # 复制文件
        copied_count = 0
        failed_count = 0
        
        for file_info in sampled_files:
            try:
                src_path = file_info['source_path']
                dst_path = os.path.join(cluster_dir, file_info['filename'])
                
                shutil.copy2(src_path, dst_path)
                copied_count += 1
                
            except Exception as e:
                print(f"    复制失败 {file_info['filename']}: {e}")
                failed_count += 1
        
        print(f"  ✅ 成功复制: {copied_count} 个文件")
        if failed_count > 0:
            print(f"  ❌ 复制失败: {failed_count} 个文件")
        
        total_copied += copied_count
        
        # 保存该聚类的文件信息
        if copied_count > 0:
            save_cluster_info(sampled_files, cluster_dir, cluster_id)
    
    print(f"\n=== 采样完成 ===")
    print(f"总共复制了 {total_copied} 个对话文件")
    print(f"结果保存在: {output_dir}")
    
    return total_copied

def save_cluster_info(sampled_files, cluster_dir, cluster_id):
    """保存聚类信息到文件"""
    
    # 准备信息数据
    info_data = []
    for file_info in sampled_files:
        info_row = file_info['cluster_info'].copy()
        info_row['source_path'] = file_info['source_path']
        info_data.append(info_row)
    
    # 保存为CSV
    info_df = pd.DataFrame(info_data)
    info_file = os.path.join(cluster_dir, f"cluster_{cluster_id}_info.csv")
    info_df.to_csv(info_file, index=False, encoding='utf-8-sig')
    
    # 保存简单的文件列表
    file_list = [f['filename'] for f in sampled_files]
    list_file = os.path.join(cluster_dir, f"file_list.txt")
    with open(list_file, 'w', encoding='utf-8') as f:
        f.write(f"聚类 {cluster_id} 采样文件列表\n")
        f.write(f"采样时间: {pd.Timestamp.now()}\n")
        f.write(f"总文件数: {len(file_list)}\n\n")
        for i, filename in enumerate(file_list, 1):
            f.write(f"{i:2d}. {filename}\n")

def quick_preview_samples(output_dir, num_preview=2):
    """快速预览采样结果"""
    
    print(f"\n=== 采样结果预览 ===")
    
    cluster_dirs = [d for d in os.listdir(output_dir) 
                   if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('cluster_')]
    
    cluster_dirs.sort()
    
    for cluster_dir_name in cluster_dirs:
        cluster_path = os.path.join(output_dir, cluster_dir_name)
        csv_files = [f for f in os.listdir(cluster_path) 
                    if f.endswith('.csv') and not f.endswith('_info.csv')]
        
        print(f"\n📁 {cluster_dir_name}: {len(csv_files)} 个对话文件")
        
        # 预览几个文件的内容
        for i, filename in enumerate(csv_files[:num_preview]):
            file_path = os.path.join(cluster_path, filename)
            try:
                df_dialog = pd.read_csv(file_path, encoding='utf-8-sig')
                print(f"  📄 {filename}:")
                print(f"     对话轮次: {len(df_dialog)}")
                
                if len(df_dialog) > 0 and '提问内容' in df_dialog.columns:
                    first_question = str(df_dialog.iloc[0]['提问内容'])[:80]
                    print(f"     首个问题: {first_question}...")
                
            except Exception as e:
                print(f"     ⚠️ 读取失败: {e}")

def generate_sampling_report(output_dir):
    """生成采样报告"""
    
    print(f"\n=== 生成采样报告 ===")
    
    report_data = []
    total_files = 0
    
    cluster_dirs = [d for d in os.listdir(output_dir) 
                   if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('cluster_')]
    
    for cluster_dir_name in sorted(cluster_dirs):
        cluster_id = cluster_dir_name.replace('cluster_', '')
        cluster_path = os.path.join(output_dir, cluster_dir_name)
        
        # 统计文件数
        csv_files = [f for f in os.listdir(cluster_path) 
                    if f.endswith('.csv') and not f.endswith('_info.csv')]
        
        file_count = len(csv_files)
        total_files += file_count
        
        report_data.append({
            'cluster_id': int(cluster_id),
            'sampled_files': file_count,
            'folder_path': cluster_path
        })
    
    # 保存报告
    report_df = pd.DataFrame(report_data)
    report_file = os.path.join(output_dir, 'sampling_report.csv')
    report_df.to_csv(report_file, index=False, encoding='utf-8-sig')
    
    print(f"采样报告已保存: {report_file}")
    print(f"总采样文件数: {total_files}")
    
    return report_df

# 主函数：简化版
def main_simplified():
    """简化版主函数"""
    
    # 配置参数
    clustered_features_file = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/split/clustering_results/clustered_features.csv"
    search_directory = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/split"
    output_directory = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/cluster_conversation_samples"
    sample_size = 50
    
    # 检查输入文件
    if not os.path.exists(clustered_features_file):
        print(f"❌ 聚类特征文件不存在: {clustered_features_file}")
        return
    
    if not os.path.exists(search_directory):
        print(f"❌ 搜索目录不存在: {search_directory}")
        return
    
    try:
        # 执行采样和复制
        total_copied = sample_and_copy_cluster_files(
            csv_file_path=clustered_features_file,
            search_directory=search_directory,
            output_dir=output_directory,
            sample_size=sample_size
        )
        
        if total_copied > 0:
            # 预览结果
            quick_preview_samples(output_directory, num_preview=2)
            
            # 生成报告
            report_df = generate_sampling_report(output_directory)
            
            print(f"\n🎉 采样完成！")
            print(f"   采样文件夹: {output_directory}")
            print(f"   可以开始定性分析了")
        else:
            print("❌ 没有成功复制任何文件，请检查文件路径")
            
    except Exception as e:
        print(f"❌ 执行过程中出错: {e}")
        import traceback
        traceback.print_exc()

# 超简化版本：如果上面还有问题
def ultra_simple_sampling():
    """超简化版本"""
    
    clustered_features_file = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/split/clustering_results/clustered_features.csv"
    search_directory = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/split"
    
    # 读取聚类结果
    df = pd.read_csv(clustered_features_file, encoding='utf-8-sig')
    df['cluster'] = df['cluster'].astype(int)
    
    # 只显示统计信息，不复制文件
    print("=== 聚类文件统计 ===")
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_files = df[df['cluster'] == cluster_id]['file_name'].tolist()
        sample_count = min(50, len(cluster_files))
        
        if len(cluster_files) > 50:
            sampled_files = random.sample(cluster_files, 50)
        else:
            sampled_files = cluster_files
        
        print(f"\n聚类 {cluster_id}:")
        print(f"  总文件数: {len(cluster_files)}")
        print(f"  采样文件数: {sample_count}")
        print(f"  示例文件:")
        
        for i, filename in enumerate(sampled_files[:3]):
            print(f"    {i+1}. {filename}")
        
        if len(sampled_files) > 3:
            print(f"    ... 还有 {len(sampled_files)-3} 个文件")
    
    print(f"\n如需复制文件，请确认搜索目录包含所需文件")

if __name__ == "__main__":
    print("选择执行模式:")
    print("1. 完整采样并复制文件")
    print("2. 仅显示采样统计（不复制文件）")
    
    choice = input("请选择 (1 或 2): ").strip()
    
    if choice == "1":
        main_simplified()
    else:
        ultra_simple_sampling()