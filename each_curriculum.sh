#!/bin/zsh

# 批量聚类分析脚本
# 遍历指定目录下的所有子文件夹，对每个文件夹执行聚类分析

# 配置参数
BASE_DIR="/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/split"
PYTHON_SCRIPT="/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/code/feature_cluster.py"
LOG_DIR="/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/logs"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 创建日志目录
mkdir -p "$LOG_DIR"

# 获取时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/batch_clustering_$TIMESTAMP.log"

echo "=== 批量聚类分析开始 ===" | tee "$MAIN_LOG"
echo "基础目录: $BASE_DIR" | tee -a "$MAIN_LOG"
echo "Python脚本: $PYTHON_SCRIPT" | tee -a "$MAIN_LOG"
echo "日志目录: $LOG_DIR" | tee -a "$MAIN_LOG"
echo "开始时间: $(date)" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 检查基础目录是否存在
if [[ ! -d "$BASE_DIR" ]]; then
    echo "${RED}❌ 错误: 基础目录不存在: $BASE_DIR${NC}" | tee -a "$MAIN_LOG"
    exit 1
fi

# 检查Python脚本是否存在
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "${RED}❌ 错误: Python脚本不存在: $PYTHON_SCRIPT${NC}" | tee -a "$MAIN_LOG"
    exit 1
fi

# 获取所有子文件夹
subfolders=()
for folder in "$BASE_DIR"/*; do
    if [[ -d "$folder" ]]; then
        folder_name=$(basename "$folder")
        # 排除一些明显不是对话数据的文件夹
        if [[ "$folder_name" != "clustering_results" && 
              "$folder_name" != "logs" && 
              "$folder_name" != "backup" &&
              "$folder_name" != "tmp" ]]; then
            subfolders+=("$folder")
        fi
    fi
done

echo "发现 ${#subfolders[@]} 个待处理文件夹:" | tee -a "$MAIN_LOG"
for folder in "${subfolders[@]}"; do
    echo "  - $(basename "$folder")" | tee -a "$MAIN_LOG"
done
echo "" | tee -a "$MAIN_LOG"

# 统计变量
total_folders=${#subfolders[@]}
success_count=0
failed_count=0
skipped_count=0

# 处理每个文件夹
for i in {1..$total_folders}; do
    folder=${subfolders[$i]}
    folder_name=$(basename "$folder")
    
    echo "${BLUE}📁 [$i/$total_folders] 处理文件夹: $folder_name${NC}" | tee -a "$MAIN_LOG"
    
    # 检查文件夹是否包含CSV文件
    csv_count=$(find "$folder" -name "*.csv" -type f | wc -l)
    
    if [[ $csv_count -eq 0 ]]; then
        echo "${YELLOW}⚠️  跳过: 文件夹中没有CSV文件${NC}" | tee -a "$MAIN_LOG"
        ((skipped_count++))
        continue
    fi
    
    echo "   发现 $csv_count 个CSV文件" | tee -a "$MAIN_LOG"
    
    # 检查是否已经处理过
    result_folder="$folder/clustering_results"
    if [[ -d "$result_folder" && -f "$result_folder/clustered_features.csv" ]]; then
        echo "${YELLOW}⚠️  跳过: 已存在聚类结果${NC}" | tee -a "$MAIN_LOG"
        ((skipped_count++))
        continue
    fi
    
    # 创建单独的日志文件
    folder_log="$LOG_DIR/clustering_${folder_name}_$TIMESTAMP.log"
    
    echo "   开始处理..." | tee -a "$MAIN_LOG"
    start_time=$(date +%s)
    
    # 执行Python脚本
    python "$PYTHON_SCRIPT" "$folder" \
        --max_k 10 \
        --variance_threshold 0.8 \
        > "$folder_log" 2>&1
    
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [[ $exit_code -eq 0 ]]; then
        echo "${GREEN}✅ 成功完成 (用时: ${duration}s)${NC}" | tee -a "$MAIN_LOG"
        ((success_count++))
        
        # 检查输出文件
        if [[ -f "$result_folder/clustered_features.csv" ]]; then
            sample_count=$(tail -n +2 "$result_folder/clustered_features.csv" | wc -l)
            echo "   生成 $sample_count 个聚类样本" | tee -a "$MAIN_LOG"
        fi
        
    else
        echo "${RED}❌ 处理失败 (退出码: $exit_code)${NC}" | tee -a "$MAIN_LOG"
        echo "   详细错误日志: $folder_log" | tee -a "$MAIN_LOG"
        ((failed_count++))
    fi
    
    echo "" | tee -a "$MAIN_LOG"
done

# 输出最终统计
echo "=== 批处理完成 ===" | tee -a "$MAIN_LOG"
echo "完成时间: $(date)" | tee -a "$MAIN_LOG"
echo "处理统计:" | tee -a "$MAIN_LOG"
echo "  总文件夹数: $total_folders" | tee -a "$MAIN_LOG"
echo "  成功处理: $success_count" | tee -a "$MAIN_LOG"
echo "  处理失败: $failed_count" | tee -a "$MAIN_LOG"
echo "  跳过文件夹: $skipped_count" | tee -a "$MAIN_LOG"

# 生成汇总报告
if [[ $success_count -gt 0 ]]; then
    echo "" | tee -a "$MAIN_LOG"
    echo "=== 成功处理的文件夹 ===" | tee -a "$MAIN_LOG"
    
    for folder in "${subfolders[@]}"; do
        result_folder="$folder/clustering_results"
        if [[ -f "$result_folder/clustered_features.csv" ]]; then
            folder_name=$(basename "$folder")
            echo "📊 $folder_name: $result_folder" | tee -a "$MAIN_LOG"
        fi
    done
fi

# 显示日志位置
echo "" | tee -a "$MAIN_LOG"
echo "📝 完整日志: $MAIN_LOG"
echo "📁 日志目录: $LOG_DIR"

# 根据结果设置退出码
if [[ $failed_count -gt 0 ]]; then
    exit 1
else
    exit 0
fi