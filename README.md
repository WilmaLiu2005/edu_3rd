# 学堂在线QA对话聚类分析工具

一个用于分析学堂在线教育平台QA对话数据的完整工具链，提供特征提取、主成分分析(PCA)、聚类分析和可视化功能。

## 🎯 主要功能

- **智能特征提取**：从QA对话文件中提取22维特征，包括对话统计、时间关系、学习行为等
- **自适应PCA降维**：根据方差解释率自动选择最优主成分数量
- **多方法聚类**：结合肘部法则的三种检测方法确定最优聚类数
- **丰富的可视化**：生成PCA图、t-SNE图、聚类热力图等多种可视化结果
- **结果组织**：自动按聚类结果组织原始对话文件

## 📊 特征体系

### 基础对话特征
- `qa_turns`: QA轮次数
- `is_multi_turn`: 是否多轮对话
- `total_time_minutes`: 总对话时长(分钟)
- `avg_qa_time_minutes`: 平均QA间隔时间
- `total_question_chars`: 问题总字符数
- `avg_question_length`: 平均问题长度

### 学习行为特征
- `if_non_class`: 是否非班级入口提问
- `has_copy_keywords`: 是否包含复制关键词
- `copy_keywords_count`: 复制关键词数量
- `question_type_why_how`: 是否为探究性问题

### 时间关系特征
- `avg_hours_to_assignment`: 距离下次作业平均小时数
- `avg_hours_since_release`: 距离上次作业发布平均小时数
- `course_progress_ratio`: 课程进度比例
- `calendar_week_since_2025_0217`: 自然周编号
- `hours_to_next_class`: 距离下次上课小时数
- `hours_from_last_class`: 距离上次下课小时数

### 情境特征
- `is_exam_week`: 是否考试周
- `day_period`: 一天中的时段(0-24小时)
- `is_weekend`: 是否周末
- `is_in_class_time`: 是否在上课时间内

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
```

### 安装依赖

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 基本使用

```bash
cd code
python run.py \
    --dialog_folder "对话文件夹路径" \
    --class_time_file "课程时间文件.csv" \
    --homework_file "作业信息文件.csv" \
    --class_schedule_file "课程表文件.csv" \
    --max_k 10 \
    --variance_threshold 0.8
```

### 可选参数

```bash
python run.py \
    --dialog_folder "/path/to/dialogs" \
    --class_time_file "/path/to/class_time.csv" \
    --homework_file "/path/to/homework.csv" \
    --class_schedule_file "/path/to/schedule.csv" \
    --school_info_file "/path/to/school_info.csv" \
    --class_info_file "/path/to/class_info.csv" \
    --max_k 15 \
    --variance_threshold 0.85
```

## 📁 数据格式要求

### 对话文件格式
每个对话CSV文件需包含以下列：
- `提问时间`: 问题提出时间 (YYYY-MM-DD HH:MM:SS)
- `提问内容`: 问题文本内容
- `AI回复`: AI回复内容
- `教学班ID`: 所属教学班标识
- `提问入口`: 提问来源渠道(可选)

### 课程时间文件
- `教学班ID`: 教学班标识
- `起始时间`: 课程开始时间
- `结束时间`: 课程结束时间

### 作业信息文件
- `教学班ID`: 教学班标识
- `发布时间`: 作业发布时间
- `提交截止时间`: 作业截止时间

### 课程表文件
- `教学班ID`: 教学班标识
- `开课时间`: 上课开始时间
- `结课时间`: 上课结束时间

## 📈 输出结果

分析完成后，在输入文件夹下会生成 `clustering_results` 目录，包含：

### 核心结果文件
- `extracted_features.csv`: 提取的原始特征数据
- `clustered_features.csv`: 包含聚类标签的特征数据
- `clustered_features_k{n}.csv`: 不同K值的聚类结果
- `cluster_statistics.csv`: 各聚类的详细统计信息

### PCA分析结果
- `pca_explained_variance_ratio.png`: 主成分方差解释率
- `pca_cumulative_explained_variance.png`: 累积方差解释率
- `pca_feature_loadings_heatmap.png`: 特征载荷热力图
- `pca_2d_projection.png`: PCA二维投影图

### 聚类分析结果
- `adaptive_elbow_method_analysis.png`: 肘部法则分析图
- `pca_cluster.png` / `tsne_cluster.png`: 聚类可视化图
- `cluster_size_cluster.png`: 聚类大小分布
- `feature_heatmap_cluster.png`: 聚类特征热力图

### 特征分布图
- `histograms_before_log/`: log变换前的特征分布
- `histograms_after_log/`: log变换后的特征分布
- 每个特征的单独分布图和统计信息

### 组织化文件
- `cluster_0/`, `cluster_1/`, ...: 按聚类结果组织的原始对话文件
- `analysis_config.json`: 分析配置和元信息
- `dialog_stats.json`: 对话处理统计信息

## 🔧 高级用法

### 编程接口使用

```python
from qa_analysis.features import extract_all_features
from qa_analysis.pca_utils import perform_pca_analysis
from qa_analysis.clustering import find_optimal_clusters_elbow_only, perform_clustering

# 特征提取
features_df = extract_all_features(
    dialog_folder="对话文件夹",
    class_time_file="课程时间.csv",
    homework_file="作业信息.csv", 
    class_schedule_file="课程表.csv"
)

# PCA分析
X_scaled, X_pca, pca, scaler, feature_cols = perform_pca_analysis(
    features_df, "输出文件夹"
)

# 聚类分析
optimal_k, X_cluster = find_optimal_clusters_elbow_only(
    X_pca, pca, max_k=10, output_folder="输出文件夹"
)

cluster_labels, features_clustered, kmeans = perform_clustering(
    X_cluster, optimal_k, features_df, "输出文件夹"
)
```

### 自定义特征选择

```python
# 选择特定特征进行分析
custom_features = [
    'qa_turns', 'avg_question_length', 'course_progress_ratio',
    'is_weekend', 'question_type_why_how'
]

# 在clustering.py中修改FEATURE_COLUMNS常量
```

## 🛠️ 技术架构

```
qa_analysis/
├── cli.py              # 命令行接口
├── features.py         # 特征提取核心模块
├── pca_utils.py       # PCA降维分析
├── clustering.py      # 聚类分析
├── io_utils.py        # 数据加载工具
├── time_utils.py      # 时间计算工具
├── homework_utils.py  # 作业相关工具
├── feature_utils.py   # 特征处理工具
└── config.py          # 配置文件
```

### 核心算法
- **特征提取**: 多维度特征工程，包含对话、时间、行为、情境四类特征
- **PCA降维**: 自适应主成分选择，根据方差解释率自动确定降维维度
- **聚类算法**: K-Means + 肘部法则三重检测(二阶导数、距离法、斜率变化)
- **可视化**: PCA/t-SNE投影 + 热力图 + 分布图等多维度展示

## ⚠️ 注意事项

1. **数据质量**: 确保对话文件包含必需的时间和内容列
2. **内存使用**: 大量文件时可能需要较大内存，建议16GB+
3. **时间范围**: 系统会自动过滤超出课程时间范围的对话
4. **编码格式**: 所有CSV文件应使用UTF-8编码

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进此工具：

1. Fork 这个仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个Pull Request

## 📜 许可证

本项目遵循 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 创建GitHub Issue
- 发送邮件至项目维护者

---

*该工具专为学堂在线教育平台的QA对话数据分析而设计，支持大规模教育数据的深度挖掘和模式发现。*
