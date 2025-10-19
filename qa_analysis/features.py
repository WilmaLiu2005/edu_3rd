import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .io_utils import load_reference_data
from .time_utils import get_teaching_week, get_time_to_next_class, get_time_from_last_class
from .homework_utils import get_hours_to_next_assignment, get_hours_since_last_assignment_release
from .feature_utils import replace_inf_with_reasonable_value, contains_copy_keywords, COPY_KEYWORDS
from .config import ANCHOR_DATE


def extract_features_from_dialog(file_path, df_class, df_homework, df_schedule, df_school=None, 
                                 stats=None,
                                 before_course_value=-0.1, after_course_value=1.1):
    """从单个对话文件中提取特征（增强版）
       - 保留原所有特征逻辑
       - 新增: 是否考试周、一天时段、是否周末、是否上课时间内、是否“为什么/怎么/为啥”提问
    """
    # if df_school is None:
        # print("NO\n\n\n")
    try:
        if stats is not None:
            stats['total'] = stats.get('total', 0) + 1
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            if stats is not None:
                stats['File does not exist'] = stats.get('File does not exist', 0) + 1
            return None

        df = pd.read_csv(file_path, encoding='utf-8-sig')
        if df.empty:
            print(f"File is empty: {os.path.basename(file_path)}")
            if stats is not None:
                stats['File is empty'] = stats.get('File is empty', 0) + 1
            return None

        required_columns = ['提问时间', '提问内容', 'AI回复']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"File missing required columns {missing_columns}: {os.path.basename(file_path)}")
            if stats is not None:
                stats['missing columns'] = stats.get('missing columns', 0) + 1
            return None
        df.fillna("", inplace=True)

        file_name = os.path.basename(file_path)

        # === 基本校验 ===
        if '教学班ID' not in df.columns:
            print(f"Warning: File missing class ID column: {file_name}")
            if stats is not None:
                stats['failed'] = stats.get('failed', 0) + 1
            return None
        class_id = df["教学班ID"].iloc[0]

        # === 转换时间 ===
        try:
            df["提问时间"] = pd.to_datetime(df["提问时间"], errors='coerce')
        except Exception as e:
            print(f"Time conversion failed: {file_name} - {e}")
            if stats is not None:
                stats['Time conversion'] = stats.get('Time conversion', 0) + 1
            return None

        if df["提问时间"].isna().any():
            print(f"Invalid QA times in: {file_name}")
            if stats is not None:
                stats['Invalid QA times'] = stats.get('Invalid QA times', 0) + 1
            return None

        # === 班级过滤逻辑（如果提供 class_info_file） ===
        if df_school is not None and isinstance(df_school, dict) and 'df_class_info' in df_school:
            df_class_info = df_school['df_class_info']
            if not df_class_info.empty:
                valid_class_ids = set(df_class_info['教学班ID'].astype(str))
                if str(class_id) not in valid_class_ids:
                    print(f"⏭️ Skipping class_id={class_id} (not in class_info_file): {file_name}")
                    if stats is not None:
                        stats['filtered_by_class_info'] = stats.get('filtered_by_class_info', 0) + 1
                    return None

        # === 课程信息（仍然用 df_class） ===
        class_info = df_class[df_class['教学班ID'] == class_id]
        if class_info.empty:
            print(f"No class info for class_id={class_id}: {file_name}")
            if stats is not None:
                stats['No class info'] = stats.get('No class info', 0) + 1
            return None

        course_start = pd.to_datetime(class_info['起始时间'].iloc[0], errors='coerce')
        course_end = pd.to_datetime(class_info['结束时间'].iloc[0], errors='coerce')
        if pd.isna(course_start) or pd.isna(course_end):
            print(f"Invalid course start/end time for class_id={class_id}: {file_name}")
            if stats is not None:
                stats['Invalid course start/end time'] = stats.get('Invalid course start/end time', 0) + 1
            return None

        qa_min_time = df["提问时间"].min()
        qa_max_time = df["提问时间"].max()

        out_of_range_early = qa_min_time < course_start
        out_of_range_late = qa_max_time > course_end
        if out_of_range_early or out_of_range_late:
            print(f"Dialog outside course window [{course_start}, {course_end}] -> mark and continue: {file_name}")
            if stats is not None:
                stats['out_of_range'] = stats.get('out_of_range', 0) + 1
            return None  # 🚫 直接跳过该文件

        # === 复制题目关键词 ===
        copy_keywords_count = 0
        for q in df["提问内容"]:
            if not isinstance(q, str):
                continue
            for kw in COPY_KEYWORDS:
                copy_keywords_count += q.count(kw)

        has_copy_keywords = int(copy_keywords_count > 0)

        # === 对话统计特征 ===
        qa_turns = len(df)
        is_multi_turn = qa_turns > 1
        if qa_turns > 1:
            total_time = (df["提问时间"].max() - df["提问时间"].min()).total_seconds() / 60
            total_time = max(0, total_time)
        else:
            total_time = 0
        avg_qa_time = total_time / (qa_turns - 1) if qa_turns > 1 and total_time > 0 else 0

        question_lengths = df["提问内容"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        total_question_chars = int(question_lengths.sum())
        avg_question_length = float(question_lengths.mean()) if len(question_lengths) > 0 else 0.0

        if '提问入口' in df.columns:
            if_non_class = int((df["提问入口"] != "班级").any()) if qa_turns > 0 else 0
        else:
            print(f"Warning: File missing entry point column: {file_name}")
            if_non_class = 0

        # === 与作业、课程的时间关系 ===
        hours_to_assignment_list = [get_hours_to_next_assignment(t, class_id, df_homework) for t in df["提问时间"]]
        avg_hours_to_assignment = float(np.mean(hours_to_assignment_list)) if hours_to_assignment_list else 720.0

        hours_since_release_list = [get_hours_since_last_assignment_release(t, class_id, df_homework) for t in df["提问时间"]]
        avg_hours_since_release = float(np.mean(hours_since_release_list)) if hours_since_release_list else 720.0

        total_weeks = max(1, ((course_end - course_start).days // 7) + 1)
        progress_values = [get_teaching_week(t, class_id, df_class) / total_weeks
                           for t in df["提问时间"]
                           if (t >= course_start) and (t <= course_end)]
        qa_start_time = df["提问时间"].min()
        if qa_start_time < course_start:
            course_progress_ratio = float(before_course_value)
        elif qa_start_time > course_end:
            course_progress_ratio = float(after_course_value)
        else:
            course_progress_ratio = float(np.mean(progress_values)) if progress_values else 0.0

        anchor = pd.Timestamp('2025-02-17')
        calendar_week_since_2025_0217 = int(((qa_start_time.normalize() - anchor.normalize()).days // 7) + 1)

        hours_to_next_class_list = [get_time_to_next_class(t, class_id, df_schedule) for t in df["提问时间"]]
        avg_hours_to_next_class = float(np.mean(hours_to_next_class_list)) if hours_to_next_class_list else float('inf')

        hours_from_last_class_list = [get_time_from_last_class(t, class_id, df_schedule) for t in df["提问时间"]]
        avg_hours_from_last_class = float(np.mean(hours_from_last_class_list)) if hours_from_last_class_list else float('inf')

        # === 原始特征 ===
        features = {
            "file_name": file_name,
            "class_id": class_id,
            "qa_turns": int(qa_turns),
            "is_multi_turn": bool(is_multi_turn),
            "total_time_minutes": float(total_time),
            "avg_qa_time_minutes": float(avg_qa_time),
            "total_question_chars": int(total_question_chars),
            "avg_question_length": float(avg_question_length),
            "if_non_class": int(if_non_class),
            "avg_hours_to_assignment": float(avg_hours_to_assignment),
            "avg_hours_since_release": float(avg_hours_since_release),
            "course_progress_ratio": float(course_progress_ratio),
            "calendar_week_since_2025_0217": int(calendar_week_since_2025_0217),
            "hours_to_next_class": float(avg_hours_to_next_class),
            "hours_from_last_class": float(avg_hours_from_last_class),
            "has_copy_keywords": int(has_copy_keywords),
            "copy_keywords_count": int(copy_keywords_count)
        }

        # === ✨ 新增特征区域 ===
        # 1️⃣ 一天时段
        # 一天中的时间（单位：小时，精确到分钟）
        day_period = qa_start_time.hour + qa_start_time.minute / 60.0

        # 2️⃣ 是否周末
        is_weekend = int(qa_start_time.weekday() >= 5)

        # 3️⃣ 是否考试周（最后两周）
        current_week = get_teaching_week(qa_start_time, class_id, df_class)
        is_exam_week = int(current_week >= total_weeks - 1)

        # 5️⃣ 是否在上课时间内
        def check_in_class_time(qa_time, class_id, df_schedule):
            schedule = df_schedule[df_schedule["教学班ID"] == class_id]
            for _, row in schedule.iterrows():
                start = pd.to_datetime(row["开课时间"], errors="coerce")
                end = pd.to_datetime(row["结课时间"], errors="coerce")
                if pd.notna(start) and pd.notna(end) and start <= qa_time <= end:
                    return True
            return False

        is_in_class_time = int(any(check_in_class_time(t, class_id, df_schedule)
                                   for t in df["提问时间"] if pd.notna(t)))

        # 6️⃣ “为什么/为啥/怎么”类问题
        question_texts = " ".join(df["提问内容"].astype(str))
        question_type_why_how = int(any(kw in question_texts for kw in ["为什么", "为啥", "怎么"]))# 看一下数据

        # 合并新增特征
        features.update({
            "is_exam_week": int(is_exam_week),
            "day_period": day_period,
            "is_weekend": int(is_weekend),
            "is_in_class_time": int(is_in_class_time),
            "question_type_why_how": int(question_type_why_how)
        })

        # === 数值检查 ===
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
        return None

def plot_feature_histograms(df, features=None, bins=50, save_dir=None, figsize=(7, 4), stats_file="feature_stats.csv"):
    """
    绘制特征分布图，并保存每个特征的 min/max/mean/var。
    
    - 数值连续变量绘制直方图；
    - 二元变量（0/1）绘制柱状图；
    - 自动识别特征列。
    
    返回：保存的图片路径列表
    """
    if save_dir is None:
        save_dir = os.path.abspath("histograms")
    os.makedirs(save_dir, exist_ok=True)

    # 自动识别数值型特征列
    if features is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = numeric_cols
        print(f"🧭 自动检测到 {len(features)} 个数值型特征：{features}")

    saved_files = []
    stats_list = []

    for feature in features:
        if feature not in df.columns:
            print(f"Skipping missing feature: {feature}")
            continue

        series = pd.to_numeric(df[feature], errors='coerce').dropna()
        if series.empty:
            print(f"No data for feature: {feature} -> skip")
            continue

        # ==== 保存统计信息 ====
        stats = {
            "feature": feature,
            "min": series.min(),
            "max": series.max(),
            "mean": series.mean(),
            "median": series.median(),
            "variance": series.var()
        }
        stats_list.append(stats)

        # ==== 绘图 ====
        plt.figure(figsize=figsize)
        unique_vals = sorted(series.unique())

        # 二元变量绘制柱状图
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
            counts = series.value_counts().sort_index()
            plt.bar(counts.index.astype(str), counts.values, color="skyblue", edgecolor="black")
            plt.title(f"{feature} (binary: {unique_vals})")
            plt.xlabel(feature)
            plt.ylabel("Count")
        else:
            plt.hist(series, bins=bins, color="steelblue", edgecolor="black", alpha=0.75)
            plt.axvline(series.mean(), color='red', linestyle='--', label=f"Mean={series.mean():.2f}")
            plt.axvline(series.median(), color='green', linestyle=':', label=f"Median={series.median():.2f}")
            plt.legend()
            plt.title(f"{feature} (n={len(series)})")
            plt.xlabel(feature)
            plt.ylabel("Count")

        fname = os.path.join(save_dir, f"{feature}_dist.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        saved_files.append(fname)
        print(f"✅ Saved plot: {fname}")

    # ==== 保存统计信息到 CSV ====
    stats_df = pd.DataFrame(stats_list)
    stats_path = os.path.join(save_dir, stats_file)
    stats_df.to_csv(stats_path, index=False)
    print(f"\n📊 已保存 {len(saved_files)} 张特征分布图，并保存统计信息到：{stats_path}")

    return saved_files, stats_df

def extract_all_features(dialog_folder, class_time_file, homework_file, class_schedule_file, 
                         school_info_file=None, class_info_file=None, plot_histograms=True):
    """提取所有对话文件的特征（并可选绘制直方图）"""
    print("Loading reference data...")
    try:
        df_class, df_homework, df_schedule, df_school = load_reference_data(
            class_time_file, 
            homework_file, 
            class_schedule_file, 
            school_info_file=school_info_file,
            class_info_file=class_info_file   # ✅ 新增
        )
    except Exception as e:
        print(f"Failed to load reference data: {e}")
        df_class, df_homework, df_schedule, df_school= pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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

        features = extract_features_from_dialog(
            file_path,
            df_class,
            df_homework,
            df_schedule,
            df_school=df_school,  # ✅ 新增
            stats=stats
        )

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
        "out_of_range_dialogs": out_of_range,        # 因超出教学周被标记的对话数（不再筛掉）
        "successful_feature_rows": int(len(features_list)),
        "stats": stats
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
                #print(f"Found {inf_count} infinite values in {feature}")
                features_df[feature] = replace_inf_with_reasonable_value(
                    features_df[feature], multiplier=1.5)
                #print(f"Replaced with max finite value * 1.5 = {features_df[feature].max():.2f}")
    # 如果需要，绘制并保存直方图
    if plot_histograms:
        hist_dir = os.path.join(stats_dir, 'histograms_before_log')
        print(f"Plotting histograms to: {hist_dir}")
        saved = plot_feature_histograms(features_df, save_dir=hist_dir)
        print(f"Saved {len(saved)} histograms")
    # 2️⃣ 再进行 log(1+x) 变换
    log_features = [
        "avg_hours_since_release", "avg_hours_to_assignment",
        "avg_qa_time_minutes", "avg_question_length",
        "copy_keywords_count", "course_progress_ratio",
        "qa_turns", "total_question_chars", "total_time_minutes","hours_from_last_class","hours_to_next_class"
    ]
    for feat in log_features:
        if feat in features_df.columns:
            features_df[feat] = np.log1p(features_df[feat])
    # 如果需要，绘制并保存直方图
    if plot_histograms:
        hist_dir = os.path.join(stats_dir, 'histograms_after_log')
        print(f"Plotting histograms to: {hist_dir}")
        saved = plot_feature_histograms(features_df, save_dir=hist_dir)
        print(f"Saved {len(saved)} histograms")

    return features_df
