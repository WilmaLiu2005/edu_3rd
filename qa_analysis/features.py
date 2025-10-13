import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .io_utils import load_reference_data
from .time_utils import get_teaching_week, get_time_to_next_class, get_time_from_last_class
from .homework_utils import get_hours_to_next_assignment, get_hours_since_last_assignment_release
from .feature_utils import replace_inf_with_reasonable_value, contains_copy_keywords
from .config import ANCHOR_DATE


def extract_features_from_dialog(file_path, df_class, df_homework, df_schedule, stats=None,
                                  before_course_value=-0.1, after_course_value=1.1):
    """从单个对话文件中提取特征（修改版）
       - 不再直接筛掉超出教学周范围的对话
       - 对于在课程开始前的对话，course_progress_ratio 设为 before_course_value
       - 对于在课程结束后的对话，course_progress_ratio 设为 after_course_value
       - 可选 stats 计数 out_of_range 比例（仍会计数，但不会返回 None）
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

        # 若任一 QA 时间不在课程区间内，不再直接筛掉该对话；改为记录并在后面设置 course_progress_ratio 的默认值
        out_of_range_early = qa_min_time < course_start
        out_of_range_late = qa_max_time > course_end
        if out_of_range_early or out_of_range_late:
            print(f"Dialog outside course window [{course_start}, {course_end}] -> mark and continue: {file_name}")
            if stats is not None:
                stats['out_of_range'] = stats.get('out_of_range', 0) + 1
            # 继续处理，不 return

        # 判定对话中是否出现“复制题目”相关关键词
        keyword_flags = []
        for ask_text in df["提问内容"]:
            keyword_flags.append(contains_copy_keywords(ask_text))

        has_copy_keywords = int(any(keyword_flags))
        copy_keywords_count = int(sum(keyword_flags))

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
        # 12. 对话位于教学进度的比例（0-1） — 仅对处于课程区间内的时间点计算
        total_weeks = max(1, ((course_end - course_start).days // 7) + 1)
        progress_values = []
        for t in df["提问时间"]:
            # 仅针对落在 course_start <= t <= course_end 的点计算教学周
            if (t >= course_start) and (t <= course_end):
                wk = get_teaching_week(t, class_id, df_class)
                if wk > 0:
                    progress_values.append(wk / total_weeks)
        # 根据对话起始时间决定缺省值（若在区间之外）
        qa_start_time = df["提问时间"].min()
        if qa_start_time < course_start:
            course_progress_ratio = float(before_course_value)
        elif qa_start_time > course_end:
            course_progress_ratio = float(after_course_value)
        else:
            course_progress_ratio = float(np.mean(progress_values)) if progress_values else 0.0
        # 12.5 对话发生的自然周（以 2025-02-17 为第1周的起点，取对话最早一次提问所在周）
        anchor = pd.Timestamp('2025-02-17')  # 周一
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
        features.update({
            "has_copy_keywords": int(has_copy_keywords),   # 二元变量：对话是否出现关键词
            "copy_keywords_count": int(copy_keywords_count)  # 可选：命中条数，便于后续调参/分析
        })
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


def plot_feature_histograms(df, features=None, bins=50, save_dir=None, figsize=(7, 4)):
    """为给定的数值特征画直方图并保存为 PNG 文件。

    参数:
        df: DataFrame
        features: 要绘制的特征列表（None 则使用默认列表）
        bins: 直方图的箱数
        save_dir: 保存目录（None 则在当前目录下创建 histograms/）
        figsize: 每张图的大小
    返回:
        保存的文件路径列表
    """
    if save_dir is None:
        save_dir = os.path.abspath("histograms")
    os.makedirs(save_dir, exist_ok=True)

    default_features = [
        "qa_turns", "total_time_minutes", "avg_qa_time_minutes",
        "total_question_chars", "avg_question_length",
        "avg_hours_to_assignment", "avg_hours_since_release",
        "course_progress_ratio", "calendar_week_since_2025_0217",
        "hours_to_next_class", "hours_from_last_class",
        "copy_keywords_count"
    ]
    if features is None:
        features = default_features

    saved_files = []
    for feature in features:
        if feature not in df.columns:
            print(f"Skipping missing feature: {feature}")
            continue
        series = pd.to_numeric(df[feature], errors='coerce').dropna()
        if series.empty:
            print(f"No data for feature: {feature} -> skip")
            continue
        plt.figure(figsize=figsize)
        plt.hist(series, bins=bins)
        plt.title(f"Histogram of {feature} (n={len(series)})")
        plt.xlabel(feature)
        plt.ylabel("Count")
        fname = os.path.join(save_dir, f"{feature}_hist.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        saved_files.append(fname)
        print(f"Saved histogram: {fname}")
    return saved_files


def extract_all_features(dialog_folder, class_time_file, homework_file, class_schedule_file, plot_histograms=True):
    """提取所有对话文件的特征（并可选绘制直方图）"""
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
        "out_of_range_dialogs": out_of_range,        # 因超出教学周被标记的对话数（不再筛掉）
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
        hist_dir = os.path.join(stats_dir, 'histograms')
        print(f"Plotting histograms to: {hist_dir}")
        saved = plot_feature_histograms(features_df, save_dir=hist_dir)
        print(f"Saved {len(saved)} histograms")

    return features_df
