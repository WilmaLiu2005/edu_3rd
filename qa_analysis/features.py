import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .io_utils import load_reference_data
from .time_utils import (
    get_teaching_week,
    get_time_to_next_class,
    get_time_from_last_class,
)
from .homework_utils import (
    get_hours_to_next_assignment,
    get_hours_since_last_assignment_release,
)
from .feature_utils import replace_inf_with_reasonable_value, contains_copy_keywords, COPY_KEYWORDS
from .config import ANCHOR_DATE


def extract_features_from_dialog(
    file_path,
    df_class,
    df_homework,
    df_schedule,
    df_school=None,
    stats=None,
    before_course_value=-0.1,
    after_course_value=1.1,
):
    """
    从单个对话文件中提取特征（增强版）

    - 保留原所有特征逻辑
    - 新增: 是否考试周、一天时段、是否周末、是否上课时间内、是否“为什么/怎么/为啥”提问
    """
    try:
        # === 初始化统计 ===
        if stats is not None:
            stats["total"] = stats.get("total", 0) + 1

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            if stats is not None:
                stats["File not found"] = stats.get("File not found", 0) + 1
            return None

        # === 加载数据 ===
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        if df.empty:
            print(f"⚠️ Empty file: {os.path.basename(file_path)}")
            stats and stats.update({"File is empty": stats.get("File is empty", 0) + 1})
            return None

        required_cols = ["提问时间", "提问内容", "AI回复"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Missing columns {missing_cols} in {os.path.basename(file_path)}")
            stats and stats.update({"missing columns": stats.get("missing columns", 0) + 1})
            return None

        df.fillna("", inplace=True)
        file_name = os.path.basename(file_path)

        # === 校验班级ID ===
        if "教学班ID" not in df.columns:
            print(f"⚠️ Missing class ID column: {file_name}")
            stats and stats.update({"failed": stats.get("failed", 0) + 1})
            return None

        class_id = df["教学班ID"].iloc[0]

        # === 时间转换 ===
        try:
            df["提问时间"] = pd.to_datetime(df["提问时间"], errors="coerce")
        except Exception as e:
            print(f"⏰ Time conversion failed in {file_name}: {e}")
            stats and stats.update({"Time conversion": stats.get("Time conversion", 0) + 1})
            return None

        if df["提问时间"].isna().any():
            print(f"⚠️ Invalid QA times in {file_name}")
            stats and stats.update({"Invalid QA times": stats.get("Invalid QA times", 0) + 1})
            return None

        # === 班级过滤 ===
        if df_school and isinstance(df_school, dict) and "df_class_info" in df_school:
            valid_ids = set(df_school["df_class_info"]["教学班ID"].astype(str))
            if str(class_id) not in valid_ids:
                print(f"⏭️ Skip class {class_id} (not in class_info_file): {file_name}")
                stats and stats.update({"filtered_by_class_info": stats.get("filtered_by_class_info", 0) + 1})
                return None

        # === 课程信息 ===
        class_info = df_class[df_class["教学班ID"] == class_id]
        if class_info.empty:
            print(f"⚠️ No class info for {class_id}: {file_name}")
            stats and stats.update({"No class info": stats.get("No class info", 0) + 1})
            return None

        course_start = pd.to_datetime(class_info["起始时间"].iloc[0], errors="coerce")
        course_end = pd.to_datetime(class_info["结束时间"].iloc[0], errors="coerce")

        if pd.isna(course_start) or pd.isna(course_end):
            print(f"⚠️ Invalid course time for {class_id}: {file_name}")
            stats and stats.update({"Invalid course start/end time": stats.get("Invalid course start/end time", 0) + 1})
            return None

        # === 超出教学时间范围 ===
        qa_min, qa_max = df["提问时间"].min(), df["提问时间"].max()
        if qa_min < course_start or qa_max > course_end:
            print(f"🚫 Out of course window [{course_start}, {course_end}] -> skip: {file_name}")
            stats and stats.update({"out_of_range": stats.get("out_of_range", 0) + 1})
            return None

        # === 检测复制题目 ===
        copy_keywords_count = sum(q.count(kw) for q in df["提问内容"] if isinstance(q, str) for kw in COPY_KEYWORDS)
        has_copy_keywords = int(copy_keywords_count > 0)

        # === 对话统计 ===
        qa_turns = len(df)
        total_time = (
            (df["提问时间"].max() - df["提问时间"].min()).total_seconds() / 60 if qa_turns > 1 else 0
        )
        avg_qa_time = total_time / (qa_turns - 1) if qa_turns > 1 else 0
        question_lengths = df["提问内容"].astype(str).str.len()
        avg_question_length = question_lengths.mean() if not question_lengths.empty else 0.0

        if_non_class = int(df.get("提问入口", pd.Series(["班级"])).ne("班级").any())

        # === 作业与时间关系 ===
        avg_hours_to_assignment = np.mean(
            [get_hours_to_next_assignment(t, class_id, df_homework) for t in df["提问时间"]]
        )
        avg_hours_since_release = np.mean(
            [get_hours_since_last_assignment_release(t, class_id, df_homework) for t in df["提问时间"]]
        )

        total_weeks = max(1, ((course_end - course_start).days // 7) + 1)
        qa_start_time = df["提问时间"].min()
        progress_values = [
            get_teaching_week(t, class_id, df_class) / total_weeks
            for t in df["提问时间"]
            if course_start <= t <= course_end
        ]

        if qa_start_time < course_start:
            course_progress_ratio = before_course_value
        elif qa_start_time > course_end:
            course_progress_ratio = after_course_value
        else:
            course_progress_ratio = float(np.mean(progress_values)) if progress_values else 0.0

        anchor = pd.Timestamp("2025-02-17")
        calendar_week = int(((qa_start_time.normalize() - anchor.normalize()).days // 7) + 1)

        avg_hours_to_next_class = np.mean(
            [get_time_to_next_class(t, class_id, df_schedule) for t in df["提问时间"]]
        )
        avg_hours_from_last_class = np.mean(
            [get_time_from_last_class(t, class_id, df_schedule) for t in df["提问时间"]]
        )

        # === 新增特征 ===
        day_period = qa_start_time.hour + qa_start_time.minute / 60
        is_weekend = int(qa_start_time.weekday() >= 5)
        current_week = get_teaching_week(qa_start_time, class_id, df_class)
        is_exam_week = int(current_week >= total_weeks - 1)

        def in_class_time(t):
            schedule = df_schedule[df_schedule["教学班ID"] == class_id]
            return any(
                pd.to_datetime(row["开课时间"], errors="coerce") <= t <= pd.to_datetime(row["结课时间"], errors="coerce")
                for _, row in schedule.iterrows()
            )

        is_in_class_time = int(any(in_class_time(t) for t in df["提问时间"] if pd.notna(t)))
        question_texts = " ".join(df["提问内容"].astype(str))
        question_type_why_how = int(any(kw in question_texts for kw in ["为什么", "为啥", "怎么"]))

        # === 汇总特征 ===
        features = dict(
            file_name=file_name,
            class_id=class_id,
            qa_turns=int(qa_turns),
            is_multi_turn=qa_turns > 1,
            total_time_minutes=float(total_time),
            avg_qa_time_minutes=float(avg_qa_time),
            total_question_chars=int(question_lengths.sum()),
            avg_question_length=float(avg_question_length),
            if_non_class=if_non_class,
            avg_hours_to_assignment=float(avg_hours_to_assignment),
            avg_hours_since_release=float(avg_hours_since_release),
            course_progress_ratio=float(course_progress_ratio),
            calendar_week_since_2025_0217=calendar_week,
            hours_to_next_class=float(avg_hours_to_next_class),
            hours_from_last_class=float(avg_hours_from_last_class),
            has_copy_keywords=has_copy_keywords,
            copy_keywords_count=int(copy_keywords_count),
            is_exam_week=is_exam_week,
            day_period=day_period,
            is_weekend=is_weekend,
            is_in_class_time=is_in_class_time,
            question_type_why_how=question_type_why_how,
        )

        # === 清理无效值 ===
        for k, v in features.items():
            if k not in ["file_name", "class_id"] and not np.isfinite(v):
                features[k] = 0.0

        stats and stats.update({"processed": stats.get("processed", 0) + 1})
        return features

    except Exception as e:
        print(f"❗ Error in {file_path}: {e}")
        import traceback

        traceback.print_exc()
        return None

def plot_feature_histograms(
    df,
    features=None,
    bins=50,
    save_dir=None,
    figsize=(7, 4),
    stats_file="feature_stats.csv",
):
    """
    绘制特征分布图，并保存每个特征的 min/max/mean/var。
    - 数值连续变量绘制直方图；
    - 二元变量（0/1）绘制柱状图；
    - 自动识别特征列。
    返回：保存的图片路径列表和统计 DataFrame
    """
    save_dir = os.path.abspath(save_dir or "histograms")
    os.makedirs(save_dir, exist_ok=True)

    # === 自动识别特征列 ===
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"🧭 检测到 {len(features)} 个数值型特征：{features}")

    saved_files, stats_list = [], []

    for feature in features:
        if feature not in df.columns:
            print(f"⚠️ 缺失列：{feature} -> 跳过")
            continue

        series = pd.to_numeric(df[feature], errors="coerce").dropna()
        if series.empty:
            print(f"⚠️ {feature} 无有效数据 -> 跳过")
            continue

        # === 统计特征 ===
        stats = {
            "feature": feature,
            "min": series.min(),
            "max": series.max(),
            "mean": series.mean(),
            "median": series.median(),
            "variance": series.var(),
        }
        stats_list.append(stats)

        # === 绘图 ===
        plt.figure(figsize=figsize)
        unique_vals = sorted(series.unique())
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
            # 二元变量
            counts = series.value_counts().sort_index()
            plt.bar(counts.index.astype(str), counts.values, color="skyblue", edgecolor="black")
            plt.title(f"{feature} (binary: {unique_vals})")
            plt.xlabel(feature)
            plt.ylabel("Count")
        else:
            # 连续变量
            plt.hist(series, bins=bins, color="steelblue", edgecolor="black", alpha=0.75)
            plt.axvline(series.mean(), color="red", linestyle="--", label=f"Mean={series.mean():.2f}")
            plt.axvline(series.median(), color="green", linestyle=":", label=f"Median={series.median():.2f}")
            plt.legend()
            plt.title(f"{feature} (n={len(series)})")
            plt.xlabel(feature)
            plt.ylabel("Count")

        out_path = os.path.join(save_dir, f"{feature}_dist.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        saved_files.append(out_path)
        print(f"✅ 已保存直方图: {out_path}")

    # === 保存统计数据 ===
    stats_df = pd.DataFrame(stats_list)
    stats_path = os.path.join(save_dir, stats_file)
    stats_df.to_csv(stats_path, index=False)
    print(f"\n📊 已生成 {len(saved_files)} 张图表，统计信息保存在: {stats_path}")

    return saved_files, stats_df


# ======================================================================

def extract_all_features(
    dialog_folder,
    class_time_file,
    homework_file,
    class_schedule_file,
    school_info_file=None,
    class_info_file=None,
    plot_histograms=True,
):
    """
    批量提取所有对话文件的特征，并可选绘制直方图。
    """
    print("📦 Loading reference data...")
    try:
        df_class, df_homework, df_schedule, df_school = load_reference_data(
            class_time_file,
            homework_file,
            class_schedule_file,
            school_info_file=school_info_file,
            class_info_file=class_info_file,
        )
    except Exception as e:
        print(f"❌ Failed to load reference data: {e}")
        df_class, df_homework, df_schedule, df_school = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    print(f"🔍 Searching CSVs in: {dialog_folder}")

    # === 递归查找所有 CSV 文件 ===
    patterns = [
        os.path.join(dialog_folder, "*.csv"),
        os.path.join(dialog_folder, "*", "*.csv"),
        os.path.join(dialog_folder, "*", "*", "*.csv"),
        os.path.join(dialog_folder, "**", "*.csv"),
    ]
    csv_files = list({fp for p in patterns for fp in glob.glob(p, recursive=True)})

    exclude_keywords = ["feature", "cluster", "result", "statistic", "analysis", "pca"]
    dialog_files = [
        f for f in csv_files if not any(kw in os.path.basename(f).lower() for kw in exclude_keywords)
    ]
    print(f"🗂️ 找到 {len(dialog_files)} 个可能的对话 CSV 文件")

    if not dialog_files:
        print("⚠️ 未找到任何对话文件，文件夹结构如下：")
        for root, _, files in os.walk(dialog_folder):
            level = root.replace(dialog_folder, "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            csv_in_dir = [f for f in files if f.endswith(".csv")]
            for f in csv_in_dir[:5]:
                print(f"{indent}  {f}")
            if len(csv_in_dir) > 5:
                print(f"{indent}  ... 还有 {len(csv_in_dir) - 5} 个 CSV 文件")
        return pd.DataFrame()

    # === 校验前10个文件结构 ===
    print("🧩 Validating file formats...")
    valid_files = []
    required_cols = ["提问时间", "提问内容", "AI回复"]

    for file_path in dialog_files[:10]:
        try:
            df_test = pd.read_csv(file_path, encoding="utf-8-sig", nrows=3)
            if all(col in df_test.columns for col in required_cols):
                print(f"✓ {os.path.basename(file_path)} 格式有效")
                valid_files.append(file_path)
            else:
                print(f"✗ {os.path.basename(file_path)} 缺少必要列 -> {df_test.columns.tolist()}")
        except Exception as e:
            print(f"✗ {os.path.basename(file_path)} 读取失败: {e}")

    if len(valid_files) == min(10, len(dialog_files)):
        valid_files = dialog_files
        print(f"✅ 前10个文件验证通过，假设全部 {len(dialog_files)} 个文件有效")
    else:
        print("🔎 逐个验证全部文件...")
        valid_files = []
        for file_path in dialog_files:
            try:
                df_test = pd.read_csv(file_path, encoding="utf-8-sig", nrows=3)
                if all(col in df_test.columns for col in required_cols):
                    valid_files.append(file_path)
            except Exception:
                continue

    print(f"📁 最终有效对话文件数: {len(valid_files)}")
    if not valid_files:
        print("❌ 没有有效的 CSV 文件！")
        return pd.DataFrame()

    # === 提取特征 ===
    features_list = []
    stats = {"total": 0, "processed": 0, "failed": 0, "out_of_range": 0}
    failed_count = 0

    for i, file_path in enumerate(valid_files, start=1):
        if i % 100 == 0:
            print(f"⏳ Progress: {i}/{len(valid_files)}")

        features = extract_features_from_dialog(
            file_path,
            df_class,
            df_homework,
            df_schedule,
            df_school=df_school,
            stats=stats,
        )
        if features is not None:
            features_list.append(features)
        else:
            failed_count += 1
            if failed_count <= 5:
                print(f"⚠️ 特征提取失败: {os.path.basename(file_path)}")

    print(f"✅ 成功提取 {len(features_list)} 个文件特征，失败 {failed_count} 个")

    # === 保存统计 ===
    stats_to_save = {
        "valid_files_found": len(valid_files),
        "total_dialog_calls": stats.get("total", 0),
        "processed_dialogs": stats.get("processed", 0),
        "out_of_range_dialogs": stats.get("out_of_range", 0),
        "successful_feature_rows": len(features_list),
        "stats": stats,
    }
    total = stats_to_save["total_dialog_calls"]
    if total > 0:
        stats_to_save.update(
            {
                "success_ratio": stats["processed"] / total,
                "out_of_range_ratio": stats["out_of_range"] / total,
                "fail_ratio": (stats["out_of_range"] + stats["failed"]) / total,
            }
        )

    stats_dir = os.path.dirname(os.path.abspath(homework_file))
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, "dialog_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_to_save, f, ensure_ascii=False, indent=2)
    print(f"📊 统计信息保存到: {stats_path}")

    if not features_list:
        print("⚠️ 没有成功的特征结果！")
        return pd.DataFrame()

    # === 构建 DataFrame ===
    features_df = pd.DataFrame(features_list)

    # === 替换无穷大值 ===
    print("♻️ 处理 class 时间特征中的无穷大值...")
    for col in ["hours_to_next_class", "hours_from_last_class"]:
        if col in features_df.columns and np.isinf(features_df[col]).any():
            features_df[col] = replace_inf_with_reasonable_value(features_df[col], multiplier=1.5)

    # === 绘制直方图 ===
    if plot_histograms:
        hist_dir = os.path.join(stats_dir, "histograms_before_log")
        print(f"🖼️ 绘制原始特征分布图到: {hist_dir}")
        plot_feature_histograms(features_df, save_dir=hist_dir)

    # === log(1+x) 变换 ===
    log_features = [
        "avg_hours_since_release",
        "avg_hours_to_assignment",
        "avg_qa_time_minutes",
        "avg_question_length",
        "copy_keywords_count",
        "course_progress_ratio",
        "qa_turns",
        "total_question_chars",
        "total_time_minutes",
        "hours_from_last_class",
        "hours_to_next_class",
    ]
    for col in log_features:
        if col in features_df.columns:
            features_df[col] = np.log1p(features_df[col])

    if plot_histograms:
        hist_dir = os.path.join(stats_dir, "histograms_after_log")
        print(f"🖼️ 绘制 log 特征分布图到: {hist_dir}")
        plot_feature_histograms(features_df, save_dir=hist_dir)

    return features_df
