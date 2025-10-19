import pandas as pd
import os

def load_reference_data(class_time_file, homework_file, class_schedule_file,
                        school_info_file=None, class_info_file=None):
    """
    加载参考数据（包含课堂时间、作业、课程安排、学校信息、班级信息）
    """
    print("📂 Loading reference CSV files...")

    # === 基础三类文件 ===
    df_class = pd.read_csv(class_time_file, encoding='utf-8-sig')
    df_homework = pd.read_csv(homework_file, encoding='utf-8-sig')
    df_schedule = pd.read_csv(class_schedule_file, encoding='utf-8-sig')
    
    # === 时间列转换 ===
    for col in ['开始时间', '结束时间']:
        if col in df_class.columns:
            df_class[col] = pd.to_datetime(df_class[col], errors='coerce')
    for col in ['发布时间', '提交截止时间']:
        if col in df_homework.columns:
            df_homework[col] = pd.to_datetime(df_homework[col], errors='coerce')
    for col in ['开课时间', '结课时间']:
        if col in df_schedule.columns:
            df_schedule[col] = pd.to_datetime(df_schedule[col], errors='coerce')

    # === 加载学校基础信息（可选）===
    df_school = pd.DataFrame()
    if school_info_file and pd.io.common.file_exists(school_info_file):
        try:
            df_school = pd.read_csv(school_info_file, encoding='utf-8-sig')
            for col in ['起始时间', '结束时间']:
                if col in df_school.columns:
                    df_school[col] = pd.to_datetime(df_school[col], errors='coerce')
            print(f"✅ Loaded school info: {len(df_school)} rows from {school_info_file}")
        except Exception as e:
            print(f"⚠️ Failed to load school info file: {e}")

    # 加载班级信息文件（如果提供）
    df_class_info = pd.DataFrame()
    if class_info_file and os.path.exists(class_info_file):
        df_class_info = pd.read_csv(class_info_file, encoding='utf-8-sig')
        print(f"Loaded class info: {len(df_class_info)} rows")

    # ✅ 将所有参考数据打包成 dict，方便在 extract_features_from_dialog 使用
    df_school_bundle = {
        'df_school': df_school,
        'df_class_info': df_class_info
    }

    return df_class, df_homework, df_schedule, df_school_bundle
