import os
import pandas as pd
def check_dialog_time(folder_path, class_info_file):
    # 读取课程时间信息
    df_class = pd.read_csv(class_info_file)
    df_class['开始时间'] = pd.to_datetime(df_class['开始时间'], errors='coerce')
    df_class['结束时间'] = pd.to_datetime(df_class['结束时间'], errors='coerce')

    # 必须列
    required_columns = [
        "平台ID", "课程名称", "教学班ID", "学生ID",
        "提问入口", "提问内容", "AI回复", "提问轮次(一个会话一天内)", "提问时间"
    ]

    # 统计结果
    stats = {
        'in_range': 0,
        'before_course': 0,
        'after_course': 0,
        'no_class_info': 0,
        'failed_read': 0,
        'invalid_columns': 0,
        'total_files': 0
    }

    # 遍历文件夹（递归）
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if not file_name.lower().endswith('.csv'):
                continue

            stats['total_files'] += 1
            file_path = os.path.join(root, file_name)

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Failed to read {file_name}: {e}")
                stats['failed_read'] += 1
                continue

            # 检查列
            if not all(col in df.columns for col in required_columns):
                print(f"File {file_name} skipped: missing required columns")
                stats['invalid_columns'] += 1
                continue

            # 转换提问时间
            df['提问时间'] = pd.to_datetime(df['提问时间'], errors='coerce')
            if df['提问时间'].isna().all():
                print(f"All question times invalid in {file_name}")
                stats['failed_read'] += 1
                continue

            # 以教学班ID为单位判断时间区间
            file_status = None
            for class_id in df['教学班ID'].unique():
                class_rows = df[df['教学班ID'] == class_id]
                class_info = df_class[df_class['教学班ID'] == class_id]

                if class_info.empty:
                    print(f"No class info for class_id={class_id}: {file_name}")
                    stats['no_class_info'] += 1
                    continue

                course_start = class_info['开始时间'].iloc[0]
                course_end = class_info['结束时间'].iloc[0]

                qa_min_time = class_rows['提问时间'].min()
                qa_max_time = class_rows['提问时间'].max()

                if qa_max_time < course_start:
                    file_status = 'before_course'
                elif qa_min_time > course_end:
                    file_status = 'after_course'
                else:
                    file_status = 'in_range'

            if file_status:
                stats[file_status] += 1

    print("统计结果:")
    for k, v in stats.items():
        print(f"{k}: {v}")

# 使用示例
folder_path = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/split"
class_info_file = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/code/class_time_range_merged.csv"

check_dialog_time(folder_path, class_info_file)
