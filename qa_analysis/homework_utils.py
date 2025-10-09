import pandas as pd

def get_hours_to_next_assignment(qa_time, class_id, df_homework):
    """计算距离下一次作业截止的小时数（最大720小时）"""
    class_homework = df_homework[df_homework['教学班ID'] == class_id]
    if class_homework.empty:
        return 720  # 默认值（30天）

    deadline_times = pd.to_datetime(class_homework['提交截止时间'], errors='coerce').dropna()
    if deadline_times.empty:
        return 720

    qa_ts = pd.to_datetime(qa_time, errors='coerce')
    if pd.isna(qa_ts):
        return 720

    future_deadlines = deadline_times[deadline_times > qa_ts]
    if not future_deadlines.empty:
        next_deadline = future_deadlines.min()
        hours_diff = (next_deadline - qa_ts).total_seconds() / 3600
        return min(hours_diff, 720)  # 最大30天=720小时
    else:
        return 720  # 后面没有作业就返回720小时


def get_hours_since_last_assignment_release(qa_time, class_id, df_homework):
    """计算距离最近一次作业发布的小时数（从最近一次发布到 qa_time 的间隔，最大720小时）"""
    class_homework = df_homework[df_homework['教学班ID'] == class_id]
    if class_homework.empty:
        return 720  # 默认值（30天）

    release_times = pd.to_datetime(class_homework['发布时间'], errors='coerce').dropna()
    if release_times.empty:
        return 720

    qa_ts = pd.to_datetime(qa_time, errors='coerce')
    if pd.isna(qa_ts):
        return 720

    past_releases = release_times[release_times <= qa_ts]
    if not past_releases.empty:
        last_release = past_releases.max()
        hours_diff = (qa_ts - last_release).total_seconds() / 3600
        return min(hours_diff, 720)  # 最大30天=720小时
    else:
        return 720  # 如果 qa_time 之前没有发布记录，返回720小时