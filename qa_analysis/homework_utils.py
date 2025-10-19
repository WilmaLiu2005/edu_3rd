import pandas as pd


def get_hours_to_next_assignment(
    qa_time: str | pd.Timestamp,
    class_id: str | int,
    df_homework: pd.DataFrame,
    max_hours: float = 720,
) -> float:
    """
    计算距离下一次作业截止的小时数（最大默认720小时 = 30天）

    参数:
        qa_time: 当前问答时间（字符串或 pandas.Timestamp）
        class_id: 教学班 ID
        df_homework: 作业信息表，包含字段 ['教学班ID', '提交截止时间']
        max_hours: 限制返回的最大小时数，默认720

    返回:
        float: 距离最近未来作业截止的小时数，若无则返回 max_hours
    """
    # 筛选该教学班的作业记录
    class_hw = df_homework[df_homework["教学班ID"] == class_id]
    if class_hw.empty:
        return max_hours

    # 解析截止时间
    deadline_times = pd.to_datetime(class_hw["提交截止时间"], errors="coerce").dropna()
    if deadline_times.empty:
        return max_hours

    # 解析问答时间
    qa_ts = pd.to_datetime(qa_time, errors="coerce")
    if pd.isna(qa_ts):
        return max_hours

    # 找到所有未来的截止时间
    future_deadlines = deadline_times[deadline_times > qa_ts]
    if future_deadlines.empty:
        return max_hours

    next_deadline = future_deadlines.min()
    hours_diff = (next_deadline - qa_ts).total_seconds() / 3600
    return min(hours_diff, max_hours)


def get_hours_since_last_assignment_release(
    qa_time: str | pd.Timestamp,
    class_id: str | int,
    df_homework: pd.DataFrame,
    max_hours: float = 720,
) -> float:
    """
    计算距离最近一次作业发布的小时数（最大默认720小时 = 30天）

    参数:
        qa_time: 当前问答时间（字符串或 pandas.Timestamp）
        class_id: 教学班 ID
        df_homework: 作业信息表，包含字段 ['教学班ID', '发布时间']
        max_hours: 限制返回的最大小时数，默认720

    返回:
        float: 从最近一次作业发布时间到 qa_time 的小时差，若无则返回 max_hours
    """
    # 筛选该教学班的作业记录
    class_hw = df_homework[df_homework["教学班ID"] == class_id]
    if class_hw.empty:
        return max_hours

    # 解析发布时间
    release_times = pd.to_datetime(class_hw["发布时间"], errors="coerce").dropna()
    if release_times.empty:
        return max_hours

    # 解析问答时间
    qa_ts = pd.to_datetime(qa_time, errors="coerce")
    if pd.isna(qa_ts):
        return max_hours

    # 找到所有过去的发布时间
    past_releases = release_times[release_times <= qa_ts]
    if past_releases.empty:
        return max_hours

    last_release = past_releases.max()
    hours_diff = (qa_ts - last_release).total_seconds() / 3600
    return min(hours_diff, max_hours)
