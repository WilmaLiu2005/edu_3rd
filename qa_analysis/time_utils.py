"""
时间相关工具函数模块

提供QA对话时间与课程时间的计算功能，包括：
- 与下次上课时间的间隔
- 与上次下课时间的间隔
- 所在教学周的计算
"""

from typing import Union
import pandas as pd
import numpy as np

# 常量定义
DEFAULT_TIME_INTERVAL = 168.0  # 默认时间间隔（小时），相当于一周
HOURS_PER_SECOND = 1.0 / 3600.0  # 秒转小时的系数
DAYS_PER_WEEK = 7  # 每周天数


def get_time_to_next_class(
    qa_time: pd.Timestamp,
    class_id: Union[str, int],
    df_schedule: pd.DataFrame
) -> float:
    """
    计算对话开始时间与接下来最近一次上课开始时间的间隔
    
    Args:
        qa_time: QA对话发生的时间
        class_id: 教学班ID
        df_schedule: 课程表DataFrame，需包含'教学班ID'和'开课时间'列
    
    Returns:
        float: 时间间隔（小时）。如果找不到未来课程，返回168.0（一周）
    """
    class_schedule = df_schedule[df_schedule['教学班ID'] == class_id]
    
    if class_schedule.empty:
        return DEFAULT_TIME_INTERVAL
    
    start_times = pd.to_datetime(class_schedule['开课时间'], errors='coerce').dropna()
    
    if start_times.empty:
        return DEFAULT_TIME_INTERVAL
    
    future_classes = start_times[start_times > qa_time]
    
    if not future_classes.empty:
        next_class = future_classes.min()
        hours_diff = (next_class - qa_time).total_seconds() * HOURS_PER_SECOND
        return max(hours_diff, 0.0)
    else:
        return DEFAULT_TIME_INTERVAL


def get_time_from_last_class(
    qa_time: pd.Timestamp,
    class_id: Union[str, int],
    df_schedule: pd.DataFrame
) -> float:
    """
    计算对话开始时间与之前最近一次上课结束时间的间隔
    
    Args:
        qa_time: QA对话发生的时间
        class_id: 教学班ID
        df_schedule: 课程表DataFrame，需包含'教学班ID'和'结课时间'列
    
    Returns:
        float: 时间间隔（小时）。如果找不到过去课程，返回168.0（一周）
    """
    class_schedule = df_schedule[df_schedule['教学班ID'] == class_id]
    
    if class_schedule.empty:
        return DEFAULT_TIME_INTERVAL
    
    end_times = pd.to_datetime(class_schedule['结课时间'], errors='coerce').dropna()
    
    if end_times.empty:
        return DEFAULT_TIME_INTERVAL
    
    past_classes = end_times[end_times < qa_time]
    
    if not past_classes.empty:
        last_class = past_classes.max()
        hours_diff = (qa_time - last_class).total_seconds() * HOURS_PER_SECOND
        return max(hours_diff, 0.0)
    else:
        return DEFAULT_TIME_INTERVAL


def get_teaching_week(
    qa_time: pd.Timestamp,
    class_id: Union[str, int],
    df_class: pd.DataFrame
) -> int:
    """
    计算QA发生在第几个教学周
    
    仅当qa_time落在课程的[起始时间, 结束时间]区间内时，才返回有效的周数；
    否则返回-1表示无效。
    
    Args:
        qa_time: QA对话发生的时间
        class_id: 教学班ID
        df_class: 教学班信息DataFrame，需包含'教学班ID'、'起始时间'、'结束时间'列
    
    Returns:
        int: 教学周数（从1开始），如果时间不在课程期间或数据无效则返回-1
    """
    class_info = df_class[df_class['教学班ID'] == class_id]
    
    if class_info.empty:
        return -1

    start_time = pd.to_datetime(class_info['起始时间'].iloc[0], errors='coerce')
    end_time = pd.to_datetime(class_info['结束时间'].iloc[0], errors='coerce')
    qa_time = pd.to_datetime(qa_time, errors='coerce')

    if pd.isna(start_time) or pd.isna(end_time) or pd.isna(qa_time):
        return -1

    if not (start_time <= qa_time <= end_time):
        return -1

    days_diff = (qa_time - start_time).days
    week_num = (days_diff // DAYS_PER_WEEK) + 1
    
    return max(1, week_num)
