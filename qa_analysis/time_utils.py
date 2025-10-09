import pandas as pd
import numpy as np

def get_time_to_next_class(qa_time, class_id, df_schedule):
    """计算对话开始时间与接下来最近一次上课开始时间的间隔（小时）"""
    # 根据教学班ID筛选课程
    class_schedule = df_schedule[df_schedule['教学班ID'] == class_id]
    
    if class_schedule.empty:
        return np.inf  # 找不到对应课程，返回无穷大
    
    # 获取有效的开课时间
    start_times = pd.to_datetime(class_schedule['开课时间'], errors='coerce').dropna()
    
    if start_times.empty:
        return np.inf
    
    # 找到在QA时间之后的所有开课时间
    future_classes = start_times[start_times > qa_time]
    
    if not future_classes.empty:
        # 找到最近的一次开课时间
        next_class = future_classes.min()
        hours_diff = (next_class - qa_time).total_seconds() / 3600  # 转换为小时
        return max(hours_diff, 0)  # 确保非负
    else:
        return np.inf  # 没有未来的课程

def get_time_from_last_class(qa_time, class_id, df_schedule):
    """计算对话开始时间与之前最近一次上课结束时间的间隔（小时）"""
    # 根据教学班ID筛选课程
    class_schedule = df_schedule[df_schedule['教学班ID'] == class_id]
    
    if class_schedule.empty:
        return np.inf  # 找不到对应课程，返回无穷大
    
    # 获取有效的结课时间
    end_times = pd.to_datetime(class_schedule['结课时间'], errors='coerce').dropna()
    
    if end_times.empty:
        return np.inf
    
    # 找到在QA时间之前的所有结课时间
    past_classes = end_times[end_times < qa_time]
    
    if not past_classes.empty:
        # 找到最近的一次结课时间
        last_class = past_classes.max()
        hours_diff = (qa_time - last_class).total_seconds() / 3600  # 转换为小时
        return max(hours_diff, 0)  # 确保非负
    else:
        return np.inf  # 没有之前的课程
    
def get_teaching_week(qa_time, class_id, df_class):
    """计算QA发生在第几个教学周；仅当 qa_time 落在[开始时间, 结束时间]内才返回有效周数，否则返回 -1"""
    class_info = df_class[df_class['教学班ID'] == class_id]
    if class_info.empty:
        return -1

    start_time = pd.to_datetime(class_info['开始时间'].iloc[0], errors='coerce')
    end_time = pd.to_datetime(class_info['结束时间'].iloc[0], errors='coerce')

    qa_time = pd.to_datetime(qa_time, errors='coerce')

    if pd.isna(start_time) or pd.isna(end_time) or pd.isna(qa_time):
        return -1

    if not (start_time <= qa_time <= end_time):
        return -1

    days_diff = (qa_time - start_time).days
    week_num = max(1, (days_diff // 7) + 1)
    return week_num
