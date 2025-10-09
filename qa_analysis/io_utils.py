import pandas as pd

def load_reference_data(class_time_file, homework_file, class_schedule_file):
    """加载参考数据（新增课堂时间数据）"""
    df_class = pd.read_csv(class_time_file, encoding='utf-8-sig')
    df_homework = pd.read_csv(homework_file, encoding='utf-8-sig')
    df_schedule = pd.read_csv(class_schedule_file, encoding='utf-8-sig')  # 新增
    
    # 转换时间格式
    df_class['开始时间'] = pd.to_datetime(df_class['开始时间'])
    df_class['结束时间'] = pd.to_datetime(df_class['结束时间'])
    df_homework['发布时间'] = pd.to_datetime(df_homework['发布时间'], errors='coerce')
    df_homework['提交截止时间'] = pd.to_datetime(df_homework['提交截止时间'], errors='coerce')
    
    # 转换课堂时间格式
    df_schedule['开课时间'] = pd.to_datetime(df_schedule['开课时间'], errors='coerce')
    df_schedule['结课时间'] = pd.to_datetime(df_schedule['结课时间'], errors='coerce')
    
    return df_class, df_homework, df_schedule