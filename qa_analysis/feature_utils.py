import numpy as np
import pandas as pd
import re

COPY_KEYWORDS = ["如下", "如上", "这道题", "怎么做", "做法", "A.", "B.", "C.", "D.", "我对这一页不懂"]

def replace_inf_with_reasonable_value(series, multiplier=1.5):
    """将无穷大值替换为合理的有限值"""
    if series.empty:
        return series
    
    # 获取非无穷大的值
    finite_values = series[np.isfinite(series)]
    
    replacement_value = 168.0  # 一周的小时数作为默认值
    
    # 替换无穷大值
    series_cleaned = series.replace([np.inf, -np.inf], replacement_value)
    
    return series_cleaned

def debug_infinite_values(features_df):
    """调试无穷大值（更新版）"""
    print("=== Checking Infinite Values ===")
    
    # 更新特征列表
    feature_columns = [
        'qa_turns', 'is_multi_turn', 'total_time_minutes', 'avg_qa_time_minutes',
        'total_question_chars', 'avg_question_length',
        'if_non_class', 'avg_hours_to_assignment', 'avg_hours_since_release',
        'course_progress_ratio', 'calendar_week_since_2025_0217',
        'hours_to_next_class', 'hours_from_last_class', 'has_copy_keywords', 'copy_keywords_count',
        'is_exam_week','day_period','is_weekend',
        'is_in_class_time','question_type_why_how'
    ]
    
    print("Available columns in features_df:")
    print(features_df.columns.tolist())
    
    for col in feature_columns:
        if col in features_df.columns:
            inf_count = np.isinf(features_df[col]).sum()
            nan_count = np.isnan(features_df[col]).sum()
            
            if inf_count > 0 or nan_count > 0:
                print(f"Column '{col}': {inf_count} infinite values, {nan_count} NaN values")
                
                if inf_count > 0:
                    inf_indices = features_df[np.isinf(features_df[col])].index
                    print(f"  Infinite values in rows: {inf_indices.tolist()[:5]}...")
                    if 'file_name' in features_df.columns:
                        print(f"  Corresponding files: {features_df.loc[inf_indices[:3], 'file_name'].tolist()}")
            
            # 显示基本统计（排除无穷大值）
            finite_data = features_df[col][np.isfinite(features_df[col])]
            if not finite_data.empty:
                print(f"  {col} (finite only): min={finite_data.min():.2f}, max={finite_data.max():.2f}, mean={finite_data.mean():.2f}")
            else:
                print(f"  {col}: All values are infinite or NaN")
    
    return

def normalize_for_keyword(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    # 去掉 markdown 链接显示的 url，只保留可见文本（可选）
    s = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s)
    # 全角转半角
    res = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000:
            code = 32
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        res.append(chr(code))
    s = "".join(res)
    # 去多余空白
    s = re.sub(r'\s+', '', s)  # 去掉所有空白，避免“这 道 题”之类被漏检
    return s

def contains_copy_keywords(text: str, keywords=COPY_KEYWORDS) -> bool:
    s = normalize_for_keyword(text)
    if not s:
        return False
    return any(k in s for k in keywords)