"""
特征工具模块

提供特征处理相关的工具函数，包括：
- 无穷大值处理
- 关键词检测
- 文本规范化
"""

from typing import List, Optional
import re
import numpy as np
import pandas as pd

# 常量定义
COPY_KEYWORDS = [
    "如下", "如上", "这道题", "怎么做", "做法",
    "A.", "B.", "C.", "D.", "我对这一页不懂"
]

DEFAULT_REPLACEMENT_VALUE = 168.0  # 默认替换值（一周的小时数）

# 特征列列表
FEATURE_COLUMNS = [
    'qa_turns', 'is_multi_turn', 'total_time_minutes', 'avg_qa_time_minutes',
    'total_question_chars', 'avg_question_length',
    'if_non_class', 'avg_hours_to_assignment', 'avg_hours_since_release',
    'course_progress_ratio', 'calendar_week_since_2025_0217',
    'hours_to_next_class', 'hours_from_last_class', 'has_copy_keywords', 'copy_keywords_count',
    'is_exam_week', 'day_period', 'is_weekend',
    'is_in_class_time', 'question_type_why_how'
]


def replace_inf_with_reasonable_value(
    series: pd.Series,
    multiplier: float = 1.5
) -> pd.Series:
    """
    将无穷大值替换为合理的有限值
    
    Args:
        series: 待处理的数据序列
        multiplier: 乘数因子（当前未使用，保留用于未来扩展）
    
    Returns:
        pd.Series: 替换后的数据序列
    """
    if series.empty:
        return series
    
    # 使用默认值替换无穷大
    series_cleaned = series.replace([np.inf, -np.inf], DEFAULT_REPLACEMENT_VALUE)
    
    return series_cleaned


def debug_infinite_values(features_df: pd.DataFrame) -> None:
    """
    调试并打印数据中的无穷大值和NaN值信息
    
    Args:
        features_df: 特征DataFrame
    """
    print("\n" + "=" * 50)
    print("Checking Infinite Values")
    print("=" * 50)
    
    print("\nAvailable columns in features_df:")
    print(features_df.columns.tolist())
    
    has_issues = False
    
    for col in FEATURE_COLUMNS:
        if col not in features_df.columns:
            continue
        
        inf_count = np.isinf(features_df[col]).sum()
        nan_count = np.isnan(features_df[col]).sum()
        
        if inf_count > 0 or nan_count > 0:
            has_issues = True
            print(f"\n⚠️ Column '{col}':")
            print(f"   - Infinite values: {inf_count}")
            print(f"   - NaN values: {nan_count}")
            
            if inf_count > 0:
                inf_indices = features_df[np.isinf(features_df[col])].index
                print(f"   - Infinite values in rows: {inf_indices.tolist()[:5]}...")
                
                if 'file_name' in features_df.columns:
                    sample_files = features_df.loc[inf_indices[:3], 'file_name'].tolist()
                    print(f"   - Sample files: {sample_files}")
        
        # 显示基本统计（排除无穷大值）
        finite_data = features_df[col][np.isfinite(features_df[col])]
        if not finite_data.empty:
            print(f"\n📊 {col} (finite values only):")
            print(f"   - Min:  {finite_data.min():.2f}")
            print(f"   - Max:  {finite_data.max():.2f}")
            print(f"   - Mean: {finite_data.mean():.2f}")
            print(f"   - Std:  {finite_data.std():.2f}")
        else:
            print(f"\n❌ {col}: All values are infinite or NaN")
    
    if not has_issues:
        print("\n✅ No infinite or NaN values found in feature columns")
    
    print("=" * 50)


def normalize_for_keyword(text: str) -> str:
    """
    规范化文本用于关键词检测
    
    处理步骤：
    1. 去除Markdown链接，保留可见文本
    2. 全角字符转半角
    3. 去除所有空白字符
    
    Args:
        text: 待规范化的文本
    
    Returns:
        str: 规范化后的文本
    """
    if text is None:
        return ""
    
    s = str(text)
    
    # 去除Markdown链接，只保留可见文本
    s = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s)
    
    # 全角转半角
    result = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000:  # 全角空格
            code = 32
        elif 0xFF01 <= code <= 0xFF5E:  # 其他全角字符
            code -= 0xFEE0
        result.append(chr(code))
    s = "".join(result)
    
    # 去除所有空白字符（避免"这 道 题"被漏检）
    s = re.sub(r'\s+', '', s)
    
    return s


def contains_copy_keywords(
    text: str,
    keywords: List[str] = COPY_KEYWORDS
) -> bool:
    """
    检测文本中是否包含复制粘贴相关的关键词
    
    Args:
        text: 待检测的文本
        keywords: 关键词列表
    
    Returns:
        bool: 是否包含关键词
    """
    normalized_text = normalize_for_keyword(text)
    
    if not normalized_text:
        return False
    
    return any(keyword in normalized_text for keyword in keywords)


def count_copy_keywords(
    text: str,
    keywords: List[str] = COPY_KEYWORDS
) -> int:
    """
    统计文本中包含的关键词数量
    
    Args:
        text: 待检测的文本
        keywords: 关键词列表
    
    Returns:
        int: 关键词出现次数
    """
    normalized_text = normalize_for_keyword(text)
    
    if not normalized_text:
        return 0
    
    count = sum(1 for keyword in keywords if keyword in normalized_text)
    return count