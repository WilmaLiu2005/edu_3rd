import pandas as pd
from openpyxl import load_workbook
# 特征信息列表
features_info = [
    {
        "Variable name": "qa_turns",
        "Description": "对话轮数",
        "How it is computed": "对话文件中行数",
        "Data type": "continuous",
        "Range": "1..n",
        "Transformation done": "ln(1+x)",
        "Summary after transformation": "Skew reduced, easier comparison"
    },
    {
        "Variable name": "is_multi_turn",
        "Description": "是否多轮对话",
        "How it is computed": "qa_turns > 1",
        "Data type": "binary",
        "Range": "0/1",
        "Transformation done": "None",
        "Summary after transformation": "unchanged"
    },
    {
        "Variable name": "total_time_minutes",
        "Description": "对话总时长（分钟）",
        "How it is computed": "(max(提问时间)-min(提问时间))",
        "Data type": "continuous",
        "Range": "0..∞",
        "Transformation done": "ln(1+x)",
        "Summary after transformation": "Right-skew reduced"
    },
    {
        "Variable name": "avg_qa_time_minutes",
        "Description": "平均轮间时间（分钟）",
        "How it is computed": "total_time/(qa_turns-1)",
        "Data type": "continuous",
        "Range": "0..∞",
        "Transformation done": "ln(1+x)",
        "Summary after transformation": "Normalized, skew reduced"
    },
    {
        "Variable name": "total_question_chars",
        "Description": "提问总字符数",
        "How it is computed": "所有提问内容字符数之和",
        "Data type": "continuous",
        "Range": "0..∞",
        "Transformation done": "ln(1+x)",
        "Summary after transformation": "Right-skew reduced"
    },
    {
        "Variable name": "avg_question_length",
        "Description": "平均提问长度",
        "How it is computed": "total_question_chars / qa_turns",
        "Data type": "continuous",
        "Range": "0..∞",
        "Transformation done": "ln(1+x)",
        "Summary after transformation": "Right-skew reduced"
    },
    {
        "Variable name": "if_non_class",
        "Description": "是否来自非班级入口",
        "How it is computed": '提问入口 != "班级"',
        "Data type": "binary",
        "Range": "0/1",
        "Transformation done": "None",
        "Summary after transformation": "unchanged"
    },
    {
        "Variable name": "avg_hours_to_assignment",
        "Description": "到下一作业发布时间的平均小时数",
        "How it is computed": "根据作业发布时间与提问时间计算",
        "Data type": "continuous",
        "Range": "0..∞",
        "Transformation done": "ln(1+x)",
        "Summary after transformation": "Skew reduced"
    },
    {
        "Variable name": "avg_hours_since_release",
        "Description": "自上次作业发布以来的平均小时数",
        "How it is computed": "提问时间 - 最近作业发布时间",
        "Data type": "continuous",
        "Range": "0..∞",
        "Transformation done": "ln(1+x)",
        "Summary after transformation": "Skew reduced"
    },
    {
        "Variable name": "course_progress_ratio",
        "Description": "对话发生在课程进度中的比例",
        "How it is computed": "(当前教学周 / 总教学周); 课程前=-0.1, 课程后=1.1",
        "Data type": "continuous",
        "Range": "-0.1..1.1",
        "Transformation done": "ln(1+x)",
        "Summary after transformation": "Scaled, interpretable"
    },
    {
        "Variable name": "calendar_week_since_2025_0217",
        "Description": "自 2025-02-17 的日历周",
        "How it is computed": "floor((qa_start_time-anchor)/7 days)+1",
        "Data type": "continuous",
        "Range": "1..n",
        "Transformation done": "None",
        "Summary after transformation": "unchanged"
    },
    {
        "Variable name": "hours_to_next_class",
        "Description": "到下一节课的平均小时数",
        "How it is computed": "根据课程表计算",
        "Data type": "continuous",
        "Range": "0..∞",
        "Transformation done": "ln(1+x)",
        "Summary after transformation": "Skew reduced"
    },
    {
        "Variable name": "hours_from_last_class",
        "Description": "自上一节课以来的平均小时数",
        "How it is computed": "根据课程表计算",
        "Data type": "continuous",
        "Range": "0..∞",
        "Transformation done": "ln(1+x)",
        "Summary after transformation": "Skew reduced"
    },
    {
        "Variable name": "has_copy_keywords",
        "Description": "是否包含复制题关键词",
        "How it is computed": "copy_keywords_count > 0",
        "Data type": "binary",
        "Range": "0/1",
        "Transformation done": "None",
        "Summary after transformation": "unchanged"
    },
    {
        "Variable name": "copy_keywords_count",
        "Description": "复制题关键词出现次数",
        "How it is computed": "遍历提问内容计数",
        "Data type": "continuous",
        "Range": "0..n",
        "Transformation done": "ln(1+x)",
        "Summary after transformation": "Skew reduced"
    },
    {
        "Variable name": "day_period",
        "Description": "对话发生时间（小时，精确到分钟）",
        "How it is computed": "qa_start_time.hour + qa_start_time.minute/60",
        "Data type": "continuous",
        "Range": "0..24",
        "Transformation done": "None",
        "Summary after transformation": "unchanged"
    },
    {
        "Variable name": "is_weekend",
        "Description": "是否周末",
        "How it is computed": "qa_start_time.weekday()>=5",
        "Data type": "binary",
        "Range": "0/1",
        "Transformation done": "None",
        "Summary after transformation": "unchanged"
    },
    {
        "Variable name": "is_exam_week",
        "Description": "是否考试周（最后两周）",
        "How it is computed": "current_week >= total_weeks-1",
        "Data type": "binary",
        "Range": "0/1",
        "Transformation done": "None",
        "Summary after transformation": "unchanged"
    },
    {
        "Variable name": "is_first_tier",
        "Description": "是否一本",
        "How it is computed": "根据 df_school 平台ID匹配",
        "Data type": "binary",
        "Range": "0/1",
        "Transformation done": "None",
        "Summary after transformation": "unchanged"
    },
    {
        "Variable name": "is_in_class_time",
        "Description": "对话是否在上课时间内",
        "How it is computed": "检查提问时间是否在课程表中",
        "Data type": "binary",
        "Range": "0/1",
        "Transformation done": "None",
        "Summary after transformation": "unchanged"
    },
    {
        "Variable name": "question_type_why_how",
        "Description": "是否包含“为什么/为啥/怎么”问题",
        "How it is computed": "遍历提问内容匹配关键词",
        "Data type": "binary",
        "Range": "0/1",
        "Transformation done": "None",
        "Summary after transformation": "unchanged"
    }
]

df = pd.DataFrame(features_info)

# 保存初步 Excel
excel_path = "features_document.xlsx"
df.to_excel(excel_path, index=False)

# 使用 openpyxl 自适应列宽
wb = load_workbook(excel_path)
ws = wb.active

for column_cells in ws.columns:
    max_length = 0
    column_letter = column_cells[0].column_letter
    for cell in column_cells:
        try:
            cell_length = len(str(cell.value))
            if cell_length > max_length:
                max_length = cell_length
        except:
            pass
    adjusted_width = max_length + 2  # 增加一点空余
    ws.column_dimensions[column_letter].width = adjusted_width

wb.save(excel_path)
print(f"✅ Excel saved with auto-adjusted column width: {excel_path}")