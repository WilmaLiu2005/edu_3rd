import pandas as pd
from datetime import datetime, timedelta

def analyze_class_time_range_with_course(interaction_file, course_file, output_file):
    """
    分析每个教学班学生对话涉及到的教学班，直接取该教学班的课堂开课/结课时间
    不再使用“持续时间少于一周则改为默认区间”的规则
    仅当课堂开/结课时间缺失时使用默认区间（2025-03-01 ~ 2025-07-07）
    """
    # 读取数据
    interaction_df = pd.read_csv(interaction_file, encoding='utf-8')
    course_df = pd.read_csv(course_file, encoding='utf-8')

    # 转时间类型
    interaction_df['提问时间'] = pd.to_datetime(interaction_df['提问时间'], errors='coerce')
    course_df['开课时间'] = pd.to_datetime(course_df['开课时间'], errors='coerce')
    course_df['结课时间'] = pd.to_datetime(course_df['结课时间'], errors='coerce')

    print(f"学生互动数据中的教学班数量: {interaction_df['教学班ID'].nunique()}")
    print(f"课堂信息数据中的教学班数量: {course_df['教学班ID'].nunique()}")

    # 课堂时间范围（按教学班聚合）
    course_result = course_df.groupby('教学班ID').agg({
        '平台ID': 'first',
        '课程名称': 'first',
        '开课时间': 'min',
        '结课时间': 'max'
    }).reset_index()

    # 只保留在互动中出现过的教学班
    interaction_classes = interaction_df[['教学班ID']].drop_duplicates()
    merged_result = pd.merge(course_result, interaction_classes, on='教学班ID', how='inner')

    print(f"处理后 - 有互动且有课堂信息的教学班: {len(merged_result)}")

    # 直接取课堂开/结课时间
    merged_result['开始时间'] = merged_result['开课时间']
    merged_result['结束时间'] = merged_result['结课时间']

    # 默认时间范围（仅在缺失时启用）
    default_start = pd.to_datetime('2025-03-01 00:00:00')
    default_end = pd.to_datetime('2025-07-07 23:59:59')

    # 仅处理缺失情况，不做“少于一周”调整
    def adjust_missing(row):
        if pd.isna(row['开始时间']) or pd.isna(row['结束时间']):
            row['开始时间'] = default_start
            row['结束时间'] = default_end
            row['调整原因'] = '缺失时间数据'
        else:
            row['调整原因'] = '无需调整'
        return row

    merged_result = merged_result.apply(adjust_missing, axis=1)

    # 统计
    adjusted_count = (merged_result['调整原因'] != '无需调整').sum()
    print(f"\n调整统计:")
    print(f"需要使用默认区间的教学班数量: {adjusted_count}")
    print("调整原因统计:")
    for reason, count in merged_result['调整原因'].value_counts().items():
        print(f"  {reason}: {count}个教学班")

    # 输出主结果
    final_result = merged_result[['平台ID', '课程名称', '教学班ID', '开始时间', '结束时间']].copy()
    final_result['平台ID'] = final_result['平台ID'].fillna('未知')
    final_result['课程名称'] = final_result['课程名称'].fillna('未知')
    final_result['开始时间'] = pd.to_datetime(final_result['开始时间'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    final_result['结束时间'] = pd.to_datetime(final_result['结束时间'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    final_result.to_csv(output_file, index=False, encoding='utf-8-sig')

    # 详细信息文件
    detail_output = output_file.replace('.csv', '_详细信息.csv')
    detail_columns = ['平台ID', '课程名称', '教学班ID', '开始时间', '结束时间', '调整原因']
    merged_to_save = merged_result.copy()
    for col in ['开始时间', '结束时间']:
        merged_to_save[col] = pd.to_datetime(merged_to_save[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    merged_to_save[detail_columns].to_csv(detail_output, index=False, encoding='utf-8-sig')

    print(f"\n分析完成！")
    print(f"主要结果已保存到: {output_file}")
    print(f"详细信息已保存到: {detail_output}")
    print(f"最终输出教学班数量: {len(final_result)}")
    print("\n结果预览:")
    print(final_result.head(10))

    return final_result

# 使用示例
if __name__ == "__main__":
    # 指定输入和输出文件路径
    interaction_file = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/学生AI互动.csv"
    course_file = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/课堂开结课时间.csv"
    output_file = "class_time_range_merged.csv"
    
    try:
        # 执行分析
        result = analyze_class_time_range_with_course(interaction_file, course_file, output_file)
        
    except FileNotFoundError as e:
        print(f"错误: 找不到文件")
        print("请确保文件路径正确")
        print(f"详细错误: {str(e)}")
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")