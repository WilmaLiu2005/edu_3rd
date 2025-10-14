import csv
from datetime import datetime

def parse_semester_info(info_str):
    """
    解析类似 "2025年2月17日至6月22日，18周" 的字符串，
    提取开始日期、结束日期和周数。
    """
    try:
        # 分割日期部分和周数部分
        date_part, week_part = info_str.split('，')
        
        # 分割开始和结束日期
        start_str, end_str = date_part.split('至')
        
        # 结束日期可能没有年份，从开始日期中提取年份并补全
        year = start_str.split('年')[0]
        if '年' not in end_str:
            end_str = f"{year}年{end_str}"
            
        # 将字符串转换为 datetime.date 对象
        start_date = datetime.strptime(start_str, '%Y年%m月%d日').date()
        end_date = datetime.strptime(end_str, '%Y年%m月%d日').date()
        
        # 提取周数
        num_weeks = int(week_part.split('周')[0])
        
        return start_date, end_date, num_weeks
    except (ValueError, IndexError) as e:
        print(f"解析学期信息时出错: '{info_str}' - 错误: {e}")
        return None, None, None

def main():
    """
    主函数，负责读取、处理和写入 CSV 文件。
    """
    # --- 用户需要配置的部分 ---
    input_csv_path = '/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/学校ID对应基础信息.csv'      # 原始 CSV 文件名
    output_csv_path = '/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/学校ID对应基础信息_new.csv'    # 修改后要保存的新文件名
    # 按顺序提供的学期信息列表
    semester_info_list = [
        "2025年2月17日至6月22日，18周",
        "2025年2月17日至2025年7月6日，20周",
        "2025年2月24日至2025年7月13日，20周",
        "2025年2月24日至2025年6月29日，18周",
        "2025年2月17日至2025年7月6日，20周",
        "2025年2月24日至2025年7月13日，20周",
        "2025年2月24日至2025年6月29日，18周",
        "2025年2月24日至2025年6月29日，18周",
        "2025年2月24日至2025年7月13日，20周",
        "2025年2月24日至2025年7月13日，20周",
    ]
    # --- 配置结束 ---

    # 预处理学期信息，转换为方便使用的格式
    parsed_data = []
    for info in semester_info_list:
        start_date, end_date, num_weeks = parse_semester_info(info)
        if start_date and end_date:
            # 分别格式化起始和结束日期
            start_date_str = start_date.strftime('%Y年%m月%d日')
            end_date_str = end_date.strftime('%Y年%m月%d日')
            parsed_data.append({
                "start_date": start_date_str,
                "end_date": end_date_str,
                "duration": str(num_weeks)
            })

    print(f"脚本开始执行，将读取 '{input_csv_path}' 并生成 '{output_csv_path}'...")

    try:
        with open(input_csv_path, mode='r', encoding='utf-8-sig', newline='') as infile, \
             open(output_csv_path, mode='w', encoding='utf-8-sig', newline='') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # 读取并处理表头
            header = next(reader)
            # 新增三列：起始时间, 结束时间, 持续多少周
            new_header = header + ['起始时间', '结束时间', '持续多少周']
            writer.writerow(new_header)

            # 逐行处理数据
            for i, row in enumerate(reader):
                # 检查是否有对应的学期信息
                if i < len(parsed_data):
                    # 获取当前行对应的学期数据
                    data_to_add = parsed_data[i]
                    # 将三个新数据追加到当前行
                    new_row = row + [data_to_add['start_date'], data_to_add['end_date'], data_to_add['duration']]
                else:
                    # 如果原始CSV行数超过提供的学期信息数，则填充空值
                    print(f"警告：原始文件第 {i+1} 行没有对应的学期信息，已填充空值。")
                    new_row = row + ['', '', '']
                
                writer.writerow(new_row)
        
        print("处理完成！请查看生成的 output_split.csv 文件。")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{input_csv_path}'。请确保文件名正确且文件在同一目录下。")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")

if __name__ == "__main__":
    main()