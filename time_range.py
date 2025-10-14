import pandas as pd
import numpy as np
import re

def load_and_clean_school_info(school_info_path, verbose=True):
    """
    读取学校信息表并尽量修正列错位、空列和中文日期格式。
    返回清洗后的 DataFrame，列名至少包含：
      - 平台ID
      - 起始时间 (datetime)
      - 结束时间 (datetime)
      - 持续多少周 (numeric or str)
      - 学校名称 / 省份 / 是否211 / 是否985 (如果存在)
    """
    # 先读原始表，保留所有列（防止列名错位）
    df_raw = pd.read_csv(school_info_path, encoding='utf-8', dtype=str, keep_default_na=False)
    if verbose:
        print("原始学校表列名：", df_raw.columns.tolist())

    # 去掉完全空的列
    non_empty_cols = [c for c in df_raw.columns if not (df_raw[c].astype(str).str.strip() == "").all()]
    df = df_raw[non_empty_cols].copy()
    if verbose and len(non_empty_cols) != len(df_raw.columns):
        print(f"删除 {len(df_raw.columns) - len(non_empty_cols)} 列（全部为空）")

    # 常见列名映射（覆盖不同命名）
    col_map_candidates = {
        '平台ID': ['平台ID', 'platformid', 'platform_id', 'platform'],
        '学校所在省份': ['学校所在省份', '省份', '所在省份'],
        '学校名称': ['学校名称', '起始时间', '办学层次', '清华大学'],  # 有时学校名称混在别列；稍后尝试自动识别
        '是否211': ['是211院校', '是否211', '211'],
        '是否985': ['是985院校', '是否985', '985'],
        '起始时间': ['起始时间', '开始时间', 'start', '起始'],
        '结束时间': ['结束时间', '结束', 'end'],
        '持续多少周': ['持续多少周', '持续周数', '周数']
    }

    # 规范小写列名便于匹配
    lower_map = {c: c.lower() for c in df.columns}
    df.columns = [c.strip() for c in df.columns]

    # 尝试找到“平台ID”列
    platform_col = None
    for c in df.columns:
        if c.strip().lower() == '平台id' or 'platform' in c.strip().lower():
            platform_col = c
            break
    if platform_col is None:
        # 尝试常见首列为平台ID的情况
        platform_col = df.columns[0]
        if verbose:
            print(f"警告：未明确找到 '平台ID' 列，使用第一列 `{platform_col}` 作为平台ID（请核对）")

    # 重命名列：尽量把起始/结束/持续周数映射出来
    rename_map = {}
    for target, candidates in col_map_candidates.items():
        for cand in candidates:
            for col in df.columns:
                if col.strip().lower() == cand.strip().lower():
                    rename_map[col] = target
    # 如果没有直接匹配 '起始时间'/'结束时间'，也尝试根据列值内容推断（例如包含“年”字的列）
    if '起始时间' not in rename_map.values() or '结束时间' not in rename_map.values():
        for col in df.columns:
            sample = " ".join(df[col].astype(str).values[:5])
            # 含有“年”或“-”且可能为日期
            if re.search(r'\d{4}年\d{1,2}月\d{1,2}', sample) or re.search(r'\d{4}-\d{2}-\d{2}', sample):
                # if not already mapped, map to 起始时间 or 结束时间 if one missing
                if '起始时间' not in rename_map.values():
                    rename_map[col] = '起始时间'
                elif '结束时间' not in rename_map.values():
                    rename_map[col] = '结束时间'
    if verbose:
        print("列重命名映射（尝试）：", rename_map)

    df = df.rename(columns=rename_map)

    # 确保我们有平台ID 列名统一为 '平台ID'
    if platform_col != '平台ID':
        df = df.rename(columns={platform_col: '平台ID'})

    # 去掉平台ID 前后空白 & 不可见字符
    df['平台ID'] = df['平台ID'].astype(str).apply(lambda x: x.strip().replace('\ufeff','').replace('\u200b',''))

    # 解析中文日期格式函数
    def parse_mixed_date(x):
        if pd.isna(x) or str(x).strip() == '':
            return pd.NaT
        s = str(x).strip()
        # 常见中文格式：2025年02月17日 或 2025年2月17日
        m = re.match(r'(\d{4})年\s*0?(\d{1,2})月\s*0?(\d{1,2})', s)
        if m:
            y, mm, dd = m.groups()
            try:
                return pd.to_datetime(f"{y}-{int(mm):02d}-{int(dd):02d}")
            except:
                return pd.NaT
        # 常见 yyyy-mm-dd 或 yyyy-mm-dd HH:MM:SS
        try:
            return pd.to_datetime(s, errors='coerce')
        except:
            return pd.NaT

    # 解析起始/结束时间（如果存在）
    for col in ['起始时间', '结束时间']:
        if col in df.columns:
            df[col + '_parsed'] = df[col].apply(parse_mixed_date)
        else:
            df[col + '_parsed'] = pd.NaT

    # 如果有“持续多少周”列名不同，尝试解析
    if '持续多少周' not in df.columns:
        # find numeric-like column as weeks
        for col in df.columns:
            if col in ['起始时间','结束时间','平台ID']: 
                continue
            sample_vals = df[col].astype(str).str.extract(r'(\d{1,2})', expand=False).dropna()
            if len(sample_vals) > 0 and sample_vals.str.len().mean() <= 2:
                df = df.rename(columns={col: '持续多少周'})
                break

    # 归一化最终列名与返回
    out_cols = ['平台ID', '起始时间_parsed', '结束时间_parsed', '持续多少周']
    for c in out_cols:
        if c not in df.columns:
            df[c] = pd.NaT if '时间' in c else np.nan

    # 清理并保留其他有用信息（省份/学校名称/是否211/是否985）
    extra_cols = []
    for cand in ['学校所在省份','学校名称','是211院校','是985院校','办学层次']:
        for col in df.columns:
            if col.strip().lower() == cand.strip().lower():
                extra_cols.append(col)
                break
    # 把原来的 extra cols 也重命名到规范名（如果存在）
    for col in extra_cols:
        df = df.rename(columns={col: col.strip()})

    # 最终构造干净表
    clean = pd.DataFrame({
        '平台ID': df['平台ID'].astype(str).str.strip(),
        '起始时间': df['起始时间_parsed'],
        '结束时间': df['结束时间_parsed'],
        '持续多少周': df.get('持续多少周', pd.Series([np.nan]*len(df))),
    })
    # 把 extra 信息合并进去（如果存在）
    for c in ['学校所在省份','学校名称','是211院校','是985院校','办学层次']:
        if c in df.columns:
            clean[c] = df[c].replace('', np.nan)

    # 打印诊断信息
    if verbose:
        print("清洗后学校表预览：")
        print(clean.head(10))
        missing_dates = clean['起始时间'].isna().sum() + clean['结束时间'].isna().sum()
        print(f"起止时间缺失项总计(起或止)：{((clean['起始时间'].isna()) | (clean['结束时间'].isna())).sum()} / {len(clean)}")
    return clean

def assign_school_time_to_classes_from_course(course_file, school_info_file, output_file):
    """
    使用课堂开结课文件，每条记录按其平台ID 使用学校表的统一起止时间（学校级），
    学校表的读取使用更健壮的 load_and_clean_school_info()。
    """
    course_df = pd.read_csv(course_file, encoding='utf-8', dtype=str)
    school_clean = load_and_clean_school_info(school_info_file, verbose=True)

    # 标准化 course_df 的平台ID
    course_df['平台ID'] = course_df['平台ID'].astype(str).apply(lambda x: x.strip().replace('\ufeff','').replace('\u200b',''))
    # 保留原始课程列
    result_rows = []
    default_start = pd.to_datetime('2025-03-01 00:00:00')
    default_end = pd.to_datetime('2025-07-07 23:59:59')

    # 建立平台ID -> 学校信息的快速查找字典
    school_by_pid = school_clean.set_index('平台ID').to_dict(orient='index')

    unmatched_pids = set()
    for _, r in course_df.iterrows():
        pid = r.get('平台ID', '')
        course_name = r.get('课程名称', '')
        class_id = r.get('教学班ID', '')

        info = school_by_pid.get(pid)
        if info is not None:
            start = info.get('起始时间')
            end = info.get('结束时间')
            weeks = info.get('持续多少周', np.nan)
            reason = "使用学校统一时间"
            if pd.isna(start) or pd.isna(end):
                start, end = default_start, default_end
                reason = "学校时间缺失 -> 使用默认区间"
        else:
            # 未匹配到平台ID
            start, end, weeks = default_start, default_end, np.nan
            reason = "平台未出现在学校信息表 -> 使用默认区间"
            unmatched_pids.add(pid)

        result_rows.append({
            '平台ID': pid,
            '课程名称': course_name,
            '教学班ID': class_id,
            '开始时间': pd.to_datetime(start).strftime('%Y-%m-%d %H:%M:%S'),
            '结束时间': pd.to_datetime(end).strftime('%Y-%m-%d %H:%M:%S'),
            '持续多少周': weeks if pd.notna(weeks) else '未知',
            '调整原因': reason
        })

    res_df = pd.DataFrame(result_rows)
    res_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n输出已保存: {output_file}")
    if unmatched_pids:
        print(f"未匹配到 {len(unmatched_pids)} 个 平台ID（示例 10 个）：")
        example = list(unmatched_pids)[:10]
        print(example)
    else:
        print("所有平台ID均已在学校信息表中匹配到（或有时间字段缺失但平台存在）。")

    return res_df

# 调用示例（请替换为你自己的路径）
course_file = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/课堂开结课时间.csv"
school_info = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/学校ID对应基础信息_new.csv"
out = "class_time_range_by_school.csv"
df_out = assign_school_time_to_classes_from_course(course_file, school_info, out)
