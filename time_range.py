import pandas as pd
import numpy as np
import re

def load_and_clean_school_info(school_info_path, verbose=True):
    """
    读取学校信息表，尽量修正列错位、空列、中文日期等问题。
    若原表无 起始时间/结束时间/持续多少周，则自动新增这些列。
    返回清洗后的 DataFrame（至少包含以下列）：
      - 平台ID
      - 起始时间 (datetime)
      - 结束时间 (datetime)
      - 持续多少周 (numeric or str)
    """
    df_raw = pd.read_csv(school_info_path, encoding='utf-8', dtype=str, keep_default_na=False)
    if verbose:
        print("原始学校表列名：", df_raw.columns.tolist())

    # 去掉全空列
    non_empty_cols = [c for c in df_raw.columns if not (df_raw[c].astype(str).str.strip() == "").all()]
    df = df_raw[non_empty_cols].copy()

    # === 识别“平台ID”列 ===
    platform_col = None
    for c in df.columns:
        if re.search(r'平台|platform', c, re.IGNORECASE):
            platform_col = c
            break
    if platform_col is None:
        platform_col = df.columns[0]
        if verbose:
            print(f"⚠️ 未找到平台ID列，默认使用首列 {platform_col}")

    df = df.rename(columns={platform_col: '平台ID'})
    df['平台ID'] = df['平台ID'].astype(str).str.strip().replace({'\ufeff':'', '\u200b':''}, regex=True)

    # === 尝试找到可能含日期的列（例如包含“年”或“-”） ===
    date_candidates = []
    for col in df.columns:
        sample = " ".join(df[col].astype(str).values[:5])
        if re.search(r'\d{4}年|\d{4}-\d{2}', sample):
            date_candidates.append(col)
    if verbose and date_candidates:
        print("可能包含日期的列：", date_candidates)

    # === 定义日期解析函数 ===
    def parse_mixed_date(x):
        if pd.isna(x) or str(x).strip() == '':
            return pd.NaT
        s = str(x).strip()
        m = re.match(r'(\d{4})年\s*0?(\d{1,2})月\s*0?(\d{1,2})', s)
        if m:
            y, mm, dd = m.groups()
            return pd.to_datetime(f"{y}-{int(mm):02d}-{int(dd):02d}", errors='coerce')
        return pd.to_datetime(s, errors='coerce')

    # === 如果没有明确列名，就从候选列里取两列当起止时间 ===
    if '起始时间' not in df.columns:
        if len(date_candidates) >= 1:
            df['起始时间'] = df[date_candidates[0]]
        else:
            df['起始时间'] = np.nan
    if '结束时间' not in df.columns:
        if len(date_candidates) >= 2:
            df['结束时间'] = df[date_candidates[-1]]
        else:
            df['结束时间'] = np.nan
    if '持续多少周' not in df.columns:
        df['持续多少周'] = np.nan

    # === 解析为 datetime ===
    for col in ['起始时间', '结束时间']:
        df[col] = df[col].apply(parse_mixed_date)

    clean = df[['平台ID', '起始时间', '结束时间', '持续多少周']].copy()

    if verbose:
        print("清洗后学校表预览：")
        print(clean.head())
        print(f"🕓 时间缺失：{((clean['起始时间'].isna()) | (clean['结束时间'].isna())).sum()} / {len(clean)} 行")
    return clean


def assign_school_time_to_classes_from_course(course_file, school_info_file, output_file):
    """
    从课程表中根据 平台ID 匹配学校统一起止时间。
    若匹配不到，则使用默认区间。
    """
    course_df = pd.read_csv(course_file, encoding='utf-8', dtype=str)
    school_clean = load_and_clean_school_info(school_info_file, verbose=True)

    # 标准化平台ID
    course_df['平台ID'] = course_df['平台ID'].astype(str).str.strip().replace({'\ufeff':'', '\u200b':''}, regex=True)
    school_clean['平台ID'] = school_clean['平台ID'].astype(str).str.strip().replace({'\ufeff':'', '\u200b':''}, regex=True)

    default_start = pd.to_datetime('2025-03-01 00:00:00')
    default_end = pd.to_datetime('2025-07-07 23:59:59')

    school_dict = school_clean.set_index('平台ID').to_dict(orient='index')

    result_rows = []
    unmatched_pids = set()

    for _, row in course_df.iterrows():
        pid = row.get('平台ID', '')
        info = school_dict.get(pid)

        if info:
            start = info.get('起始时间')
            end = info.get('结束时间')
            weeks = info.get('持续多少周', np.nan)
            reason = "使用学校统一时间"
            if pd.isna(start) or pd.isna(end):
                start, end = default_start, default_end
                reason = "学校时间缺失 -> 使用默认区间"
        else:
            start, end, weeks = default_start, default_end, np.nan
            reason = "平台未出现在学校信息表 -> 使用默认区间"
            unmatched_pids.add(pid)

        # 输出保留原字段 + 时间字段
        result = row.to_dict()
        result.update({
            '起始时间': pd.to_datetime(start).strftime('%Y-%m-%d %H:%M:%S'),
            '结束时间': pd.to_datetime(end).strftime('%Y-%m-%d %H:%M:%S'),
            '持续多少周': weeks if pd.notna(weeks) else '未知',
            '调整原因': reason
        })
        result_rows.append(result)

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n✅ 输出已保存: {output_file}")
    if unmatched_pids:
        print(f"⚠️ 未匹配到 {len(unmatched_pids)} 个平台ID（示例 10 个）:")
        print(list(unmatched_pids)[:10])
    else:
        print("🎯 所有平台ID均匹配成功。")

    return result_df


# 示例调用
course_file = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/班级情况.csv"
school_info = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/学校ID对应基础信息_new.csv"
out = "class_time_range_by_school.csv"

df_out = assign_school_time_to_classes_from_course(course_file, school_info, out)
