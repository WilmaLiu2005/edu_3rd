# 先时间间隔15分钟，然后再LLM划分
import pandas as pd
import numpy as np
import os
import json
import json5
import re
import time
import random
from datetime import datetime
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
# ==============================    

MAX_WORKERS = 15
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 3
API_KEY = "sk-0jErqj61bIYM135CEqhfj318rKIM1TIa"  # 填写你的 API key
BASE_URL = "https://api-gateway.glm.ai/v1"
MODEL_NAME = "gemini-2.5-flash"  # 或你自己的模型
INPUT_FOLDER = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/processed_split_field_output"   # 输入 CSV 文件夹
OUTPUT_BASE_FOLDER = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/split" # 输出 CSV 文件夹

# ==============================
# 工具函数
# ==============================
os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)

def robust_read_csv(file_path, text_columns=None):
    """
    安全读取 CSV 文件，适用于含有换行符、Markdown 图片链接、双引号等情况。
    """
    try:
        df = pd.read_csv(
            file_path,
            encoding="utf-8-sig",
            engine="python",
            quotechar='"',
            doublequote=True,
            keep_default_na=False,
        )
    except Exception as e:
        print(f"⚠️ 读取 CSV 失败: {file_path}, 错误: {e}")
        return None
    
    if df.empty:
        print(f"⚠️ 文件 {file_path} 是空的")
        return None
    
    # 去除列名前后空格
    df.columns = df.columns.str.strip()
    
    # 检查必要列
    required_cols = ["学生ID", "提问时间"]
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️ 文件 {file_path} 缺少必要列: {col}")
            return None
    
    # 处理文本列换行符
    if text_columns:
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace("\n", " ", regex=False)
    
    # 转换时间列
    df["提问时间"] = pd.to_datetime(df["提问时间"], errors="coerce")
    df = df.dropna(subset=["提问时间"])
    
    if df.empty:
        print(f"⚠️ 文件 {file_path} 全部行无效（提问时间解析失败）")
        return None
    
    return df

# ==============================
# GPT API 调用
# ==============================
def gpt_api_call(messages, model=MODEL_NAME):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=10000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                delay = min(INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_RETRY_DELAY)
                time.sleep(delay)
            else:
                return None

# ==============================
# JSON 解析（增强鲁棒性）
# ==============================
def robust_json_parse(text):
    """
    鲁棒的JSON解析，能处理各种格式的返回
    """
    if not text:
        return {}
    
    # 清理文本
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    
    try:
        result = json.loads(cleaned)
    except Exception:
        try:
            result = json5.loads(cleaned)
        except Exception:
            print(f"    ⚠️ JSON解析失败，原始输出: {text[:200]}")
            return {}
    
    # 处理不同格式的返回
    if isinstance(result, dict):
        return result
    elif isinstance(result, list):
        # 如果是列表格式 [1, 1, 2, 2, 3]，转换为字典
        print(f"    ⚠️ LLM返回了列表格式，尝试转换")
        converted = {}
        for idx, session_id in enumerate(result, 1):
            converted[str(idx)] = session_id
        return converted
    else:
        print(f"    ⚠️ 未知的返回格式: {type(result)}")
        return {}
        
PROMPT_TEMPLATE = """
You are a dialogue segmentation expert. Analyze the following student-AI tutor interaction logs and segment them into coherent dialogue sessions.

Input: Chronologically ordered interactions with:
- Row number
- Timestamp  
- Student question
- AI response

Segmentation criteria:
1. Preserve chronological order strictly
2. Split when time gap exceeds reasonable conversation pause
3. Split when topic shifts significantly
4. Keep related exchanges in same session

Output: JSON mapping{{row_number: session_id}}, session_id starting from 1
No additional text or explanation.

Interaction logs:
{dialogues_json}
"""
# ==============================
# Stage 0: 提问入口划分（新增）
# ==============================
def entrance_based_split(df):
    """
    根据提问入口的变化进行划分
    当提问入口发生变化时，认为是不同的对话场景
    """
    results = []
    if df.empty:
        return results
    
    # 确保按时间排序
    df = df.sort_values("提问时间").reset_index(drop=True)
    
    current_chunk = [df.iloc[0]]
    current_entrance = df.iloc[0].get("提问入口", "")
    
    for i in range(1, len(df)):
        row_entrance = df.iloc[i].get("提问入口", "")
        
        # 如果提问入口发生变化，则分割
        if row_entrance != current_entrance:
            results.append(pd.DataFrame(current_chunk))
            current_chunk = [df.iloc[i]]
            current_entrance = row_entrance
        else:
            current_chunk.append(df.iloc[i])
    
    # 添加最后一个chunk
    if current_chunk:
        results.append(pd.DataFrame(current_chunk))
    
    print(f"  - 按提问入口划分为 {len(results)} 个片段")
    for idx, chunk in enumerate(results, 1):
        entrance = chunk.iloc[0].get("提问入口", "未知")
        print(f"    片段{idx}: 入口={entrance}, 记录数={len(chunk)}")
    
    return results

# ==============================
# Stage 1: 时间划分（修改后）
# ==============================
def time_based_split(df, time_threshold=15):
    """
    在同一个提问入口内，根据时间间隔进行划分
    """
    results = []
    if df.empty:
        return results
    
    current_chunk = [df.iloc[0]]
    for i in range(1, len(df)):
        delta = (df.iloc[i]["提问时间"] - df.iloc[i - 1]["提问时间"]).total_seconds() / 60
        if delta > time_threshold:
            results.append(pd.DataFrame(current_chunk))
            current_chunk = [df.iloc[i]]
        else:
            current_chunk.append(df.iloc[i])
    
    if current_chunk:
        results.append(pd.DataFrame(current_chunk))
    
    return results

# ==============================
# Stage 2: LLM 划分
# ==============================
def llm_split(group_df):
    """使用LLM对对话进行细分"""
    try:
        dialogues = []
        
        # 检查必要的列
        required_columns = ['提问时间', '提问内容', 'AI回复']
        missing_columns = [col for col in required_columns if col not in group_df.columns]
        
        if missing_columns:
            print(f"    ⚠️ 缺少必要的列: {missing_columns}，跳过LLM划分")
            return {}
        
        # 构建对话数据
        for idx in range(len(group_df)):
            row = group_df.iloc[idx]
            dialogues.append({
                "row_number": idx + 1,
                "timestamp": str(row["提问时间"]),
                "question": str(row["提问内容"])[:500],
                "ai_response": str(row["AI回复"])[:500]
            })
        
        dialogues_json = json.dumps(dialogues, ensure_ascii=False, indent=2)
        
        # 调用LLM
        prompt = PROMPT_TEMPLATE.format(dialogues_json=dialogues_json)
        
        messages = [
            {"role": "system", "content": "You are a dialogue segmentation expert. Always output JSON dictionary format, never array."},
            {"role": "user", "content": prompt}
        ]
        
        response = gpt_api_call(messages)
        
        if not response:
            print("    ⚠️ LLM无响应")
            return {}
        
        # 解析结果
        parsed_result = robust_json_parse(response)
        
        # 再次验证是否为字典
        if not isinstance(parsed_result, dict):
            print(f"    ⚠️ 解析后仍不是字典格式: {type(parsed_result)}")
            return {}
        
        # 确保所有键都是字符串，值都是整数
        result = {}
        for key, value in parsed_result.items():
            try:
                result[str(key)] = int(value)
            except (ValueError, TypeError) as e:
                print(f"    ⚠️ 转换键值对失败: {key}={value}, 错误: {e}")
                continue
        
        return result
        
    except Exception as e:
        print(f"    ❌ LLM划分出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

# ==============================
# 处理单个学生的数据（增强错误处理）
# ==============================
def process_student_data(student_df, student_id, output_folder):
    """处理单个学生的所有对话数据"""
    # 按时间排序
    student_df = student_df.sort_values("提问时间").reset_index(drop=True)
    
    print(f"  开始处理学生 {student_id} 的数据...")
    print(f"  - 总记录数: {len(student_df)}")
    
    # Stage 0: 先按提问入口划分（如果有这个列）
    if "提问入口" in student_df.columns:
        entrance_splits = entrance_based_split(student_df)
    else:
        print("  - 没有'提问入口'列，跳过入口划分")
        entrance_splits = [student_df]
    
    file_index = 1  # 该学生的对话编号
    
    # 对每个入口片段进行后续处理
    for entrance_idx, entrance_group in enumerate(entrance_splits, start=1):
        if "提问入口" in entrance_group.columns:
            entrance_name = entrance_group.iloc[0].get("提问入口", "未知")
            print(f"\n  处理入口片段 {entrance_idx}/{len(entrance_splits)}: {entrance_name}")
        
        # Stage 1: 时间间隔划分
        time_splits = time_based_split(entrance_group)
        print(f"    - 时间划分为 {len(time_splits)} 个子片段")
        
        # Stage 2: LLM 划分
        for time_idx, group in enumerate(time_splits, start=1):
            # 重置索引
            group = group.reset_index(drop=True)
            
            # 如果片段太小，直接保存
            if len(group) <= 2:
                output_csv = os.path.join(output_folder, f"{student_id}_{file_index}.csv")
                group.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
                print(f"    ✅ 片段较小，直接保存: {student_id}_{file_index}.csv, 共 {len(group)} 行")
                file_index += 1
                continue
            
            # 使用LLM进行更细致的划分
            mapping = llm_split(group)
            
            # 检查mapping是否有效
            if not mapping or not isinstance(mapping, dict):
                # LLM划分失败或返回格式错误，整个时间片作为一个对话保存
                print(f"    ⚠️ LLM划分无效，整体保存")
                output_csv = os.path.join(output_folder, f"{student_id}_{file_index}.csv")
                group.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
                print(f"    ✅ 已生成 {student_id}_{file_index}.csv, 共 {len(group)} 行")
                file_index += 1
                continue
            
            # 将 LLM 输出的 session_id 映射到对话
            session_ids = []
            prev_session = 1
            
            for idx in range(len(group)):
                key = str(idx + 1)  # 行号从 1 开始
                if key in mapping:
                    sid = mapping[key]
                else:
                    # 如果映射中没有这个键，使用前一个session id
                    print(f"    ⚠️ 映射中缺少键 '{key}'，使用前一个session ID")
                    sid = prev_session
                
                session_ids.append(sid)
                prev_session = sid
            
            # 统计session数量
            unique_sessions = len(set(session_ids))
            print(f"    - LLM划分为 {unique_sessions} 个对话")
            
            # 根据 session_id 切分子会话
            current_session = []
            current_id = session_ids[0]
            
            for idx, sid in enumerate(session_ids):
                if sid != current_id:
                    # 保存上一段子会话
                    if current_session:  # 确保不为空
                        sub_group = group.iloc[current_session].copy().reset_index(drop=True)
                        output_csv = os.path.join(output_folder, f"{student_id}_{file_index}.csv")
                        sub_group.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
                        print(f"    ✅ 已生成 {student_id}_{file_index}.csv, 共 {len(sub_group)} 行")
                        file_index += 1
                    
                    # 开启新子会话
                    current_session = [idx]
                    current_id = sid
                else:
                    current_session.append(idx)
            
            # 保存最后一段子会话
            if current_session:
                sub_group = group.iloc[current_session].copy().reset_index(drop=True)
                output_csv = os.path.join(output_folder, f"{student_id}_{file_index}.csv")
                sub_group.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
                print(f"    ✅ 已生成 {student_id}_{file_index}.csv, 共 {len(sub_group)} 行")
                file_index += 1
    
    return file_index - 1  # 返回生成的对话数量

# ==============================
# 处理单个 CSV 文件
# ==============================
def process_csv_file(file_path):
    """处理单个CSV文件，按学生ID分组并分别处理"""
    try:
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 创建对应的输出文件夹
        output_folder = os.path.join(OUTPUT_BASE_FOLDER, base_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"📁 开始处理文件: {base_name}")
        print(f"📂 输出文件夹: {output_folder}")
        print(f"{'='*60}")
        
        # 读取 CSV
        df = robust_read_csv(file_path, text_columns=["提问内容", "AI回复"])
        if df is None or df.empty:
            print(f"⚠️ {file_path} 无有效数据，跳过")
            return
        
        # 按学生ID分组
        student_groups = df.groupby("学生ID")
        total_students = len(student_groups)
        
        print(f"📊 发现 {total_students} 个学生的对话记录")
        
        total_dialogues = 0
        
        # 处理每个学生的数据
        for student_idx, (student_id, student_df) in enumerate(student_groups, start=1):
            print(f"\n[{student_idx}/{total_students}] 处理学生 {student_id} 的数据...")
            print(f"  - 总记录数: {len(student_df)}")
            
            # 处理该学生的所有对话
            dialogue_count = process_student_data(student_df, student_id, output_folder)
            total_dialogues += dialogue_count
            
            print(f"  - 生成对话数: {dialogue_count}")
        
        print(f"\n✅ 文件 {base_name} 处理完成！")
        print(f"  - 学生数: {total_students}")
        print(f"  - 总对话数: {total_dialogues}")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理文件 {file_path} 出错: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==============================
# 主程序
# ==============================
def main():
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")]
    
    if not csv_files:
        print("⚠️ 未找到CSV文件")
        return
    
    print(f"🎯 找到 {len(csv_files)} 个CSV文件待处理")
    print(f"📁 输入文件夹: {INPUT_FOLDER}")
    print(f"📂 输出基础文件夹: {OUTPUT_BASE_FOLDER}")
    print(f"{'='*60}\n")
    
    success_count = 0
    failed_files = []
    
    # 使用线程池处理文件
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_csv_file, os.path.join(INPUT_FOLDER, f)): f 
            for f in csv_files
        }
        
        # 等待任务完成
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                else:
                    failed_files.append(file_name)
            except Exception as e:
                print(f"❌ 处理 {file_name} 出错: {e}")
                failed_files.append(file_name)
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"🎉 所有文件处理完成！")
    print(f"✅ 成功: {success_count}/{len(csv_files)}")
    
    if failed_files:
        print(f"❌ 失败的文件:")
        for f in failed_files:
            print(f"  - {f}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()