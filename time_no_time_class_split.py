import os
import csv
import shutil
import argparse
from typing import Optional, Set

def try_open_csv(path: str, mode: str = 'r'):
    """
    尝试用常见中文/UTF-8编码打开 CSV 文件，返回 (file_obj, encoding)
    """
    for enc in ('utf-8-sig', 'utf-8', 'gb18030'):
        try:
            f = open(path, mode, encoding=enc, newline='')
            # 试读一行以确认编码没问题
            _ = f.read(1024)
            f.seek(0)
            return f, enc
        except UnicodeDecodeError:
            try:
                f.close()
            except Exception:
                pass
            continue
    # 若都失败，最后一次尝试不指定编码（不推荐，但尽量兜底）
    return open(path, mode, newline=''), None

def load_reference_ids(reference_csv: str, column_name: str = "教学班ID") -> Set[str]:
    """
    从参考 CSV 文件读取教学班ID集合
    """
    f, enc = try_open_csv(reference_csv, 'r')
    with f:
        reader = csv.DictReader(f)
        if column_name not in reader.fieldnames:
            raise ValueError(f"参考文件中缺少列: {column_name}；实际列为: {reader.fieldnames}")
        ids: Set[str] = set()
        for row in reader:
            val = (row.get(column_name) or "").strip()
            if val:
                ids.add(val)
        return ids

def get_first_class_id(dialog_csv: str, column_name: str = "教学班ID") -> Optional[str]:
    """
    从对话 CSV 中读取第一条非空的教学班ID
    """
    f, enc = try_open_csv(dialog_csv, 'r')
    with f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column_name not in reader.fieldnames:
            # 如果字段名不匹配，返回 None
            return None
        for row in reader:
            val = (row.get(column_name) or "").strip()
            if val:
                return val
    return None

def copy_preserving_relpath(src_file: str, src_root: str, dest_root: str):
    """
    复制文件到目标根目录下，并保持相对路径结构不变
    """
    rel = os.path.relpath(src_file, src_root)
    dest_path = os.path.join(dest_root, rel)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(src_file, dest_path)

def main():
    parser = argparse.ArgumentParser(description="按教学班ID是否出现在参考CSV中分类并复制对话CSV")
    parser.add_argument("--input-dir", default="/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/split",
                        help="待遍历的对话CSV根目录")
    parser.add_argument("--reference-csv", default="/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/课堂开结课时间.csv",
                        help="参考CSV，包含列“教学班ID”")
    parser.add_argument("--present-output", default="/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/split_present",
                        help="教学班ID出现在参考CSV中的样本输出根目录")
    parser.add_argument("--missing-output", default="/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/split_missing",
                        help="教学班ID未出现在参考CSV中的样本输出根目录")
    parser.add_argument("--column-name", default="教学班ID",
                        help="教学班ID列名（若你的文件列名不同，可修改）")

    args = parser.parse_args()

    # 加载参考ID集合
    print(f"加载参考CSV: {args.reference_csv}")
    ref_ids = load_reference_ids(args.reference_csv, args.column_name)
    print(f"参考CSV共加载教学班ID数量: {len(ref_ids)}")

    present_count = 0
    missing_count = 0
    error_count = 0
    total_files = 0

    # 遍历输入目录
    for root, dirs, files in os.walk(args.input_dir):
        for name in files:
            if not name.lower().endswith(".csv"):
                continue
            total_files += 1
            src_path = os.path.join(root, name)
            try:
                class_id = get_first_class_id(src_path, args.column_name)
                if not class_id:
                    # 无法读到教学班ID，视为不出现（也可选择丢进错误类别）
                    error_count += 1
                    copy_preserving_relpath(src_path, args.input_dir, args.missing_output)
                    missing_count += 1
                    continue

                if class_id in ref_ids:
                    copy_preserving_relpath(src_path, args.input_dir, args.present_output)
                    present_count += 1
                else:
                    copy_preserving_relpath(src_path, args.input_dir, args.missing_output)
                    missing_count += 1
            except Exception as e:
                print(f"处理文件出错: {src_path}，错误: {e}")
                error_count += 1
                # 出错也放入 missing 目录，避免丢失样本
                try:
                    copy_preserving_relpath(src_path, args.input_dir, args.missing_output)
                    missing_count += 1
                except Exception as e2:
                    print(f"复制出错: {src_path} -> {args.missing_output}，错误: {e2}")

    print("处理完成：")
    print(f"- 总CSV文件数: {total_files}")
    print(f"- 成功分类（出现）: {present_count}")
    print(f"- 成功分类（不出现）: {missing_count}")
    print(f"- 读取或解析出错文件数: {error_count}")
    print(f"- 输出目录（出现）: {args.present_output}")
    print(f"- 输出目录（不出现）: {args.missing_output}")

if __name__ == "__main__":
    main()