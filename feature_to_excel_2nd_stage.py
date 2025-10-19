import pandas as pd
from openpyxl import load_workbook
import os

def append_csv_to_features_excel(csv_path, excel_path, sheet_name=None, skiprows=0):
    """
    读取 CSV，并将数值列追加到已有 features Excel 后面。
    CSV 的列会被重命名为 transform_{原列名}。
    
    参数：
    - csv_path: CSV 文件路径
    - excel_path: 已存在的 features Excel 文件路径
    - sheet_name: Excel sheet 名称，默认第一个 sheet
    - skiprows: 读取 CSV 时跳过前几行（可根据 CSV 表头调整）
    """
    
    # 1️⃣ 读取 CSV
    df_csv = pd.read_csv(csv_path, skiprows=skiprows)
    
    # 如果第一列是聚类编号或没有列名，可以手动设置列名
    if df_csv.columns[0] is None or df_csv.columns[0] == 0:
        df_csv.rename(columns={df_csv.columns[0]: 'cluster'}, inplace=True)
    
    # 保留数值列
    df_numeric = df_csv.select_dtypes(include='number').copy()
    
    # 重命名列
    df_numeric.columns = [f"transform_{col}" for col in df_numeric.columns]
    
    # 2️⃣ 读取已有 Excel
    if sheet_name:
        df_excel = pd.read_excel(excel_path, sheet_name=sheet_name)
    else:
        df_excel = pd.read_excel(excel_path)
    
    # 3️⃣ 拼接
    df_final = pd.concat([df_excel, df_numeric], axis=1)
    
    # 4️⃣ 保存回 Excel
    df_final.to_excel(excel_path, index=False)
    
    # 5️⃣ 使用 openpyxl 自适应列宽
    wb = load_workbook(excel_path)
    ws = wb.active if not sheet_name else wb[sheet_name]

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
        ws.column_dimensions[column_letter].width = max_length + 2

    wb.save(excel_path)
    print(f"✅ CSV 已追加到 Excel 并保存：{excel_path}")
    print(f"   新增列名示例：{df_numeric.columns.tolist()[:5]}")

    return df_final

