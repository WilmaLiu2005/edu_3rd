import os
import shutil

# === 教育部学科门类映射表 ===
discipline_map = {
    "材料工程": "工学",
    "电气与电子工程": "工学",
    "核工程": "工学",
    "环境工程": "工学",
    "机械与能源工程": "工学",
    "土木建筑工程": "工学",
    "计算机科学与技术": "工学",
    "化学": "理学",
    "物理学": "理学",
    "数学与统计": "理学",
    "公共卫生": "医学",
    "护理学": "医学",
    "基础医学": "医学",
    "临床医学": "医学",
    "经济与金融": "经济学",
    "人力资源管理": "管理学",
    "教育学": "教育学",
    "心理学": "教育学",
    "人文与文化": "文学",
    "英语与外语": "文学",
    "通识与职业发展": "交叉学科"
}

# === 设置你的根目录路径 ===
root_dir = "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/split"  # ← 改成你要整理的文件夹路径，比如 "/Users/vince/undergraduate/KEG/edu/学堂在线数据3rd/"

# === 遍历子文件夹 ===
for subfolder in os.listdir(root_dir):
    sub_path = os.path.join(root_dir, subfolder)
    if not os.path.isdir(sub_path):
        continue  # 跳过非文件夹
    
    # 判断该文件夹属于哪个学科大类
    category = discipline_map.get(subfolder)
    if category is None:
        print(f"⚠️ 未识别分类：{subfolder} —— 跳过")
        continue
    
    # 创建目标大类文件夹
    category_path = os.path.join(root_dir, category)
    os.makedirs(category_path, exist_ok=True)
    
    # 移动子文件夹
    target_path = os.path.join(category_path, subfolder)
    if os.path.exists(target_path):
        print(f"⚠️ 目标已存在：{target_path} —— 跳过")
    else:
        print(f"📦 移动 {subfolder} → {category}/")
        shutil.move(sub_path, target_path)

print("\n✅ 文件夹合并完成！")