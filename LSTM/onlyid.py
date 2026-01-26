import os
import glob
import argparse
from collections import defaultdict
import csv
import re

def extract_file_prefix(file_name):
    """
    从文件名中提取数字和字母组合的前缀
    示例: 
        "1a.csv" -> "1a"
        "23bc.csv" -> "23bc"
        "45.csv" -> "45"
    """
    base_name = os.path.splitext(file_name)[0]  # 移除扩展名
    
    # 使用正则表达式匹配数字+字母的组合
    match = re.match(r'^(\d+[a-zA-Z]*)', base_name)
    if match:
        return match.group(1)
    
    # 如果正则匹配失败，尝试提取数字部分
    prefix = ""
    for char in base_name:
        if char.isdigit() or char.isalpha():
            prefix += char
        else:
            break
    return prefix if prefix else None

def fix_tracking_ids(input_folder, output_folder=None):
    """
    智能修复跟踪ID冲突，使用文件前缀+类别ID+原始ID生成新ID
    :param input_folder: 包含原始CSV文件的文件夹
    :param output_folder: 修复后文件输出文件夹
    """
    # 设置默认输出路径
    if output_folder is None:
        output_folder = os.path.join(input_folder, "fixed")
    
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"警告: 在 {input_folder} 中未找到任何CSV文件")
        return
    
    print(f"正在处理 {len(csv_files)} 个CSV文件...")
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_folder, file_name)
        
        # ====== 从文件名提取前缀（数字+字母） ======
        file_prefix = extract_file_prefix(file_name)
        if not file_prefix:
            print(f"警告: 文件名 {file_name} 中未找到有效前缀，跳过此文件")
            continue
        # ========================================
        
        # 记录每帧已使用的ID（用于冲突检测）
        used_ids_per_frame = defaultdict(set)
        fixed_rows = []
        
        with open(file_path, 'r', newline='') as f_in:
            reader = csv.reader(f_in)
            header = next(reader)  # 读取标题行
            fixed_rows.append(header)
            
            for row_idx, row in enumerate(reader, start=1):
                if len(row) < 3:  # 确保有足够的列
                    fixed_rows.append(row)
                    continue
                
                try:
                    frame = row[0]
                    orig_id = row[1]
                    class_id = row[2]
                    
                    # 生成新ID: [文件前缀][类别ID][原始ID]
                    new_id = f"{file_prefix}{class_id}{orig_id}"
                    
                    # 检查当前帧是否已存在此ID
                    if new_id in used_ids_per_frame[frame]:
                        # 生成冲突解决方案: 添加后缀
                        suffix = 10000
                        conflict_id = f"{new_id}_{suffix}"
                        while conflict_id in used_ids_per_frame[frame]:
                            suffix += 10000
                            conflict_id = f"{new_id}_{suffix}"
                        
                        new_id = conflict_id
                        print(f"冲突修复: {file_name} 帧 {frame} - 新ID {new_id}")
                    
                    # 记录已使用ID
                    used_ids_per_frame[frame].add(new_id)
                    
                    # 更新行数据
                    row[1] = new_id
                    fixed_rows.append(row)
                except Exception as e:
                    print(f"处理文件 {file_name} 第 {row_idx} 行时出错: {e}")
                    print(f"问题行: {row}")
                    fixed_rows.append(row)  # 保留原始行
        
        # 写入修复后的CSV文件
        with open(output_path, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerows(fixed_rows)
        
        print(f"已修复 {file_name} -> 保存到 {output_path}")
    
    print(f"处理完成! 所有修复后的文件保存在 {output_folder}")

def main():
    parser = argparse.ArgumentParser(description='智能修复CSV文件中的跟踪ID冲突')
    parser.add_argument('input_folder', help='包含原始CSV文件的文件夹路径')
    parser.add_argument('-o', '--output', help='修复后文件输出文件夹路径', default=None)
    
    args = parser.parse_args()
    
    fix_tracking_ids(args.input_folder, args.output)

if __name__ == "__main__":
    main()
