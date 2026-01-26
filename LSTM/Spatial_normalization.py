#!/usr/bin/env python3
import os
import pandas as pd
import argparse

def process_csv(file_path, output_folder, video_width, video_height, threshold=820, subtract_amount=860):
    """处理单个CSV文件：
       - 如果 x1 或 x2 > threshold，则减去 subtract_amount
       - 重新计算 w/h、中心点、归一化中心点
    """
    try:
        # 读取CSV文件（支持逗号或制表符）
        df = pd.read_csv(file_path, sep=r'\t|,', engine='python', dtype=str)
        # 清理列名空白
        df.columns = df.columns.str.strip()

        # 确保必须的列存在
        required_cols = ['x1', 'y1', 'x2', 'y2']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"[跳过] 文件 {os.path.basename(file_path)} 缺少列: {missing}")
            return

        # 把相关列转换为数值（出错时置为 NaN）
        for col in ['x1', 'y1', 'x2', 'y2', 'w', 'h', 'conf']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 修正 x1, x2：如果 > threshold，则减去 subtract_amount
        df['x1'] = df['x1'].where(df['x1'] <= threshold, df['x1'] - subtract_amount)
        df['x2'] = df['x2'].where(df['x2'] <= threshold, df['x2'] - subtract_amount)

        # 重新计算 w, h（如果存在则覆盖；否则创建）
        df['w'] = df['x2'] - df['x1']
        df['h'] = df['y2'] - df['y1']

        # 计算中心点（基于修正后的 x1,x2）
        df['center_x'] = (df['x1'] + df['x2']) / 2
        df['center_y'] = (df['y1'] + df['y2']) / 2

        # 归一化中心点
        if video_width == 0 or video_height == 0:
            raise ValueError("video_width 和 video_height 必须为非零整数。")

        df['normalized_x'] = df['center_x'] / float(video_width)
        df['normalized_y'] = df['center_y'] / float(video_height)

        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)

        # 输出文件路径
        output_path = os.path.join(output_folder, os.path.basename(file_path))

        # 检查是否含有 NaN 并提示
        nan_counts = df[['x1','x2','y1','y2','center_x','center_y']].isna().sum()
        if nan_counts.any():
            print(f"[警告] {os.path.basename(file_path)} 存在 NaN 值：\n{nan_counts[nan_counts>0].to_dict()}")

        # 保存结果
        df.to_csv(output_path, index=False, sep=',', float_format='%.6f')
        print(f"处理完成: {file_path} -> {output_path}")

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='批量处理目标检测CSV数据（修正 x1/x2 并计算中心点与归一化）')
    parser.add_argument('input_folder', help='包含CSV文件的输入文件夹路径')
    parser.add_argument('--video_width', type=int, default=1688, help='视频宽度（像素）')
    parser.add_argument('--video_height', type=int, default=1248, help='视频高度（像素）')
    parser.add_argument('--output', default='processed', help='输出文件夹路径（默认为 processed）')
    parser.add_argument('--threshold', type=float, default=820.0, help='x 大于此阈值时进行减法（默认 820）')
    parser.add_argument('--subtract', type=float, default=860.0, help='当 x > threshold 时减去的值（默认 860）')
    args = parser.parse_args()

    # 遍历文件夹中的CSV文件
    any_files = False
    for file_name in os.listdir(args.input_folder):
        if file_name.lower().endswith('.csv'):
            any_files = True
            file_path = os.path.join(args.input_folder, file_name)
            process_csv(file_path, args.output, args.video_width, args.video_height,
                        threshold=args.threshold, subtract_amount=args.subtract)

    if not any_files:
        print("输入文件夹中未找到任何 .csv 文件。")
    else:
        print(f"\n所有文件处理完成！输出目录: {args.output}")

if __name__ == "__main__":
    main()
