import os
import pandas as pd
import argparse

def process_csv(file_path, output_folder, video_width, video_height):
    """处理单个CSV文件：计算中心点并归一化"""
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path, sep='\t|,', engine='python')
        
        # 计算原始中心点坐标
        df['center_x'] = (df['x1'] + df['x2']) / 2
        df['center_y'] = (df['y1'] + df['y2']) / 2
        
        # 计算归一化中心点坐标
        df['normalized_x'] = df['center_x'] / video_width
        df['normalized_y'] = df['center_y'] / video_height
        
        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)
        
        # 保存处理后的文件
        output_path = os.path.join(output_folder, os.path.basename(file_path))
        df.to_csv(output_path, index=False, sep=',')
        print(f"处理完成: {file_path} -> {output_path}")
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='批量处理目标检测CSV数据')
    parser.add_argument('input_folder', help='包含CSV文件的输入文件夹路径')
    parser.add_argument('video_width', type=int, help='视频宽度（像素）')
    parser.add_argument('video_height', type=int, help='视频高度（像素）')
    parser.add_argument('--output', default='processed', help='输出文件夹路径（默认为processed）')
    
    args = parser.parse_args()
    
    # 遍历文件夹中的CSV文件
    for file_name in os.listdir(args.input_folder):
        if file_name.lower().endswith('.csv'):
            file_path = os.path.join(args.input_folder, file_name)
            process_csv(file_path, args.output, args.video_width, args.video_height)
    
    print(f"\n所有文件处理完成！输出目录: {args.output}")

if __name__ == "__main__":
    main()