import supervision as sv
from ultralytics import YOLO 
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import os
from datetime import datetime

def process_video(
        source_weights_path: str, 
        source_video_path: str,
        target_video_dir: str,  # 改为目录参数
        output_csv_dir: str,     # 改为目录参数
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7
) -> None:
    # 加载YOLOv8模型
    model = YOLO(source_weights_path)
    
    # 创建输出目录
    os.makedirs(target_video_dir, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)
    
    # 生成带时间戳的输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.basename(source_video_path)
    output_video_path = os.path.join(target_video_dir, f"tracked_{timestamp}_{video_filename}")
    output_csv_path = os.path.join(output_csv_dir, f"trajectories_{timestamp}_{video_filename}.csv")
    
    # 初始化组件（使用更新后的类名）
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()  # 使用更新后的类名
    label_annotator = sv.LabelAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    
    # 准备CSV输出
    trajectory_data = []
    csv_columns = ["frame_number", "track_id", "class_name", 
                   "x_center", "y_center", "width", "height", "confidence"]
    
    with sv.VideoSink(target_path=output_video_path, video_info=video_info) as sink:
        for frame_number, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):
            # YOLO推理
            results = model(frame, verbose=False, conf=confidence_threshold, iou=iou_threshold)[0]
            
            # 获取检测结果
            detections = sv.Detections.from_ultralytics(results)
            
            # 更新追踪器
            detections = tracker.update_with_detections(detections)
            
            # 记录轨迹数据
            for i in range(len(detections)):
                if detections.tracker_id[i] is None:
                    continue  # 跳过未追踪的目标
                    
                bbox = detections.xyxy[i]
                class_id = detections.class_id[i]
                class_name = model.names[class_id]
                
                # 计算边界框中心点
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                trajectory_data.append([
                    frame_number,
                    detections.tracker_id[i],
                    class_name,
                    x_center,
                    y_center,
                    width,
                    height,
                    detections.confidence[i]
                ])
            
            # 可视化
            labels = [
                f"#{track_id} {model.names[class_id]}"
                for class_id, track_id
                in zip(detections.class_id, detections.tracker_id)
            ]
            
            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), 
                detections=detections
            )
            
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )
            
            sink.write_frame(frame=annotated_frame)
    
    # 保存轨迹数据到CSV
    if trajectory_data:
        df = pd.DataFrame(trajectory_data, columns=csv_columns)
        df.to_csv(output_csv_path, index=False)
        print(f"轨迹数据已保存至: {output_csv_path}")
    else:
        print("未检测到任何鱼类轨迹")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("鱼类追踪分析系统") 
    parser.add_argument("--source_weights_path", required=True, help="YOLO模型权重路径", type=str)
    parser.add_argument("--source_video_path", required=True, help="输入视频路径", type=str)
    parser.add_argument("--target_video_dir", required=True, help="输出视频目录", type=str)
    parser.add_argument("--output_csv_dir", required=True, help="轨迹数据CSV输出目录", type=str)
    parser.add_argument("--confidence_threshold", default=0.3, help="检测置信度阈值", type=float)
    parser.add_argument("--iou_threshold", default=0.7, help="NMS的IoU阈值", type=float)
    
    args = parser.parse_args()
    
    process_video(
        source_weights_path=args.source_weights_path, 
        source_video_path=args.source_video_path,
        target_video_dir=args.target_video_dir,
        output_csv_dir=args.output_csv_dir,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold
    )