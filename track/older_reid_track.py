import cv2
import time
import os
import numpy as np
import torch
from ultralytics import YOLO
from loguru import logger
from collections import defaultdict
from torchreid.utils import FeatureExtractor
from scipy.optimize import linear_sum_assignment

# 确保输出目录存在
os.makedirs("videos", exist_ok=True)

# 初始化YOLOv8检测模型
det_model = YOLO("/home/bwang/yolov8-main/runs/detect/train3/weights/best.pt")

# 初始化ReID特征提取模型
reid_model = FeatureExtractor(
    model_name='osnet_x0_25',
    model_path='osnet_x0_25_msmt17.pt',
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
)

# 视频路径
video_path = "/home/bwang/备份/12.mp4"

# 创建轨迹文件
track_file = "/home/bwang/tracks.txt"
with open(track_file, "w") as f:
    f.write("frame_id,track_id,class_id,conf,x1,y1,x2,y2\n")
logger.info(f"轨迹数据将保存至: {track_file}")

# 打开视频文件
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    logger.error(f"无法打开视频文件: {video_path}")
    exit()

# 获取视频参数
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 创建输出视频
out = cv2.VideoWriter(
    "videos/output_reid.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'), 
    fps, 
    (width, height)
)

# 自定义跟踪器类（整合ReID的ByteTrack）
class ReIDByteTracker:
    def __init__(self, max_age=30, iou_threshold=0.3, emb_threshold=0.7):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.emb_threshold = emb_threshold
        self.tracks = []
        self.next_id = 1
        self.track_history = defaultdict(list)
        self.track_embeddings = {}
        
    def update(self, detections, embeddings):
        """更新跟踪器状态"""
        # 为现有轨迹分配ID
        active_tracks = [t for t in self.tracks if t['lost'] < self.max_age]
        
        # 如果没有检测结果，更新所有轨迹的丢失计数
        if len(detections) == 0:
            for track in active_tracks:
                track['lost'] += 1
            return []
        
        # 计算IOU距离矩阵
        iou_matrix = np.zeros((len(active_tracks), len(detections)))
        for i, track in enumerate(active_tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self.compute_iou(track['bbox'], det[:4])
        
        # 计算外观距离矩阵
        emb_matrix = np.zeros((len(active_tracks), len(detections)))
        for i, track in enumerate(active_tracks):
            for j, emb in enumerate(embeddings):
                # 如果轨迹有历史特征，计算平均相似度
                if track['id'] in self.track_embeddings:
                    track_emb = np.mean(self.track_embeddings[track['id']], axis=0)
                    emb_matrix[i, j] = np.dot(track_emb, emb) / (
                        np.linalg.norm(track_emb) * np.linalg.norm(emb))
                else:
                    # 新轨迹没有历史特征，设为最低相似度
                    emb_matrix[i, j] = 0
        
        # 融合距离矩阵（IOU和外观相似度）
        # 权重可以根据场景调整 (0.7表示更依赖IOU)
        combined_matrix = 0.7 * iou_matrix + 0.3 * emb_matrix
        
        # 使用匈牙利算法进行匹配
        if len(active_tracks) > 0 and len(detections) > 0:
            row_ind, col_ind = linear_sum_assignment(-combined_matrix)
        else:
            row_ind, col_ind = [], []
        
        # 处理匹配结果
        matched_tracks = set()
        matched_detections = set()
        current_tracks = []
        
        # 更新匹配的轨迹
        for i, j in zip(row_ind, col_ind):
            if combined_matrix[i, j] < self.iou_threshold:
                continue
                
            track = active_tracks[i]
            det = detections[j]
            emb = embeddings[j]
            
            # 更新轨迹
            track['bbox'] = det[:4]
            track['conf'] = det[4]
            track['lost'] = 0
            
            # 更新特征历史
            if track['id'] not in self.track_embeddings:
                self.track_embeddings[track['id']] = []
            self.track_embeddings[track['id']].append(emb)
            # 只保留最近5个特征
            if len(self.track_embeddings[track['id']]) > 5:
                self.track_embeddings[track['id']] = self.track_embeddings[track['id']][-5:]
            
            matched_tracks.add(i)
            matched_detections.add(j)
            current_tracks.append(track)
        
        # 处理未匹配的轨迹
        for i, track in enumerate(active_tracks):
            if i not in matched_tracks:
                track['lost'] += 1
                if track['lost'] < self.max_age:
                    current_tracks.append(track)
        
        # 处理未匹配的检测（新目标）
        for j, det in enumerate(detections):
            if j not in matched_detections:
                new_track = {
                    'id': self.next_id,
                    'bbox': det[:4],
                    'conf': det[4],
                    'lost': 0
                }
                self.next_id += 1
                current_tracks.append(new_track)
                
                # 初始化特征存储
                self.track_embeddings[new_track['id']] = [embeddings[j]]
        
        # 更新轨迹列表
        self.tracks = current_tracks
        return current_tracks
    
    @staticmethod
    def compute_iou(box1, box2):
        """计算两个边界框的IOU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # 计算交集区域
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

# 初始化跟踪器
tracker = ReIDByteTracker(max_age=30, iou_threshold=0.3, emb_threshold=0.7)

# 主循环
frame_id = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 使用YOLOv8进行检测
    results = det_model(frame, conf=0.25, iou=0.5, classes=[0,1], verbose=False)[0]
    
    # 提取检测结果
    detections = []
    if results.boxes:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_id in zip(boxes, confs, class_ids):
            detections.append([*box, conf, cls_id])
    
    # 裁剪检测区域并提取ReID特征
    crops = []
    valid_detections = []
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        if x2 > x1 and y2 > y1:  # 验证边界
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:  # 确保图像不为空
                # 调整大小以适应ReID模型
                resized_crop = cv2.resize(crop, (128, 256))
                crops.append(resized_crop)
                valid_detections.append(det)
    
    # 提取ReID特征
    embeddings = []
    if crops:
        # 转换为模型输入格式
        crops_tensor = torch.stack([
            torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float() / 255.0
            for crop in crops
        ])
        
        # 提取特征
        with torch.no_grad():
            embeddings = reid_model(crops_tensor).cpu().numpy()
    
    # 归一化特征向量
    for i in range(len(embeddings)):
        embeddings[i] /= np.linalg.norm(embeddings[i])
    
    # 更新跟踪器
    tracks = tracker.update(valid_detections, embeddings)
    
    # 在帧上绘制结果
    display_frame = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = map(int, track['bbox'])
        track_id = track['id']
        conf = track['conf']
        
        # 绘制边界框
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制ID标签
        label = f"ID:{track_id}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # 计算文本背景位置
        text_bg_x1 = x1
        text_bg_y1 = y1 - text_size[1] - 10
        text_bg_x2 = x1 + text_size[0] + 5
        text_bg_y2 = y1
        
        # 确保文本背景在图像范围内
        if text_bg_y1 < 0:
            text_bg_y1 = y1 + text_size[1] + 10
            text_bg_y2 = y1 + 2 * text_size[1] + 10
        
        # 绘制ID背景和文本
        cv2.rectangle(
            display_frame,
            (text_bg_x1, text_bg_y1),
            (text_bg_x2, text_bg_y2),
            (50, 50, 50),  # 深灰色背景
            -1
        )
        cv2.putText(
            display_frame,
            label,
            (x1 + 2, text_bg_y1 + text_size[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),  # 白色文本
            2
        )
        
        # 写入轨迹数据
        with open(track_file, "a") as f:
            f.write(f"{frame_id},{track_id},{0},{conf:.4f},{x1},{y1},{x2},{y2}\n")
    
    # 计算并显示FPS
    current_time = time.time()
    elapsed = current_time - start_time
    fps = 1 / elapsed if elapsed > 0 else 0
    
    cv2.putText(
        display_frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    start_time = current_time
    
    # 写入输出视频
    out.write(display_frame)
    
    # 进度日志
    frame_id += 1
    if frame_id % 50 == 0:
        progress = frame_id / total_frames * 100
        logger.info(f"处理中: {frame_id}/{total_frames} ({progress:.1f}%)")

# 释放资源
cap.release()
out.release()
logger.success(f"跟踪完成! 输出视频已保存至: videos/output_reid.mp4")
logger.success(f"轨迹数据已保存至: {track_file}")
