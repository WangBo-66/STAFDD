# dataset_from_csv.py
import pandas as pd
import os
import numpy as np

def get_fish_track(data_path, fps=25.0):
    paths = os.listdir(data_path)
    track_dict = {}
    label_dict = {}
    file_and_order_dict = {}
    count = 0

    for data_file in paths:
        data_file = os.path.join(data_path, data_file)
        df = pd.read_csv(data_file)

        # 图像宽高（假设相同视频中不变）
        img_w = df['x2'][0] - df['x1'][0] + df['w'][0]
        img_h = df['y2'][0] - df['y1'][0] + df['h'][0]

        for index, track_id in enumerate(df['track_id']):
            file_and_order = data_file + "_" + str(track_id)

            if file_and_order not in file_and_order_dict:
                file_and_order_dict[file_and_order] = count
                count += 1
                track_dict[file_and_order_dict[file_and_order]] = []
                label_dict[file_and_order_dict[file_and_order]] = int(df['class_id'][index])

        # 按 track_id 分组
        grouped = df.groupby('track_id')
        for tid, group in grouped:
            group = group.sort_values(by='frame_id')  # 按帧排序
            xs = group['normalized_x'].values
            ys = group['normalized_y'].values
            cx = group['center_x'].values
            cy = group['center_y'].values
            frame_ids = group['frame_id'].values

            # 计算像素速度并归一化到图像尺度，然后得到 v, cosθ, sinθ
            # 初始化数组
            L = len(xs)
            vx_pixels = np.zeros(L, dtype=np.float32)
            vy_pixels = np.zeros(L, dtype=np.float32)
            vx_norm = np.zeros(L, dtype=np.float32)
            vy_norm = np.zeros(L, dtype=np.float32)

            for i in range(1, L):
                dt = (frame_ids[i] - frame_ids[i - 1]) / fps  # 实际时间间隔（秒）
                if dt <= 0:
                    dt = 1.0 / fps  # 容错：避免 0
                dx = cx[i] - cx[i - 1]
                dy = cy[i] - cy[i - 1]
                vx_pixels[i] = dx / dt
                vy_pixels[i] = dy / dt
                # 归一化到图像尺度
                vx_norm[i] = vx_pixels[i] / img_w
                vy_norm[i] = vy_pixels[i] / img_h

            # 构建最终每帧特征 [normalized_x, normalized_y, v, cosθ, sinθ]
            seq = []
            for i in range(L):
                v_norm = float(np.sqrt(vx_norm[i] * vx_norm[i] + vy_norm[i] * vy_norm[i]))
                v_pix_mag = float(np.sqrt(vx_pixels[i] * vx_pixels[i] + vy_pixels[i] * vy_pixels[i]))
                if v_pix_mag > 0:
                    cos_t = float(vx_pixels[i] / v_pix_mag)
                    sin_t = float(vy_pixels[i] / v_pix_mag)
                else:
                    cos_t = 0.0
                    sin_t = 0.0
                seq.append([float(xs[i]), float(ys[i]), v_norm, cos_t, sin_t])

            file_and_order = data_file + "_" + str(tid)
            track_dict[file_and_order_dict[file_and_order]] = seq

    return track_dict, label_dict
