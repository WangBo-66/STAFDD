import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO
from loguru import logger
from torchreid.utils import FeatureExtractor
from scipy.optimize import linear_sum_assignment
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='ReID ByteTrack: track objects with features and output video/csv')
    parser.add_argument('--video',  default="./try.mp4", help='Path to input video file')
    parser.add_argument('--det_model', default='./best.pt',
                        help='Path to YOLOv8 detection model (.pt)')
    parser.add_argument('--reid_model', default='./Reid.pt',
                        help='Path to ReID model (.pt)')
    parser.add_argument('--output', default='outputt', help='Directory to save outputs')
    parser.add_argument('--conf', type=float, default=0.4, help='Detection confidence threshold (for YOLO)')
    parser.add_argument('--iou', type=float, default=0.3, help='Detection NMS IOU threshold')
    parser.add_argument('--max_age', type=int, default=30, help='Tracker max age')
    parser.add_argument('--iou_thresh', type=float, default=0.3, help='Matching threshold on combined score')
    parser.add_argument('--emb_thresh', type=float, default=0.7, help='Tracker embedding threshold (not used directly)')
    parser.add_argument('--theta_high', type=float, default=0.6,
                        help='High confidence threshold (theta_high) for first-stage matching')
    parser.add_argument('--theta_low', type=float, default=0.4,
                        help='Low confidence threshold (theta_low) for second-stage matching / new track creation')
    return parser.parse_args()


class ReIDByteTracker:
    def __init__(self, max_age=30, match_threshold=0.3, emb_threshold=0.7, theta_high=0.6, theta_low=0.1):
        self.max_age = max_age
        self.match_threshold = match_threshold
        self.emb_threshold = emb_threshold
        self.theta_high = theta_high
        self.theta_low = theta_low
        self.tracks = []
        self.next_id = 1
        self.track_embeddings = {}

    def update(self, detections, embeddings):
        active = [t for t in self.tracks if t['lost'] < self.max_age]
        if not detections:
            for t in active:
                t['lost'] += 1
            self.tracks = [t for t in active if t['lost'] < self.max_age]
            return self.tracks

        
        high_idx = [i for i, d in enumerate(detections) if d[4] >= self.theta_high]
        
        low_idx = [i for i, d in enumerate(detections) if self.theta_low <= d[4] < self.theta_high]

        def compute_combined_matrix(track_list, det_indices):
            if not track_list or not det_indices:
                return np.zeros((len(track_list), len(det_indices)))
            iou_m = np.zeros((len(track_list), len(det_indices)))
            emb_m = np.zeros_like(iou_m)
            for i, t in enumerate(track_list):
                for j_idx, det_j in enumerate(det_indices):
                    iou_m[i, j_idx] = self.compute_iou(t['bbox'], detections[det_j][:4])
                    if t['id'] in self.track_embeddings:
                        tr_emb = np.mean(self.track_embeddings[t['id']], axis=0)
                        emb = embeddings[det_j]
                        denom = np.linalg.norm(tr_emb) * np.linalg.norm(emb)
                        emb_m[i, j_idx] = np.dot(tr_emb, emb) / (denom if denom > 0 else 1.0)
            return 0.7 * iou_m + 0.3 * emb_m

        matched_a, matched_d = set(), set()

    
        if active and high_idx:
            combined_high = compute_combined_matrix(active, high_idx)
            row, col = linear_sum_assignment(-combined_high)
            for r, c0 in zip(row, col):
                det_j = high_idx[c0]
                if combined_high[r, c0] < self.match_threshold:
                    continue
                t, det, emb = active[r], detections[det_j], embeddings[det_j]
                t.update({'bbox': det[:4], 'conf': det[4], 'class_id': int(det[5]), 'lost': 0})
                self.track_embeddings.setdefault(t['id'], []).append(emb)
                self.track_embeddings[t['id']] = self.track_embeddings[t['id']][-5:]
                matched_a.add(r)
                matched_d.add(det_j)

   
        unmatched_active = [i for i in range(len(active)) if i not in matched_a]
        low_idx_unmatched = [i for i in low_idx if i not in matched_d]

        if unmatched_active and low_idx_unmatched:
            combined_low = compute_combined_matrix([active[i] for i in unmatched_active], low_idx_unmatched)
            row2, col2 = linear_sum_assignment(-combined_low)
            for rr, cc in zip(row2, col2):
                act_idx, det_j = unmatched_active[rr], low_idx_unmatched[cc]
                if combined_low[rr, cc] < self.match_threshold:
                    continue
                t, det, emb = active[act_idx], detections[det_j], embeddings[det_j]
                t.update({'bbox': det[:4], 'conf': det[4], 'class_id': int(det[5]), 'lost': 0})
                self.track_embeddings.setdefault(t['id'], []).append(emb)
                self.track_embeddings[t['id']] = self.track_embeddings[t['id']][-5:]
                matched_a.add(act_idx)
                matched_d.add(det_j)

  
        new_tracks = []
        for i, t in enumerate(active):
            if i in matched_a:
                new_tracks.append(t)
            else:
                t['lost'] += 1
                if t['lost'] < self.max_age:
                    new_tracks.append(t)

 
        for j, det in enumerate(detections):
            if j in matched_d:
                continue
            if det[4] >= self.theta_low:
                new = {'id': self.next_id, 'bbox': det[:4], 'conf': det[4],
                       'class_id': int(det[5]), 'lost': 0}
                self.track_embeddings[self.next_id] = [embeddings[j]]
                self.next_id += 1
                new_tracks.append(new)

        self.tracks = new_tracks
        return new_tracks

    @staticmethod
    def compute_iou(b1, b2):
        x1, y1, x2, y2 = b1
        x1b, y1b, x2b, y2b = b2
        ix1, iy1 = max(x1, x1b), max(y1, y1b)
        ix2, iy2 = min(x2, x2b), min(y2, y2b)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (x2 - x1) * (y2 - y1) + (x2b - x1b) * (y2b - y1b) - inter
        return inter / union if union > 0 else 0


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]
    track_csv = os.path.join(args.output, f"{base}_tracks.csv")
    out_video = os.path.join(args.output, f"{base}_reid.mp4")

    det_model = YOLO(args.det_model)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    reid_model = FeatureExtractor(model_name='osnet_x0_25',
                                  model_path=args.reid_model, device=device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {args.video}")
        return
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    with open(track_csv, 'w') as f:
        f.write("frame_id,track_id,class_id,conf,x1,y1,x2,y2,w,h,center_x,center_y,normalized_x,normalized_y\n")

    logger.info(f"轨迹数据保存到: {track_csv}")

    tracker = ReIDByteTracker(max_age=args.max_age,
                              match_threshold=args.iou_thresh,
                              emb_threshold=args.emb_thresh,
                              theta_high=args.theta_high,
                              theta_low=args.theta_low)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        res = det_model(frame, conf=args.conf, iou=args.iou, classes=[0, 1], verbose=False)[0]
        dets = []
        if res.boxes:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            for b, c, ci in zip(boxes, confs, cls):
                dets.append([*b, float(c), int(ci)])

        crops, valid = [], []
        for d in dets:
            x1, y1, x2, y2 = map(int, d[:4])
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(cv2.resize(crop, (128, 256)))
                    valid.append([x1, y1, x2, y2, d[4], d[5]])

        embs = []
        if crops:
            ts = torch.stack([
                torch.from_numpy(cv2.cvtColor(c, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float() / 255.
                for c in crops]).to(device)
            with torch.no_grad():
                embs = reid_model(ts).cpu().numpy()
            embs = [e / np.linalg.norm(e) for e in embs]

        tracks = tracker.update(valid, embs)

        disp = frame.copy()
        for t in tracks:
            if t['lost'] > 0:
                continue
            x1, y1, x2, y2 = map(int, t['bbox'])
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2, y1 + h / 2
            nx, ny = cx / width, cy / height
            nw, nh = w / width, h / height
            tid, cid, conf = t['id'], t['class_id'], t['conf']

      
            col = (0, 255, 0) if cid == 0 else (0, 0, 255)
            cv2.rectangle(disp, (x1, y1), (x2, y2), col, 1)

            name = f"{'normal' if cid == 0 else 'sick'}:{tid}"
            tsz, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty1 = y1 - tsz[1] - 5 if y1 - tsz[1] - 5 > 0 else y1 + tsz[1] + 5
            cv2.rectangle(disp, (x1, ty1 - tsz[1] - 2), (x1 + tsz[0] + 2, ty1 + 2), (50, 50, 50), -1)
            cv2.putText(disp, name, (x1 + 1, ty1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            with open(track_csv, 'a') as f:
                f.write(
                    f"{frame_id},{tid},{cid},{conf:.4f},{x1},{y1},{x2},{y2},{nw:.4f},{nh:.4f},{cx:.2f},{cy:.2f},{nx:.4f},{ny:.4f}\n"
                )

        out.write(disp)

    cap.release()
    out.release()
    logger.success(f"完成! 视频: {out_video}")
    logger.success(f"CSV: {track_csv}")


if __name__ == '__main__':
    main()
