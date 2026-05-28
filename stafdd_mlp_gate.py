import argparse
import os
import json
import math
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise ImportError("This script requires PyTorch. Please install torch first.") from e

try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:
    raise ImportError("This script requires scipy. Please install scipy first.") from e


# -----------------------------
# Basic utilities
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_paths(values: List[str]) -> List[str]:
    if len(values) == 1 and "," in values[0]:
        return [x.strip() for x in values[0].split(",") if x.strip()]
    return values


def check_pred_columns(df: pd.DataFrame):
    req = {
        "frame_id", "track_id", "normalized_x", "normalized_y", "w", "h",
        "yolo_class", "yolo_conf", "lstm_class", "lstm_conf"
    }
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Prediction CSV missing columns: {sorted(miss)}")


def check_gt_columns(df: pd.DataFrame):
    req = {"frame_id", "normalized_x", "normalized_y", "w", "h", "true_class"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"GT CSV missing columns: {sorted(miss)}")


def iou_norm_center(a, b) -> float:
    """a/b = [cx, cy, w, h], normalized center format."""
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    ax1, ay1, ax2, ay2 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx1, by1, bx2, by2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def matched_pred_with_gt(pred: pd.DataFrame, gt: pd.DataFrame, iou_thr: float = 0.5) -> pd.DataFrame:
    """
    Per-frame Hungarian matching.
    Return matched prediction rows with true_class appended.
    Unmatched predictions and unmatched GTs are ignored for gate training.
    """
    check_pred_columns(pred)
    check_gt_columns(gt)

    pred = pred.copy().reset_index(drop=True)
    pred["pred_uid"] = np.arange(len(pred))

    rows = []
    common_frames = sorted(set(pred["frame_id"].unique()) & set(gt["frame_id"].unique()))
    if not common_frames:
        raise ValueError(
            "No common frame_id between predictions and GT. "
            "Check whether pred frame starts from 1 while GT starts from 0. "
            "Use --pred-frame-offset -1 if needed."
        )

    for f in common_frames:
        pf = pred[pred["frame_id"] == f].reset_index(drop=True)
        gf = gt[gt["frame_id"] == f].reset_index(drop=True)
        if len(pf) == 0 or len(gf) == 0:
            continue

        M = np.zeros((len(gf), len(pf)), dtype=float)
        for i in range(len(gf)):
            gb = [gf.loc[i, "normalized_x"], gf.loc[i, "normalized_y"], gf.loc[i, "w"], gf.loc[i, "h"]]
            for j in range(len(pf)):
                pb = [pf.loc[j, "normalized_x"], pf.loc[j, "normalized_y"], pf.loc[j, "w"], pf.loc[j, "h"]]
                M[i, j] = iou_norm_center(gb, pb)

        gi_idx, pj_idx = linear_sum_assignment(-M)
        for gi, pj in zip(gi_idx, pj_idx):
            if M[gi, pj] >= iou_thr:
                r = pf.loc[pj].to_dict()
                r["true_class"] = int(gf.loc[gi, "true_class"])
                r["match_iou"] = float(M[gi, pj])
                rows.append(r)

    if not rows:
        raise ValueError(f"No matched prediction-GT pairs at IoU >= {iou_thr}.")
    return pd.DataFrame(rows)


def class_conf_to_pos_prob(cls: pd.Series, conf: pd.Series, positive_class: int = 1, mode: str = "complement") -> np.ndarray:
    """
    Convert predicted class + confidence to probability of positive_class.
    mode='complement':
        if pred == positive: p=conf; else p=1-conf
    mode='zero':
        if pred == positive: p=conf; else p=0
    """
    cls = pd.to_numeric(cls, errors="coerce").fillna(-1).astype(int).values
    conf = pd.to_numeric(conf, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0).values
    if mode == "complement":
        return np.where(cls == int(positive_class), conf, 1.0 - conf)
    elif mode == "zero":
        return np.where(cls == int(positive_class), conf, 0.0)
    else:
        raise ValueError("--prob-mode must be 'complement' or 'zero'")


def add_gate_features(
    df: pd.DataFrame,
    tref: float = 200.0,
    positive_class: int = 1,
    prob_mode: str = "complement",
    age_mode: str = "frame",
) -> pd.DataFrame:
    """
    Add p_yolo, p_lstm, track_age, s_len and other helper columns.
    age_mode:
      - frame: frame_id - first frame_id of the same track
      - order: row order within the same track
    """
    check_pred_columns(df)
    out = df.copy()
    out["yolo_conf"] = pd.to_numeric(out["yolo_conf"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["lstm_conf"] = pd.to_numeric(out["lstm_conf"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["yolo_class"] = pd.to_numeric(out["yolo_class"], errors="coerce").fillna(-1).astype(int)
    out["lstm_class"] = pd.to_numeric(out["lstm_class"], errors="coerce").fillna(-1).astype(int)

    out["p_yolo"] = class_conf_to_pos_prob(out["yolo_class"], out["yolo_conf"], positive_class, prob_mode)
    out["p_lstm"] = class_conf_to_pos_prob(out["lstm_class"], out["lstm_conf"], positive_class, prob_mode)

    # stable sorting for track age
    sort_cols = ["track_id", "frame_id"]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    if age_mode == "frame":
        start = out.groupby("track_id")["frame_id"].transform("min")
        out["track_age"] = pd.to_numeric(out["frame_id"], errors="coerce") - pd.to_numeric(start, errors="coerce")
    elif age_mode == "order":
        out["track_age"] = out.groupby("track_id").cumcount()
    else:
        raise ValueError("--age-mode must be 'frame' or 'order'")

    out["track_age"] = out["track_age"].clip(lower=0).astype(float)
    out["s_len"] = (out["track_age"] / float(tref)).clip(0.0, 1.0)

    out["conf_gap"] = (out["p_yolo"] - out["p_lstm"]).abs()
    out["agree"] = (out["yolo_class"] == out["lstm_class"]).astype(float)
    return out


def make_feature_matrix(df: pd.DataFrame, feature_set: str = "basic") -> np.ndarray:
    """
    basic: [p_yolo, p_lstm, s_len]
    full : [p_yolo, p_lstm, s_len, yolo_conf, lstm_conf, conf_gap, agree]
    """
    if feature_set == "basic":
        cols = ["p_yolo", "p_lstm", "s_len"]
    elif feature_set == "full":
        cols = ["p_yolo", "p_lstm", "s_len", "yolo_conf", "lstm_conf", "conf_gap", "agree"]
    else:
        raise ValueError("--feature-set must be 'basic' or 'full'")
    return df[cols].astype(float).values


# -----------------------------
# Gate models
# -----------------------------
class BasicMLPGate(nn.Module):
    """
    普通 MLP gate:
        alpha = sigmoid(MLP([p_yolo, p_lstm, s]))
    不强制 alpha 随 s 单调增加，但表达能力更自由。
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MonotonicLengthAwareGate(nn.Module):
   
    def __init__(self, input_dim: int = 3, hidden_dim: int = 8, s_index: int = 2):
        super().__init__()
        self.s_index = s_index
        base_dim = input_dim - 1
        self.base_net = nn.Sequential(
            nn.Linear(base_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.beta_raw = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        s = x[:, self.s_index]
        base_x = torch.cat([x[:, :self.s_index], x[:, self.s_index+1:]], dim=1)
        base = self.base_net(base_x).squeeze(-1)
        beta = F.softplus(self.beta_raw)
        alpha = torch.sigmoid(base + beta * s + self.bias)
        return alpha


def build_model(gate_type: str, input_dim: int, hidden_dim: int, s_index: int = 2) -> nn.Module:
    if gate_type == "mlp":
        return BasicMLPGate(input_dim=input_dim, hidden_dim=hidden_dim)
    elif gate_type == "monotonic":
        return MonotonicLengthAwareGate(input_dim=input_dim, hidden_dim=hidden_dim, s_index=s_index)
    else:
        raise ValueError("--gate-type must be 'mlp' or 'monotonic'")


def fusion_prob_from_alpha(alpha: torch.Tensor, p_yolo: torch.Tensor, p_lstm: torch.Tensor) -> torch.Tensor:
    return (1.0 - alpha) * p_yolo + alpha * p_lstm


# -----------------------------
# Metrics
# -----------------------------
def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    classes = sorted(list(set(y_true.tolist()) | set(y_pred.tolist()) | {0, 1}))

    per = {}
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        tn = int(((y_true != c) & (y_pred != c)).sum())
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        per[c] = {"P": p, "R": r, "F1": f, "TP": tp, "FP": fp, "FN": fn, "TN": tn}

    macro_p = float(np.mean([per[c]["P"] for c in classes]))
    macro_r = float(np.mean([per[c]["R"] for c in classes]))
    macro_f1 = float(np.mean([per[c]["F1"] for c in classes]))
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {
        "Accuracy": acc,
        "Macro_Precision": macro_p,
        "Macro_Recall": macro_r,
        "Macro_F1": macro_f1,
        **{f"Class{c}_P": per[c]["P"] for c in classes},
        **{f"Class{c}_R": per[c]["R"] for c in classes},
        **{f"Class{c}_F1": per[c]["F1"] for c in classes},
    }


def evaluate_models_on_matched(
    matched: pd.DataFrame,
    positive_class: int = 1,
    prob_mode: str = "complement",
    threshold: float = 0.5,
) -> pd.DataFrame:
    y = matched["true_class"].astype(int).values
    rows = []

    for name, cls_col, conf_col in [
        ("YOLO", "yolo_class", "yolo_conf"),
        ("LSTM", "lstm_class", "lstm_conf"),
        ("Fusion", "final_class", "final_conf"),
    ]:
        if cls_col not in matched.columns or conf_col not in matched.columns:
            continue
        pred = matched[cls_col].astype(int).values
        m = binary_metrics(y, pred)
        m["model"] = name
        m["n"] = len(y)
        rows.append(m)

    return pd.DataFrame(rows)


# -----------------------------
# Commands
# -----------------------------
def cmd_inspect(args):
    pred_paths = parse_paths(args.preds)
    all_rows = []
    for p in pred_paths:
        df = pd.read_csv(p)
        check_pred_columns(df)
        df["file"] = os.path.basename(p)
        df = add_gate_features(df, tref=args.tref, positive_class=args.positive_class, prob_mode=args.prob_mode, age_mode=args.age_mode)
        all_rows.append(df)

        lens = df.groupby("track_id").size()
        max_age = df.groupby("track_id")["track_age"].max()
        print(f"\n=== {p} ===")
        print(f"Rows: {len(df)}, Frames: {df['frame_id'].min()}–{df['frame_id'].max()} ({df['frame_id'].nunique()} unique), Tracks: {df['track_id'].nunique()}")
        print("Track length quantiles:")
        print(lens.describe(percentiles=[.10, .25, .50, .75, .90, .95, .99]).to_string())
        print("Track age max quantiles:")
        print(max_age.describe(percentiles=[.10, .25, .50, .75, .90, .95, .99]).to_string())

    all_df = pd.concat(all_rows, ignore_index=True)
    bins = [0, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    all_df["age_bin"] = pd.cut(all_df["track_age"], bins=bins, include_lowest=True, right=False)
    all_df["agree"] = (all_df["yolo_class"] == all_df["lstm_class"]).astype(float)
    stat = all_df.groupby("age_bin", observed=False).agg(
        n=("frame_id", "size"),
        yolo_conf_mean=("yolo_conf", "mean"),
        lstm_conf_mean=("lstm_conf", "mean"),
        lstm_conf_median=("lstm_conf", "median"),
        yolo_lstm_agree=("agree", "mean"),
    ).reset_index()
    print("\n=== Overall confidence by track age bin ===")
    print(stat.to_string(index=False))

    print("\nSuggested T_ref:")
    print(f"- Your files are about 745–750 frames long, but the mean track length is around 200 frames and many tracks are short fragments.")
    print(f"- LSTM confidence and YOLO-LSTM agreement usually become stable after roughly 100–200 frames.")
    print(f"- Recommended default: T_ref = 200 frames (about 8 seconds at 25 fps).")
    print(f"- If you want a more conservative setting, try T_ref = 300 frames as a supplementary comparison.")


def load_and_match_many(pred_paths: List[str], gt_paths: List[str], args) -> pd.DataFrame:
    if len(pred_paths) != len(gt_paths):
        raise ValueError("The number of prediction CSV files must equal the number of GT CSV files.")

    matched_all = []
    for idx, (pp, gp) in enumerate(zip(pred_paths, gt_paths)):
        pred = pd.read_csv(pp)
        gt = pd.read_csv(gp)
        check_pred_columns(pred)
        check_gt_columns(gt)

        if args.pred_frame_offset != 0:
            pred["frame_id"] = pred["frame_id"].astype(int) + int(args.pred_frame_offset)
        if args.gt_frame_offset != 0:
            gt["frame_id"] = gt["frame_id"].astype(int) + int(args.gt_frame_offset)

        pred = add_gate_features(
            pred,
            tref=args.tref,
            positive_class=args.positive_class,
            prob_mode=args.prob_mode,
            age_mode=args.age_mode,
        )
        m = matched_pred_with_gt(pred, gt, iou_thr=args.iou_thr)
        m["source_file"] = os.path.basename(pp)
        m["source_index"] = idx
        matched_all.append(m)
        print(f"[MATCH] {pp} + {gp}: matched {len(m)} rows")

    out = pd.concat(matched_all, ignore_index=True)
    return out


def split_by_group(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    groups = (df["source_file"].astype(str) + "_track" + df["track_id"].astype(str)).values
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_ratio))) if len(uniq) > 1 else 0
    val_groups = set(uniq[:n_val])
    is_val = np.array([g in val_groups for g in groups])
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]
    if len(val_idx) == 0:
        val_idx = train_idx
    return train_idx, val_idx


def cmd_train(args):
    set_seed(args.seed)
    pred_paths = parse_paths(args.train_preds)
    gt_paths = parse_paths(args.train_gts)
    matched = load_and_match_many(pred_paths, gt_paths, args)

    if args.matched_out:
        matched.to_csv(args.matched_out, index=False)
        print(f"[SAVE] matched training data saved to {args.matched_out}")

    X = make_feature_matrix(matched, feature_set=args.feature_set)
    y = (matched["true_class"].astype(int).values == int(args.positive_class)).astype(np.float32)
    input_dim = X.shape[1]
    s_index = 2  # p_yolo, p_lstm, s_len; for full, s_len is still index 2

    train_idx, val_idx = split_by_group(matched, val_ratio=args.val_ratio, seed=args.seed)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(args.gate_type, input_dim=input_dim, hidden_dim=args.hidden_dim, s_index=s_index).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    xva = torch.tensor(X_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state = None
    patience_count = 0

    print(f"[TRAIN] n_train={len(X_train)}, n_val={len(X_val)}, gate_type={args.gate_type}, feature_set={args.feature_set}, device={device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=device)
        total_loss = 0.0
        for start in range(0, len(xtr), args.batch_size):
            idx = perm[start:start + args.batch_size]
            xb = xtr[idx]
            yb = ytr[idx]
            alpha = model(xb)
            p_yolo = xb[:, 0]
            p_lstm = xb[:, 1]
            pf = fusion_prob_from_alpha(alpha, p_yolo, p_lstm).clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(pf, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(idx)

        train_loss = total_loss / max(len(xtr), 1)

        model.eval()
        with torch.no_grad():
            alpha_v = model(xva)
            pf_v = fusion_prob_from_alpha(alpha_v, xva[:, 0], xva[:, 1]).clamp(1e-6, 1 - 1e-6)
            val_loss = float(F.binary_cross_entropy(pf_v, yva).item())
            pred_v = (pf_v.detach().cpu().numpy() >= args.threshold).astype(int)
            met = binary_metrics(y_val.astype(int), pred_v)

        if epoch == 1 or epoch % args.print_every == 0:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | val_acc={met['Accuracy']:.4f} | val_macroF1={met['Macro_F1']:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"[EARLY STOP] epoch={epoch}, best_val_loss={best_val:.5f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final train/val metrics
    model.eval()
    with torch.no_grad():
        for split_name, xx, yy in [("train", xtr, ytr), ("val", xva, yva)]:
            alpha = model(xx)
            pf = fusion_prob_from_alpha(alpha, xx[:, 0], xx[:, 1])
            pred_pos = (pf.detach().cpu().numpy() >= args.threshold).astype(int)
            met = binary_metrics(yy.detach().cpu().numpy().astype(int), pred_pos)
            print(f"[{split_name.upper()}] acc={met['Accuracy']:.4f}, macroP={met['Macro_Precision']:.4f}, macroR={met['Macro_Recall']:.4f}, macroF1={met['Macro_F1']:.4f}, alpha_mean={float(alpha.mean()):.4f}")

    cfg = {
        "gate_type": args.gate_type,
        "feature_set": args.feature_set,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "tref": args.tref,
        "positive_class": args.positive_class,
        "prob_mode": args.prob_mode,
        "age_mode": args.age_mode,
        "threshold": args.threshold,
        "iou_thr": args.iou_thr,
        "pred_frame_offset": args.pred_frame_offset,
        "gt_frame_offset": args.gt_frame_offset,
    }
    save_obj = {"state_dict": model.state_dict(), "config": cfg}
    torch.save(save_obj, args.model_out)
    print(f"[SAVE] model saved to {args.model_out}")


def load_gate_model(model_path: str, device: torch.device):
    obj = torch.load(model_path, map_location=device)
    cfg = obj["config"]
    model = build_model(
        cfg["gate_type"],
        input_dim=int(cfg["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        s_index=2,
    ).to(device)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, cfg


def apply_gate_to_df(df: pd.DataFrame, model, cfg: Dict, threshold: Optional[float] = None, device: Optional[torch.device] = None) -> pd.DataFrame:
    if device is None:
        device = next(model.parameters()).device
    threshold = float(cfg.get("threshold", 0.5) if threshold is None else threshold)
    positive_class = int(cfg.get("positive_class", 1))
    negative_class = 1 - positive_class

    feat_df = add_gate_features(
        df,
        tref=float(cfg["tref"]),
        positive_class=positive_class,
        prob_mode=cfg.get("prob_mode", "complement"),
        age_mode=cfg.get("age_mode", "frame"),
    )
    X = make_feature_matrix(feat_df, feature_set=cfg.get("feature_set", "basic"))
    x = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        alpha = model(x)
        pf = fusion_prob_from_alpha(alpha, x[:, 0], x[:, 1]).detach().cpu().numpy()
        alpha_np = alpha.detach().cpu().numpy()

    out = feat_df.copy()
    out["gate_alpha_lstm"] = alpha_np
    out["gate_weight_yolo"] = 1.0 - alpha_np
    out["p_fused_positive"] = pf
    out["final_class"] = np.where(pf >= threshold, positive_class, negative_class).astype(int)
    out["final_conf"] = np.where(out["final_class"].values == positive_class, pf, 1.0 - pf).astype(float)

    # Restore original row order if possible: sorting happened inside add_gate_features
    # We keep sorted by track/frame because that is usually convenient.
    return out


def cmd_predict(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_gate_model(args.model, device)

    df = pd.read_csv(args.input)
    if args.pred_frame_offset != 0:
        # Usually do NOT offset for output prediction unless you need aligned frame_id in the saved file.
        df["frame_id"] = df["frame_id"].astype(int) + int(args.pred_frame_offset)

    out = apply_gate_to_df(df, model, cfg, threshold=args.threshold, device=device)
    out.to_csv(args.output, index=False)
    print(f"[SAVE] gate fusion result saved to {args.output}")
    print(f"[INFO] alpha mean={out['gate_alpha_lstm'].mean():.4f}, median={out['gate_alpha_lstm'].median():.4f}, min={out['gate_alpha_lstm'].min():.4f}, max={out['gate_alpha_lstm'].max():.4f}")
    print(f"[INFO] final class counts: {out['final_class'].value_counts().to_dict()}")


def cmd_evaluate(args):
    pred = pd.read_csv(args.preds)
    gt = pd.read_csv(args.gt)
    if args.pred_frame_offset != 0:
        pred["frame_id"] = pred["frame_id"].astype(int) + int(args.pred_frame_offset)
    if args.gt_frame_offset != 0:
        gt["frame_id"] = gt["frame_id"].astype(int) + int(args.gt_frame_offset)

    # ensure feature columns don't hurt, just match
    matched = matched_pred_with_gt(pred, gt, iou_thr=args.iou_thr)
    summary = evaluate_models_on_matched(
        matched,
        positive_class=args.positive_class,
        prob_mode=args.prob_mode,
        threshold=args.threshold,
    )
    print(summary[["model", "n", "Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1"]].to_string(index=False))
    if args.summary_out:
        summary.to_csv(args.summary_out, index=False)
        print(f"[SAVE] summary saved to {args.summary_out}")
    if args.matched_out:
        matched.to_csv(args.matched_out, index=False)
        print(f"[SAVE] matched eval rows saved to {args.matched_out}")


def cmd_run(args):
    # Train
    train_args = args
    cmd_train(train_args)
    # Predict
    pred_args = argparse.Namespace(
        input=args.test_preds,
        model=args.model_out,
        output=args.output,
        threshold=args.threshold,
        pred_frame_offset=0,  # keep original frames in saved output
        device=args.device,
    )
    cmd_predict(pred_args)
    # Evaluate
    eval_args = argparse.Namespace(
        preds=args.output,
        gt=args.test_gt,
        iou_thr=args.iou_thr,
        pred_frame_offset=args.pred_frame_offset,
        gt_frame_offset=args.gt_frame_offset,
        positive_class=args.positive_class,
        prob_mode=args.prob_mode,
        threshold=args.threshold,
        summary_out=args.summary_out,
        matched_out=args.eval_matched_out,
    )
    cmd_evaluate(eval_args)


# -----------------------------
# CLI
# -----------------------------
def add_common_train_args(p):
    p.add_argument("--tref", type=float, default=200.0, help="Reference trajectory length for s_len = min(track_age / T_ref, 1). Default: 200.")
    p.add_argument("--positive-class", type=int, default=1, help="Positive class, usually diseased=1.")
    p.add_argument("--prob-mode", choices=["complement", "zero"], default="complement", help="How to convert class+conf to positive probability.")
    p.add_argument("--age-mode", choices=["frame", "order"], default="frame", help="Track age definition.")
    p.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for pred-GT matching.")
    p.add_argument("--pred-frame-offset", type=int, default=0, help="Add this offset to prediction frame_id before matching. Use -1 if pred starts at 1 and GT starts at 0.")
    p.add_argument("--gt-frame-offset", type=int, default=0, help="Add this offset to GT frame_id before matching.")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for positive probability.")


def main():
    ap = argparse.ArgumentParser(description="STAFDD learnable MLP gate fusion")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="Inspect prediction CSV and suggest T_ref")
    p.add_argument("--preds", nargs="+", required=True, help="Prediction CSV files")
    p.add_argument("--tref", type=float, default=200.0)
    p.add_argument("--positive-class", type=int, default=1)
    p.add_argument("--prob-mode", choices=["complement", "zero"], default="complement")
    p.add_argument("--age-mode", choices=["frame", "order"], default="frame")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("train", help="Train learnable gate")
    p.add_argument("--train-preds", nargs="+", required=True, help="Training prediction CSV files")
    p.add_argument("--train-gts", nargs="+", required=True, help="Training GT CSV files")
    p.add_argument("--model-out", required=True, help="Output .pt model")
    p.add_argument("--matched-out", default=None, help="Optional matched training rows CSV")
    p.add_argument("--gate-type", choices=["mlp", "monotonic"], default="monotonic")
    p.add_argument("--feature-set", choices=["basic", "full"], default="basic")
    p.add_argument("--hidden-dim", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--device", default=None)
    add_common_train_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="Apply trained gate to prediction CSV")
    p.add_argument("--input", required=True, help="Input prediction CSV")
    p.add_argument("--model", required=True, help="Trained .pt model")
    p.add_argument("--output", required=True, help="Output gate-fused CSV")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--pred-frame-offset", type=int, default=0, help="Usually keep 0. Only use if you want shifted frame_id in output.")
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("evaluate", help="Evaluate YOLO/LSTM/Fusion on matched rows")
    p.add_argument("--preds", required=True, help="Prediction CSV with final_class/final_conf")
    p.add_argument("--gt", required=True, help="GT CSV")
    p.add_argument("--summary-out", default=None)
    p.add_argument("--matched-out", default=None)
    add_common_train_args(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("run", help="Train + predict + evaluate")
    p.add_argument("--train-preds", nargs="+", required=True)
    p.add_argument("--train-gts", nargs="+", required=True)
    p.add_argument("--test-preds", required=True)
    p.add_argument("--test-gt", required=True)
    p.add_argument("--model-out", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-out", default="gate_eval_summary.csv")
    p.add_argument("--eval-matched-out", default=None)
    p.add_argument("--matched-out", default=None)
    p.add_argument("--gate-type", choices=["mlp", "monotonic"], default="monotonic")
    p.add_argument("--feature-set", choices=["basic", "full"], default="basic")
    p.add_argument("--hidden-dim", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--device", default=None)
    add_common_train_args(p)
    p.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
