# FishTrackDataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.preprocessing import StandardScaler

class FishTrackDataset(Dataset):
    def __init__(self, track_dict, label_dict, scaler=None):
        """
        track_dict: {int_id: list_of_feature_vectors}
        label_dict: {int_id: label}
        scaler: StandardScaler (fit on train, reused for val)
        """
        self.track_list = []
        self.label_list = []
        self.lengths = []
        self.keys = list(track_dict.keys())
        self.scaler = scaler

        # 收集所有点以便拟合 scaler
        all_points = []
        for k in track_dict:
            all_points.extend(track_dict[k])
        all_points = np.asarray(all_points, dtype=np.float32)

        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(all_points)

        # 对每条轨迹做标准化
        for k in track_dict:
            arr = np.asarray(track_dict[k], dtype=np.float32)
            arr_scaled = self.scaler.transform(arr)
            self.track_list.append(arr_scaled)
            self.label_list.append(label_dict[k])
            self.lengths.append(len(arr_scaled))

    def __len__(self):
        return len(self.track_list)

    def __getitem__(self, index):
        seq = torch.tensor(self.track_list[index], dtype=torch.float32)
        label = torch.tensor(self.label_list[index], dtype=torch.long)
        length = self.lengths[index]
        return seq, length, label

    def get_scaler(self):
        return self.scaler


def collate_fn(batch):
    """
    batch: list of (seq_tensor, length, label)
    """
    tracks, lengths, labels = zip(*batch)
    padded = pad_sequence(tracks, batch_first=True, padding_value=0.0)  # (B, max_len, feat)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


def get_dataset(data_path, batch_size=32, fps=25.0, num_workers=4):
    from dataset_from_csv import get_fish_track

    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')

    train_data, train_labels = get_fish_track(train_dir, fps=fps)
    val_data, val_labels = get_fish_track(val_dir, fps=fps)

    # 创建训练集，拟合 scaler
    train_set = FishTrackDataset(train_data, train_labels, scaler=None)
    train_scaler = train_set.get_scaler()

    # 验证集使用相同 scaler
    val_set = FishTrackDataset(val_data, val_labels, scaler=train_scaler)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader
