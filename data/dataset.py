"""
Sliding-window dataset for GALAR GI endoscopy videos (pre-extracted DINOv2 features).

Pipeline:
  1. Run extract_features.py to save per-video features as .npy files
  2. This Dataset loads .npy + CSV files and yields fixed-size windows

Expected file structure:
  features_dir/{video_id}_features.npy  : [N, feat_dim] float32
  features_dir/{video_id}_frames.npy    : [N] int64  (actual frame numbers)
  labels_dir/{video_id}.csv             : columns include 'frame' and label names
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


ANATOMY_LABELS = [
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus", "ileocecal valve"
]
PATHOLOGY_LABELS = [
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer"
]
ALL_LABELS = ANATOMY_LABELS + PATHOLOGY_LABELS  # 17 classes, fixed order


class GalarWindowDataset(Dataset):
    """
    Sliding-window Dataset.
    Each item is (features, labels, frame_nums) for a contiguous window_size-frame segment.
    The last window is zero-padded if shorter than window_size.
    """

    def __init__(
        self,
        video_ids: list,
        features_dir: str,
        labels_dir: str,
        window_size: int = 512,
        stride: int = 256,
        preloaded: dict = None,
    ):
        self.features_dir = features_dir
        self.labels_dir   = labels_dir
        self.window_size  = window_size
        self.windows     = []
        self.feat_cache  = {}
        self.frame_cache = {}
        self.label_cache = {}

        for vid_id in video_ids:
            if preloaded is not None:
                if vid_id not in preloaded:
                    print(f"[SKIP] {vid_id}: not found in preloaded cache")
                    continue
                feats, frame_arr, label_arr = preloaded[vid_id]
                n = len(feats)
            else:
                feat_path  = os.path.join(features_dir, f"{vid_id}_features.npy")
                frame_path = os.path.join(features_dir, f"{vid_id}_frames.npy")
                label_path = os.path.join(labels_dir, f"{vid_id}.csv")
                if not all(os.path.exists(p) for p in [feat_path, frame_path, label_path]):
                    print(f"[SKIP] {vid_id}: missing feature file or label CSV")
                    continue
                feats     = np.load(feat_path, mmap_mode='r').astype(np.float32)
                frame_arr = np.load(frame_path).astype(np.int64)
                df        = pd.read_csv(label_path).sort_values("frame").reset_index(drop=True)
                label_arr = np.zeros((len(df), 17), dtype=np.float32)
                for i, col in enumerate(ALL_LABELS):
                    if col in df.columns:
                        label_arr[:, i] = df[col].values.astype(np.float32)
                n = min(len(feats), len(df))
                feats, frame_arr, label_arr = feats[:n], frame_arr[:n], label_arr[:n]

            self.feat_cache[vid_id]  = feats
            self.frame_cache[vid_id] = frame_arr
            self.label_cache[vid_id] = label_arr

            for start in range(0, max(1, n - window_size + 1), stride):
                self.windows.append((vid_id, start, min(start + window_size, n)))
            if n > window_size and (n - window_size) % stride != 0:
                self.windows.append((vid_id, n - window_size, n))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        vid_id, start, end = self.windows[idx]

        feats     = np.array(self.feat_cache[vid_id][start:end],  dtype=np.float32)
        frame_arr = np.array(self.frame_cache[vid_id][start:end], dtype=np.int64)
        labels    = np.array(self.label_cache[vid_id][start:end],  dtype=np.float32)
        T_actual  = len(feats)

        # zero-pad the last window if shorter than window_size
        if T_actual < self.window_size:
            pad = self.window_size - T_actual
            feats     = np.vstack([feats,     np.zeros((pad, feats.shape[1]), dtype=np.float32)])
            frame_arr = np.concatenate([frame_arr, np.full(pad, -1, dtype=np.int64)])
            labels    = np.vstack([labels,    np.zeros((pad, 17),             dtype=np.float32)])

        total_rows = len(self.feat_cache[vid_id])
        video_pos_ratio = float(start) / max(total_rows - 1, 1)  # 0.0~1.0

        return {
            "features":        torch.from_numpy(feats),      # [T, D]
            "labels":          torch.from_numpy(labels),     # [T, 17]
            "frame_nums":      torch.from_numpy(frame_arr),  # [T]
            "valid_len":       T_actual,
            "video_id":        vid_id,
            "video_pos_ratio": torch.tensor(video_pos_ratio, dtype=torch.float32),
        }


def compute_pos_weights(dataset: GalarWindowDataset) -> torch.Tensor:
    """
    Compute inverse positive-frequency weights per class over the entire training set.
    Used as pos_weight in BCEWithLogitsLoss.
    """
    all_labels = np.vstack([
        dataset.label_cache[vid]
        for vid in dataset.label_cache
    ])  # [total_frames, 17]

    pos = all_labels.mean(axis=0).clip(1e-4, 1 - 1e-4)  # [17]
    weights = (1 - pos) / pos
    return torch.from_numpy(weights.astype(np.float32))


def make_weighted_sampler(dataset: GalarWindowDataset) -> WeightedRandomSampler:
    """
    WeightedRandomSampler that oversamples windows containing rare pathology classes.
    Window weight is proportional to the number of positive pathology frames inside.
    """
    pathology_idx = list(range(8, 17))  # pathology class indices in ALL_LABELS

    weights = []
    for vid_id, start, end in dataset.windows:
        window_labels = dataset.label_cache[vid_id][start:end]  # [T, 17]
        rare_count = window_labels[:, pathology_idx].sum()
        weights.append(1.0 + rare_count * 2.0)

    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return WeightedRandomSampler(weights_tensor, num_samples=len(weights), replacement=True)


def stratified_video_split(
    video_ids: list,
    labels_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> "tuple[list, list]":
    """
    Video-level train/val split that ensures rare pathology classes appear in both splits.

    For each video, compute the set of pathology classes it contains.
    Assign videos to val first if they cover pathology classes not yet in val,
    then fill the remaining val quota, then assign the rest to train.
    """
    np.random.seed(seed)

    # aggregate pathology classes present per video
    video_pathology = {}
    for vid_id in video_ids:
        label_path = os.path.join(labels_dir, f"{vid_id}.csv")
        if not os.path.exists(label_path):
            video_pathology[vid_id] = set()
            continue
        df = pd.read_csv(label_path)
        present = set()
        for col in PATHOLOGY_LABELS:
            if col in df.columns and df[col].sum() > 0:
                present.add(col)
        video_pathology[vid_id] = present

    val_set, train_set = [], []
    covered_in_val = set()

    # sort by number of pathology classes (fewest first) to prioritize rare coverage
    sorted_vids = sorted(video_ids, key=lambda v: len(video_pathology[v]))
    np.random.shuffle(sorted_vids)

    n_val = max(1, int(len(video_ids) * val_ratio))

    for vid in sorted_vids:
        path_set = video_pathology[vid]
        # prioritize videos with pathology classes not yet covered in val
        if len(val_set) < n_val and not path_set.issubset(covered_in_val):
            val_set.append(vid)
            covered_in_val |= path_set
        elif len(val_set) < n_val:
            val_set.append(vid)
        else:
            train_set.append(vid)

    # assign any remaining videos to train
    for vid in sorted_vids:
        if vid not in val_set and vid not in train_set:
            train_set.append(vid)

    return train_set, val_set


def preload_features(video_ids: list, features_dir: str, labels_dir: str) -> dict:
    """Load all video features into memory once via mmap. Used as a shared cache during training."""
    cache = {}
    for vid_id in video_ids:
        feat_path  = os.path.join(features_dir, f"{vid_id}_features.npy")
        frame_path = os.path.join(features_dir, f"{vid_id}_frames.npy")
        label_path = os.path.join(labels_dir,   f"{vid_id}.csv")
        if not all(os.path.exists(p) for p in [feat_path, frame_path, label_path]):
            continue
        feats     = np.load(feat_path, mmap_mode='r')
        frame_arr = np.load(frame_path).astype(np.int64)
        df        = pd.read_csv(label_path).sort_values("frame").reset_index(drop=True)
        label_arr = np.zeros((len(df), 17), dtype=np.float32)
        for i, col in enumerate(ALL_LABELS):
            if col in df.columns:
                label_arr[:, i] = df[col].values.astype(np.float32)
        n = min(len(feats), len(df))
        cache[vid_id] = (feats[:n], frame_arr[:n], label_arr[:n])
    return cache
