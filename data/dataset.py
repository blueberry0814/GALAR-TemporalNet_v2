"""
Galar 데이터셋 — 슬라이딩 윈도우 기반 (DINOv2 사전 추출 특징 사용)

파이프라인:
  1. extract_features.py 로 각 비디오의 특징을 .npy로 저장
  2. 이 Dataset이 .npy + CSV를 로드해 window 단위로 반환

파일 구조 가정:
  features_dir/{video_id}_features.npy  : [N, feat_dim] float32
  features_dir/{video_id}_frames.npy    : [N] int64  (실제 프레임 번호)
  labels_dir/{video_id}.csv             : index, frame, label 컬럼 포함
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
ALL_LABELS = ANATOMY_LABELS + PATHOLOGY_LABELS  # 17개 고정 순서


class GalarWindowDataset(Dataset):
    """
    슬라이딩 윈도우 Dataset.
    각 아이템 = 연속된 window_size 프레임의 (특징, 라벨, 프레임번호).
    마지막 윈도우는 패딩으로 채움.
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
                    print(f"[SKIP] {vid_id}: preloaded 캐시 없음")
                    continue
                feats, frame_arr, label_arr = preloaded[vid_id]
                n = len(feats)
            else:
                feat_path  = os.path.join(features_dir, f"{vid_id}_features.npy")
                frame_path = os.path.join(features_dir, f"{vid_id}_frames.npy")
                label_path = os.path.join(labels_dir, f"{vid_id}.csv")
                if not all(os.path.exists(p) for p in [feat_path, frame_path, label_path]):
                    print(f"[SKIP] {vid_id}: 특징 파일 또는 라벨 없음")
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

        # 패딩 (윈도우 끝 부분)
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
    전체 학습 데이터에서 각 클래스별 양성 비율의 역수 계산.
    BCE pos_weight 인자로 사용.
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
    희귀 병변 클래스를 포함한 윈도우를 더 자주 샘플링하는 WeightedRandomSampler.
    병변 양성 프레임 수 비례로 윈도우 가중치를 계산.
    """
    # 병변 클래스 인덱스 (ALL_LABELS 기준 8~16번)
    pathology_idx = list(range(8, 17))

    weights = []
    for vid_id, start, end in dataset.windows:
        window_labels = dataset.label_cache[vid_id][start:end]  # [T, 17]
        # 희귀 병변 양성 프레임 수 → 가중치
        rare_count = window_labels[:, pathology_idx].sum()
        weights.append(1.0 + rare_count * 2.0)  # 5.0 → 2.0: 과도한 쏠림으로 loss 폭발 방지

    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return WeightedRandomSampler(weights_tensor, num_samples=len(weights), replacement=True)


def stratified_video_split(
    video_ids: list,
    labels_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> "tuple[list, list]":
    """
    희귀 병변 클래스가 train/val 양쪽에 고루 포함되도록 비디오 단위 split.

    각 비디오의 '보유 병변 집합'을 계산하고,
    희귀 클래스 분포가 균형 잡히도록 정렬 후 분할.
    """
    np.random.seed(seed)

    # 비디오별 병변 보유 여부 집계
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

    # 드문 병변을 가진 비디오를 먼저 val에 배분 (최소 1개 보장)
    val_set, train_set = [], []
    covered_in_val = set()

    # 희귀 순으로 정렬 (병변 종류 적은 것 먼저)
    sorted_vids = sorted(video_ids, key=lambda v: len(video_pathology[v]))
    np.random.shuffle(sorted_vids)

    n_val = max(1, int(len(video_ids) * val_ratio))

    for vid in sorted_vids:
        path_set = video_pathology[vid]
        # val에 아직 없는 병변을 가지면 val에 우선 배정
        if len(val_set) < n_val and not path_set.issubset(covered_in_val):
            val_set.append(vid)
            covered_in_val |= path_set
        elif len(val_set) < n_val:
            val_set.append(vid)
        else:
            train_set.append(vid)

    # 나머지 비디오는 train
    for vid in sorted_vids:
        if vid not in val_set and vid not in train_set:
            train_set.append(vid)

    return train_set, val_set


def preload_features(video_ids: list, features_dir: str, labels_dir: str) -> dict:
    """모든 비디오 피처를 mmap으로 한 번만 로드. Optuna 전역 캐시용."""
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
