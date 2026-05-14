"""
Post-processing utilities

1. anatomy_pathology_gating : Suppress anatomically implausible pathology predictions
   - Soft gate computed from data-driven conditional probability
   - Multiplication instead of hard-zero (preserves rare co-occurrences)

2. gt_style_event_composition : GT-aligned segment generation
   - New segment starts whenever the active label set changes
   - Tracks the full label set rather than per-class hysteresis

3. per_class_threshold_optimizer : Per-class threshold search
   - Optimizes F1 or balanced accuracy per class
   - Falls back to lower threshold for rare classes

Usage:
  from utils.postprocess import (
      build_anatomy_pathology_gate,
      apply_anatomy_gating,
      gt_style_event_composition,
      optimize_per_class_thresholds,
  )
"""

import os
import csv
import numpy as np
from typing import Optional


ANATOMY_LABELS = [
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus", "ileocecal valve",
]
PATHOLOGY_LABELS = [
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer",
]
_TRANSITION_MARKERS = {"z-line", "pylorus", "ileocecal valve"}


# ── 1. Anatomy-Pathology Gating ───────────────────────────────────────────────

def build_anatomy_pathology_gate(
    label_dir: str,
    soft_threshold: float = 0.005,
    hard_threshold: float = 0.001,
    min_gate: float = 0.05,
) -> np.ndarray:
    """
    Compute anatomy × pathology soft gate matrix from training data.

    gate[a, p] = soft gate value in [min_gate, 1.0]
      - P(path p | anatomy a) > soft_threshold → 1.0 (fully allowed)
      - P < hard_threshold                     → min_gate (suppressed, not zeroed)
      - in between                             → linear interpolation

    Returns
    -------
    gate : np.ndarray [8, 9] float, values in [min_gate, 1.0]
    """
    comat    = np.zeros((len(ANATOMY_LABELS), len(PATHOLOGY_LABELS)), dtype=np.float64)
    anat_cnt = np.zeros(len(ANATOMY_LABELS), dtype=np.float64)

    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".csv"):
            continue
        with open(os.path.join(label_dir, fname)) as f:
            for row in csv.DictReader(f):
                av = [row.get(a, "0").strip() == "1" for a in ANATOMY_LABELS]
                pv = [row.get(p, "0").strip() == "1" for p in PATHOLOGY_LABELS]
                for ai, a_active in enumerate(av):
                    if a_active:
                        anat_cnt[ai] += 1
                        for pi, p_active in enumerate(pv):
                            if p_active:
                                comat[ai, pi] += 1

    cond = np.zeros_like(comat)
    for ai in range(len(ANATOMY_LABELS)):
        if anat_cnt[ai] > 0:
            cond[ai] = comat[ai] / anat_cnt[ai]

    # transition markers (z-line, pylorus, ileocecal) inherit gate values from adjacent anatomy
    # z-line ↔ esophagus, pylorus ↔ stomach, ileocecal ↔ colon
    inherit_map = {
        ANATOMY_LABELS.index("z-line"):           ANATOMY_LABELS.index("esophagus"),
        ANATOMY_LABELS.index("pylorus"):           ANATOMY_LABELS.index("stomach"),
        ANATOMY_LABELS.index("ileocecal valve"):   ANATOMY_LABELS.index("colon"),
    }
    for child, parent in inherit_map.items():
        cond[child] = np.maximum(cond[child], cond[parent] * 0.5)

    gate = np.clip(
        (cond - hard_threshold) / max(soft_threshold - hard_threshold, 1e-9),
        0.0, 1.0,
    )
    gate = gate * (1.0 - min_gate) + min_gate
    return gate.astype(np.float32)


def apply_anatomy_gating(
    anatomy_probs:   np.ndarray,
    pathology_probs: np.ndarray,
    gate:            np.ndarray,
) -> np.ndarray:
    """
    Anatomy-Pathology Soft Gating.

    Parameters
    ----------
    anatomy_probs   : [T, 8]  sigmoid probabilities (0~1)
    pathology_probs : [T, 9]  sigmoid probabilities (0~1)
    gate            : [8, 9]  gate matrix from build_anatomy_pathology_gate()

    Returns
    -------
    gated : [T, 9]  weighted pathology probabilities
    """
    # anatomy-weighted average of gate values -> per-frame effective gate [T, 9]
    effective_gate = anatomy_probs @ gate        # [T, 9]
    effective_gate = np.clip(effective_gate, 0.05, 1.0)
    return pathology_probs * effective_gate


# ── 2. GT-Style Event Composition ────────────────────────────────────────────

def gt_style_event_composition(
    frame_nums:      np.ndarray,
    anatomy_bin:     np.ndarray,
    pathology_bin:   np.ndarray,
    label_names:     Optional[list] = None,
) -> list:
    """
    GT-aligned segment generation: starts a new segment whenever the active label set changes.

    Example: {SI, angiectasia} → {SI} → {SI, angiectasia} produces 3 segments.
    Per-class hysteresis would merge these into 1, reducing temporal IoU.

    Parameters
    ----------
    frame_nums    : [T]       int, actual frame numbers
    anatomy_bin   : [T, 8]    int binary (0 or 1)
    pathology_bin : [T, 9]    int binary (0 or 1)
    label_names   : list of 17 label names (optional, for debug)

    Returns
    -------
    segments : list of dicts
      {
        "start_frame": int,
        "end_frame":   int,
        "labels": {label_name: 1, ...},
        "label_indices": [c1, c2, ...]   (0-16)
      }
    """
    T = len(frame_nums)
    all_bin = np.concatenate([anatomy_bin, pathology_bin], axis=1)  # [T, 17]

    if label_names is None:
        label_names = ANATOMY_LABELS + PATHOLOGY_LABELS

    segments = []
    seg_start = 0
    seg_label_set = tuple(all_bin[0].tolist())

    for t in range(1, T):
        current_set = tuple(all_bin[t].tolist())
        if current_set != seg_label_set:
            # label set changed — close previous segment
            if any(seg_label_set):
                active_idx = [i for i, v in enumerate(seg_label_set) if v]
                segments.append({
                    "start_frame":  int(frame_nums[seg_start]),
                    "end_frame":    int(frame_nums[t - 1]),
                    "labels":       {label_names[i]: 1 for i in active_idx},
                    "label_indices": active_idx,
                })
            seg_start     = t
            seg_label_set = current_set

    # final segment
    if any(seg_label_set):
        active_idx = [i for i, v in enumerate(seg_label_set) if v]
        segments.append({
            "start_frame":  int(frame_nums[seg_start]),
            "end_frame":    int(frame_nums[T - 1]),
            "labels":       {label_names[i]: 1 for i in active_idx},
            "label_indices": active_idx,
        })

    return segments


def segments_to_per_class(
    segments: list,
    n_classes: int = 17,
) -> dict:
    """
    Convert gt_style_event_composition output to a per-class segment list.
    Compatible with inference.py _frames_to_segments format.

    Returns
    -------
    per_class : {cls_idx: [(start, end, 1.0), ...]}
    """
    per_class = {c: [] for c in range(n_classes)}
    for seg in segments:
        for ci in seg["label_indices"]:
            per_class[ci].append((seg["start_frame"], seg["end_frame"], 1.0))
    return per_class


# ── 3. Per-Class Threshold Optimizer ────────────────────────────────────────

def optimize_per_class_thresholds(
    all_probs:   np.ndarray,
    all_labels:  np.ndarray,
    method:      str = "f1",
    n_thresholds: int = 100,
    min_pos:     int = 10,
) -> np.ndarray:
    """
    Search for the optimal per-class threshold (F1 or balanced accuracy).

    Parameters
    ----------
    all_probs  : [N, C]  sigmoid probabilities (0~1)
    all_labels : [N, C]  binary GT (0 or 1)
    method     : "f1" | "ba" (balanced accuracy)
    n_thresholds : number of threshold grid points
    min_pos    : minimum positive samples required (falls back to 0.3 if below)

    Returns
    -------
    thresholds : [C] float
    """
    n, C = all_probs.shape
    thresholds = np.full(C, 0.5)
    grid = np.linspace(0.05, 0.95, n_thresholds)

    for c in range(C):
        y_true = all_labels[:, c]
        y_prob = all_probs[:, c]
        n_pos  = y_true.sum()
        if n_pos < min_pos:
            thresholds[c] = 0.3  # rare class: lower threshold
            continue

        best_score, best_thr = -1.0, 0.5
        for thr in grid:
            y_pred = (y_prob >= thr).astype(float)
            if method == "f1":
                tp = (y_true * y_pred).sum()
                fp = ((1 - y_true) * y_pred).sum()
                fn = (y_true * (1 - y_pred)).sum()
                prec   = tp / max(tp + fp, 1e-8)
                recall = tp / max(tp + fn, 1e-8)
                score  = 2 * prec * recall / max(prec + recall, 1e-8)
            else:  # balanced accuracy
                tp_rate = y_pred[y_true == 1].mean() if n_pos > 0 else 0.0
                n_neg   = len(y_true) - n_pos
                tn_rate = (1 - y_pred)[y_true == 0].mean() if n_neg > 0 else 0.0
                score   = (tp_rate + tn_rate) / 2
            if score > best_score:
                best_score = score
                best_thr   = thr

        thresholds[c] = best_thr

    return thresholds


# ── Anatomy vote smoothing ───────────────────────────────────────────────────

def anatomy_vote_smoothing(
    anatomy_probs: np.ndarray,
    radius:        int = 1,
) -> np.ndarray:
    """
    Local majority voting for anatomy predictions.
    radius=1 uses a 3-frame window. Smaller radius preserves transition zones better.

    Parameters
    ----------
    anatomy_probs : [T, 8]  probabilities or binary values
    radius        : window radius

    Returns
    -------
    smoothed : [T, 8] float (majority-voted 0/1)
    """
    T, C = anatomy_probs.shape
    binary = (anatomy_probs >= 0.5).astype(np.float32)
    smoothed = np.zeros_like(binary)

    for t in range(T):
        lo = max(0, t - radius)
        hi = min(T, t + radius + 1)
        window = binary[lo:hi]
        smoothed[t] = (window.mean(axis=0) >= 0.5).astype(np.float32)

    return smoothed


# ── Debug: print co-occurrence stats ─────────────────────────────────────────

def print_cooccurrence_stats(label_dir: str, threshold: float = 0.01):
    """Print anatomy-pathology co-occurrence summary from data (for debugging)."""
    gate = build_anatomy_pathology_gate(label_dir, soft_threshold=threshold)
    print(f"=== Anatomy-Pathology Gate (threshold={threshold*100:.1f}%) ===")
    header = "".join(f"{p[:8]:>10}" for p in PATHOLOGY_LABELS)
    print(f"{'':22s}{header}")
    for ai, aname in enumerate(ANATOMY_LABELS):
        row = "".join(f"{gate[ai, pi]:>9.3f} " for pi in range(len(PATHOLOGY_LABELS)))
        print(f"  {aname:20s}{row}")
    return gate
