"""
Anatomy-Pathology co-occurrence matrix and gating.

P[i][j] = fraction of anatomy_i frames where pathology_j is also active
           (aggregated over the entire training set, frame-level)

Usage:
  matrix = compute_or_load_cooccurrence(labels_dir, cache_path)
  pred_probs = apply_cooccurrence_gating(pred_probs, matrix, threshold=0.02)
"""

import os
import csv
import numpy as np

ANATOMY_LABELS = [
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus", "ileocecal valve",
]
PATHOLOGY_LABELS = [
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer",
]


def compute_cooccurrence(labels_dir: str) -> np.ndarray:
    """
    Read all training label CSVs and compute the [8, 9] co-occurrence matrix.
    co_matrix[i][j] = P(pathology_j active | anatomy_i active), range 0.0~1.0
    """
    co_count   = np.zeros((8, 9), dtype=np.float64)  # joint active count
    anat_count = np.zeros(8, dtype=np.float64)        # anatomy active count

    csv_files = sorted([
        os.path.join(labels_dir, f)
        for f in os.listdir(labels_dir)
        if f.endswith(".csv")
    ])

    for csv_path in csv_files:
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for ai, anat in enumerate(ANATOMY_LABELS):
                        if row.get(anat, "0").strip() == "1":
                            anat_count[ai] += 1
                            for pi, path in enumerate(PATHOLOGY_LABELS):
                                if row.get(path, "0").strip() == "1":
                                    co_count[ai][pi] += 1
        except Exception:
            continue

    # P(pathology_j | anatomy_i)
    safe_anat = np.maximum(anat_count, 1)[:, np.newaxis]
    co_matrix = co_count / safe_anat
    return co_matrix.astype(np.float32)


def compute_or_load_cooccurrence(labels_dir: str, cache_path: str) -> np.ndarray:
    """Load from cache if available; otherwise compute and save."""
    if os.path.exists(cache_path):
        matrix = np.load(cache_path)
        print(f"[Co-occurrence] loaded from cache: {cache_path}")
        return matrix

    print(f"[Co-occurrence] computing... ({labels_dir})")
    matrix = compute_cooccurrence(labels_dir)
    np.save(cache_path, matrix)
    print(f"[Co-occurrence] saved: {cache_path}")
    _print_matrix(matrix)
    return matrix


def apply_cooccurrence_gating(
    pred_probs: np.ndarray,     # [T, 17] — anatomy(0:8) + pathology(8:17)
    co_matrix:  np.ndarray,     # [8, 9]
    threshold:  float = 0.01,
    mode:       str   = "hard",
) -> np.ndarray:
    """
    Two-level gating based on anatomy-pathology co-occurrence statistics:
      Level 1 (hard): zero out pathology predictions for anatomy-pathology pairs
                      with co_occur = 0.000 across the entire training set
                      (biologically impossible combinations)
      Level 2 (soft): proportionally scale down pairs with 0 < co_occur < threshold
                      (rare but possible — not fully suppressed)

    mode="hard": level 1 only (remove zero-cooccurrence pairs)
    mode="soft": level 1 + level 2
    """
    out = pred_probs.copy()

    anat_probs = pred_probs[:, :8]   # [T, 8]
    path_probs = pred_probs[:, 8:]   # [T, 9]

    # weighted co-occurrence: anatomy probability distribution weighted expected co-rate
    anat_w = anat_probs / (anat_probs.sum(axis=1, keepdims=True) + 1e-8)
    weighted_co = anat_w @ co_matrix   # [T, 9]

    # Level 1: suppress pairs where the dominant anatomy has 0% co-occurrence
    dominant_anat = np.argmax(anat_probs, axis=1)           # [T]
    dominant_co   = co_matrix[dominant_anat]                # [T, 9]
    hard_possible = (dominant_co > 0).astype(np.float32)    # [T, 9]

    if mode == "hard":
        out[:, 8:] = path_probs * hard_possible
    else:
        # Level 1 + Level 2: remove zero pairs and scale rare pairs
        soft_scale = np.clip(weighted_co / (threshold + 1e-8), 0.0, 1.0)
        out[:, 8:] = path_probs * hard_possible * soft_scale

    return out


def _print_matrix(co_matrix: np.ndarray):
    """Print co-occurrence matrix for debugging."""
    print("\n[Co-occurrence Matrix] P(pathology | anatomy)")
    header = "            " + " ".join(f"{p[:6]:>8}" for p in PATHOLOGY_LABELS)
    print(header)
    for i, anat in enumerate(ANATOMY_LABELS):
        row_str = f"{anat:>12}" + " ".join(f"{co_matrix[i][j]:8.3f}" for j in range(9))
        print(row_str)
    print()
