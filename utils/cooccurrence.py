"""
Anatomy-Pathology co-occurrence matrix 계산 및 gating 적용.

P[i][j] = anatomy_i 가 활성인 프레임에서 pathology_j 가 등장하는 비율
         (프레임 단위, 학습 데이터 전체 집계)

사용:
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
    학습 레이블 CSV 전체를 읽어 [8, 9] co-occurrence matrix를 계산.
    co_matrix[i][j] = anatomy_i 프레임 중 pathology_j 가 활성인 비율 (0.0~1.0)
    """
    co_count  = np.zeros((8, 9), dtype=np.float64)   # 동시 활성 count
    anat_count = np.zeros(8, dtype=np.float64)         # anatomy 활성 count

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
    """캐시가 있으면 로드, 없으면 계산 후 저장."""
    if os.path.exists(cache_path):
        matrix = np.load(cache_path)
        print(f"[Co-occurrence] 캐시 로드: {cache_path}")
        return matrix

    print(f"[Co-occurrence] 계산 중... ({labels_dir})")
    matrix = compute_cooccurrence(labels_dir)
    np.save(cache_path, matrix)
    print(f"[Co-occurrence] 저장 완료: {cache_path}")
    _print_matrix(matrix)
    return matrix


def apply_cooccurrence_gating(
    pred_probs: np.ndarray,     # [T, 17] — anatomy(0:8) + pathology(8:17)
    co_matrix:  np.ndarray,     # [8, 9]
    threshold:  float = 0.01,
    mode:       str   = "hard",
) -> np.ndarray:
    """
    두 단계 gating:
      Level 1 (hard): 학습 데이터 전체에서 co_occur = 0.000 인 조합 → 무조건 0
                      (데이터로 확인된 생물학적 불가능)
      Level 2 (soft): 0 < co_occur < threshold 인 조합 → 확률 비례 감쇄
                      (드물지만 가능 → 완전 제거 아님)

    mode="hard": level1만 적용 (0% 조합만 제거)
    mode="soft": level1 + level2 모두 적용
    """
    out = pred_probs.copy()

    anat_probs = pred_probs[:, :8]   # [T, 8]
    path_probs = pred_probs[:, 8:]   # [T, 9]

    # anatomy 확률로 각 pathology의 최대 공존율 계산
    # max_co[t][j] = max over anatomies of (anat_prob[i] > 0.5 ? co_matrix[i][j] : 0)
    # → dominant anatomy 기준이라 더 직관적
    # weighted_co[t][j]: anatomy 분포를 고려한 기대 공존율
    anat_w = anat_probs / (anat_probs.sum(axis=1, keepdims=True) + 1e-8)
    weighted_co = anat_w @ co_matrix   # [T, 9]

    # Level 1: 학습 데이터에서 한 번도 공존하지 않은 조합 완전 제거
    # max_possible_co[j] = max over all anatomy of co_matrix[i][j]
    # → 어떤 anatomy이든 0%이면 절대 불가능
    # 각 프레임의 dominant anatomy에서 해당 pathology 공존율이 0인지 확인
    dominant_anat = np.argmax(anat_probs, axis=1)           # [T]
    dominant_co   = co_matrix[dominant_anat]                # [T, 9]
    hard_possible = (dominant_co > 0).astype(np.float32)    # [T, 9] — 0이면 절대 불가

    if mode == "hard":
        # Level 1만: 정확히 0%인 조합만 제거
        out[:, 8:] = path_probs * hard_possible
    else:
        # Level 1 + Level 2: 0% 제거 + 희귀 감쇄
        soft_scale = np.clip(weighted_co / (threshold + 1e-8), 0.0, 1.0)
        out[:, 8:] = path_probs * hard_possible * soft_scale

    return out


def _print_matrix(co_matrix: np.ndarray):
    """디버그용 출력."""
    print("\n[Co-occurrence Matrix] P(pathology | anatomy)")
    header = "            " + " ".join(f"{p[:6]:>8}" for p in PATHOLOGY_LABELS)
    print(header)
    for i, anat in enumerate(ANATOMY_LABELS):
        row_str = f"{anat:>12}" + " ".join(f"{co_matrix[i][j]:8.3f}" for j in range(9))
        print(row_str)
    print()
