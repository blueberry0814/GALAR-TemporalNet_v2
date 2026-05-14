"""
Viterbi post-processing for anatomy sequence predictions.

VCE (Video Capsule Endoscopy) anatomy progression order:
  mouth(0) → esophagus(1) → z-line(5) → stomach(2) → pylorus(6)
  → small intestine(3) → ileocecal valve(7) → colon(4)

Viterbi enforces this biological sequence constraint on per-frame predictions,
boosting short-duration transition classes (pylorus, z-line, ileocecal valve).
"""

import numpy as np
from scipy.signal import medfilt

# Linear sequence index in ANATOMY_LABELS order:
# mouth=0, esophagus=1, stomach=2, SI=3, colon=4, z-line=5, pylorus=6, ileocecal=7
_ANAT_LINEAR_SEQ = [0, 1, 5, 2, 6, 3, 7, 4]
_POS_IN_SEQ = {cls: i for i, cls in enumerate(_ANAT_LINEAR_SEQ)}

# 클래스별 smooth window (long-duration 클래스는 크게, transition marker는 작게)
# 인덱스: mouth=0, esophagus=1, stomach=2, SI=3, colon=4, z-line=5, pylorus=6, ileocecal=7
_ANATOMY_SMOOTH_WINDOWS = [5, 5, 11, 21, 11, 3, 3, 5]


def smooth_anatomy_probs(pred_probs_anatomy: np.ndarray) -> np.ndarray:
    """
    클래스별로 다른 median filter window 적용.
    - stomach(2), SI(3), colon(4): 큰 window → 중간의 짧은 dip 제거
    - z-line(5), pylorus(6): 작은 window → 짧은 이벤트 보존
    """
    out = pred_probs_anatomy.copy()
    for c, w in enumerate(_ANATOMY_SMOOTH_WINDOWS):
        if w > 1:
            out[:, c] = medfilt(out[:, c].astype(np.float32), kernel_size=w)
    return out

# Log transition matrix (precomputed)
_N_ANAT = 8
_LOG_TRANS = None


def _build_log_trans():
    global _LOG_TRANS
    if _LOG_TRANS is not None:
        return _LOG_TRANS

    LOG_STAY  = np.log(0.90)
    LOG_ADJ   = np.log(0.75)   # 1 step apart (e.g., stomach → pylorus)
    LOG_SKIP1 = np.log(0.15)   # 2 steps (e.g., stomach → SI, skipping pylorus)
    LOG_SKIP2 = np.log(0.03)   # 3 steps
    LOG_FAR   = np.log(1e-4)   # physiologically impossible

    t = np.full((_N_ANAT, _N_ANAT), LOG_FAR, dtype=np.float64)
    for i in range(_N_ANAT):
        for j in range(_N_ANAT):
            if i == j:
                t[i, j] = LOG_STAY
            else:
                dist = abs(_POS_IN_SEQ[i] - _POS_IN_SEQ[j])
                if dist == 1:
                    t[i, j] = LOG_ADJ
                elif dist == 2:
                    t[i, j] = LOG_SKIP1
                elif dist == 3:
                    t[i, j] = LOG_SKIP2
    _LOG_TRANS = t
    return _LOG_TRANS


def viterbi_anatomy(
    pred_probs_anatomy: np.ndarray,
    alpha:    float = 0.7,
    lookback: int   = 1,
) -> np.ndarray:
    """
    Forward-only Viterbi: 역방향 전환만 억제, 확률값은 최대한 보존.

    기존 방식 문제:
      - 승자 외 모든 클래스를 (1-alpha)로 억제 → transition zone에서 두 번째 클래스 소멸
      - SI + pylorus 동시 활성 불가 → 경계 파편화 원인

    변경 방식:
      - Viterbi path로 현재 시퀀스 위치(path_pos) 계산
      - path_pos 기준 lookback 이하 단계 뒤 클래스만 억제 (역방향 방지)
      - lookback 이내(transition zone)와 앞 방향: 원래 확률 그대로 보존
      - 결과: SI 구간에서 pylorus(1단계 뒤)는 살고, stomach(2단계 뒤)만 억제

    Args:
        pred_probs_anatomy : [T, 8] sigmoid probabilities
        alpha    : 억제 강도 — 0=억제 없음, 1=완전 제거 (default 0.7 → ×0.3)
        lookback : transition zone 허용 범위 — 1=1단계 뒤까지 허용

    Returns:
        [T, 8] forward-constrained probabilities
    """
    T, N = pred_probs_anatomy.shape
    if T == 0 or N != _N_ANAT:
        return pred_probs_anatomy

    log_trans = _build_log_trans()

    # Viterbi path 계산 (확률 정규화는 path 계산 전용)
    probs    = pred_probs_anatomy.astype(np.float64)
    probs    = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
    log_emit = np.log(probs + 1e-8)

    V       = np.full((T, N), -np.inf, dtype=np.float64)
    backptr = np.zeros((T, N), dtype=np.int32)
    V[0]    = log_emit[0]
    for t in range(1, T):
        trans_scores = V[t - 1, :, None] + log_trans
        best         = trans_scores.argmax(axis=0)
        V[t]         = trans_scores[best, np.arange(N)] + log_emit[t]
        backptr[t]   = best

    path     = np.zeros(T, dtype=np.int32)
    path[-1] = int(V[-1].argmax())
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]

    # Forward-only constraint: path보다 (lookback+1)단계 이상 뒤에 있는 클래스만 억제
    # transition zone (lookback 이내)와 앞 방향은 원래 확률 보존
    smoothed = pred_probs_anatomy.astype(np.float32).copy()
    suppress = 1.0 - alpha  # 억제된 클래스가 유지하는 비율
    for t in range(T):
        path_pos = _POS_IN_SEQ[path[t]]
        for cls in range(N):
            if _POS_IN_SEQ[cls] < path_pos - lookback:
                smoothed[t, cls] *= suppress

    return smoothed
