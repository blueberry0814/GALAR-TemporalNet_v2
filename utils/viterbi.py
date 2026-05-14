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

# Per-class median filter window (larger for long-duration classes, smaller for transition markers)
# indices: mouth=0, esophagus=1, stomach=2, SI=3, colon=4, z-line=5, pylorus=6, ileocecal=7
_ANATOMY_SMOOTH_WINDOWS = [5, 5, 11, 21, 11, 3, 3, 5]


def smooth_anatomy_probs(pred_probs_anatomy: np.ndarray) -> np.ndarray:
    """
    Apply per-class median filter with class-specific window sizes.
    - stomach(2), SI(3), colon(4): large window to remove short dips
    - z-line(5), pylorus(6): small window to preserve brief events
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
    Forward-only Viterbi: suppress backward transitions while preserving probability values.

    Problem with naive Viterbi:
      - Suppressing all non-winner classes by (1-alpha) collapses the second class
        in transition zones, causing boundary fragmentation (e.g., SI+pylorus impossible)

    This implementation:
      - Computes the Viterbi path to determine the current sequence position (path_pos)
      - Suppresses only classes that are more than (lookback+1) steps behind path_pos
      - Classes within the transition zone (<=lookback steps behind) and forward classes
        retain their original probabilities
      - Result: in an SI segment, pylorus (1 step behind) survives; stomach (2 steps behind) is suppressed

    Args:
        pred_probs_anatomy : [T, 8] sigmoid probabilities
        alpha    : suppression strength — 0=none, 1=full removal (default 0.7 → ×0.3)
        lookback : transition zone width — 1 means 1 step behind is allowed

    Returns:
        [T, 8] forward-constrained probabilities
    """
    T, N = pred_probs_anatomy.shape
    if T == 0 or N != _N_ANAT:
        return pred_probs_anatomy

    log_trans = _build_log_trans()

    # Viterbi path computation (normalize probabilities for path only)
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

    # Forward-only constraint: suppress classes more than (lookback+1) steps behind path
    # transition zone (within lookback) and forward classes keep original probabilities
    smoothed = pred_probs_anatomy.astype(np.float32).copy()
    suppress = 1.0 - alpha  # fraction of probability retained after suppression
    for t in range(T):
        path_pos = _POS_IN_SEQ[path[t]]
        for cls in range(N):
            if _POS_IN_SEQ[cls] < path_pos - lookback:
                smoothed[t, cls] *= suppress

    return smoothed
