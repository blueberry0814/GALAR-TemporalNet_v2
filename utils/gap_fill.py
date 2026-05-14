"""
Anatomy gap filling post-processing.

MAX_GAP values derived from GT analysis:
  - small intestine: 500 frames  (GT gap p75=621; gaps >=1000 are true transitions)
  - colon          : 700 frames  (GT gaps mostly <1000)
  - stomach        : 400 frames  (GT gap median=97; conservative)
  - esophagus      : 150 frames  (GT gaps rarely large)
  - z-line/pylorus/ileocecal/mouth: not filled (transition-point classes)
"""

import numpy as np

# class index -> MAX_GAP in frames (0 = skip)
DEFAULT_MAX_GAP = {
    0: 0,    # mouth           — transition point
    1: 150,  # esophagus
    2: 400,  # stomach
    3: 500,  # small intestine
    4: 700,  # colon
    5: 0,    # z-line          — transition point
    6: 0,    # pylorus         — transition point
    7: 0,    # ileocecal valve — transition point
}


def fill_anatomy_gaps(
    pred_binary: np.ndarray,   # [N, 17] int32
    frame_nums: np.ndarray,    # [N] int64, actual frame numbers
    max_gap: dict = None,      # {cls: max_gap_frames}; defaults to DEFAULT_MAX_GAP
) -> np.ndarray:
    """
    Fill prediction gaps for anatomy classes (indices 0-7).

    A gap is filled when all three conditions hold:
      1. The same class is active on both sides of the gap
      2. Gap size (in frames) <= max_gap[cls]
      3. No other anatomy class is active for >50% of the gap frames
         (majority competing anatomy indicates a true transition)

    Returns:
        Modified copy of pred_binary [N, 17]
    """
    if max_gap is None:
        max_gap = DEFAULT_MAX_GAP

    out = pred_binary.copy()

    for cls in range(8):  # anatomy only
        mg = max_gap.get(cls, 0)
        if mg == 0:
            continue

        arr = out[:, cls].astype(bool)
        N = len(arr)
        i = 0

        while i < N:
            # find start of active segment
            if not arr[i]:
                i += 1
                continue

            # find end of active segment
            seg1_end = i
            while seg1_end + 1 < N and arr[seg1_end + 1]:
                seg1_end += 1

            # gap starts here
            gap_start = seg1_end + 1
            if gap_start >= N:
                break

            # find start of next active segment
            j = gap_start
            while j < N and not arr[j]:
                j += 1

            if j >= N:
                # no next active segment
                i = j
                continue

            seg2_start = j
            gap_end = seg2_start - 1

            # gap size in actual frames
            gap_frames = int(frame_nums[gap_end]) - int(frame_nums[gap_start]) + 1

            if gap_frames <= mg:
                # condition 3: check if any other anatomy dominates the gap
                gap_len = gap_end - gap_start + 1
                competing = False
                for other_cls in range(8):
                    if other_cls == cls:
                        continue
                    other_active = out[gap_start:gap_end + 1, other_cls].sum()
                    if other_active > gap_len * 0.5:
                        competing = True
                        break

                if not competing:
                    out[gap_start:gap_end + 1, cls] = 1

            i = seg2_start

    return out
