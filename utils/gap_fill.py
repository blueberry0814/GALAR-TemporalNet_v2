"""
Anatomy gap filling post-processing.

GT 분석 결과 기반 MAX_GAP 설정:
  - small intestine: 500 frames  (GT gap p75=621, ≥1000은 실제 전환)
  - colon          : 700 frames  (GT gap 대부분 <1000)
  - stomach        : 400 frames  (GT gap median=97, 보수적)
  - esophagus      : 150 frames  (GT gap 거의 없음)
  - z-line/pylorus/ileocecal/mouth: 적용 안 함 (transition-point)
"""

import numpy as np

# cls index → MAX_GAP (0이면 fill 안 함)
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
    frame_nums: np.ndarray,    # [N] int64, 실제 프레임 번호
    max_gap: dict = None,      # {cls: max_gap_frames}, None이면 DEFAULT 사용
) -> np.ndarray:
    """
    anatomy 클래스(0~7)의 예측 gap을 채워 반환.

    조건:
      1. 같은 클래스가 gap 양쪽에 존재
      2. gap 크기(프레임 수) <= max_gap[cls]
      3. gap 구간 내에서 다른 anatomy 클래스가 과반 이상 활성화되지 않음
         (다른 anatomy가 강하게 예측된 곳은 실제 전환으로 간주)

    Returns:
        수정된 pred_binary 복사본 [N, 17]
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
            # active 구간 시작 찾기
            if not arr[i]:
                i += 1
                continue

            # active 구간 끝 찾기
            seg1_end = i
            while seg1_end + 1 < N and arr[seg1_end + 1]:
                seg1_end += 1

            # gap 시작
            gap_start = seg1_end + 1
            if gap_start >= N:
                break

            # gap 끝 (다음 active 구간 시작)
            j = gap_start
            while j < N and not arr[j]:
                j += 1

            if j >= N:
                # 다음 active 구간 없음
                i = j
                continue

            seg2_start = j
            gap_end = seg2_start - 1

            # gap 크기 (실제 프레임 수)
            gap_frames = int(frame_nums[gap_end]) - int(frame_nums[gap_start]) + 1

            if gap_frames <= mg:
                # 조건 3: gap 구간 내 다른 anatomy가 과반 이상인지 확인
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
