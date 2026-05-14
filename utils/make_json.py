"""
예측 결과 (프레임별 이진 배열) → 대회 제출용 JSON 변환

대회 JSON 형식:
{
  "videos": [
    {
      "video_id": "VID_001",
      "events": [
        {"start": 0, "end": 100, "label": "mouth"},
        {"start": 101, "end": 200, "label": "ulcer"}
      ]
    }
  ]
}

start/end 는 CSV의 'frame' 컬럼 값 (실제 프레임 번호)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


ANATOMY_LABELS = [
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus", "ileocecal valve"
]
PATHOLOGY_LABELS = [
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer"
]
ALL_LABELS = ANATOMY_LABELS + PATHOLOGY_LABELS  # 17개, 순서 고정


def predictions_to_events(frame_nums: np.ndarray, pred_binary: np.ndarray, video_id: str) -> dict:
    """
    프레임 예측 배열을 이벤트 구간으로 변환.

    Args:
        frame_nums  : [N] 실제 프레임 번호 배열 (CSV frame 컬럼)
        pred_binary : [N, 17] 이진 예측 (0 or 1)
        video_id    : 비디오 ID 문자열

    Returns:
        {"video_id": ..., "events": [...]}
    """
    assert len(frame_nums) == len(pred_binary), "프레임 수와 예측 수가 다릅니다"

    events = []
    N = len(frame_nums)

    # 각 라벨마다 독립적으로 연속 구간 탐색
    for cls_idx, label_name in enumerate(ALL_LABELS):
        active = pred_binary[:, cls_idx].astype(bool)
        if not active.any():
            continue

        # 연속된 활성 구간 찾기
        in_event = False
        start_frame = None

        for i in range(N):
            if active[i] and not in_event:
                in_event = True
                start_frame = int(frame_nums[i])
            elif not active[i] and in_event:
                in_event = False
                end_frame = int(frame_nums[i - 1])
                events.append({"start": start_frame, "end": end_frame, "label": label_name})

        if in_event:  # 끝까지 활성인 경우
            events.append({"start": start_frame, "end": int(frame_nums[-1]), "label": label_name})

    # start 기준 정렬
    events.sort(key=lambda e: (e["start"], e["label"]))
    return {"video_id": video_id, "events": events}


def build_json_from_predictions(video_predictions: list, output_path: str):
    """
    여러 비디오 예측을 합쳐 최종 JSON 생성.

    Args:
        video_predictions: list of {"video_id", "frame_nums", "pred_binary"} dicts
        output_path: 저장할 .json 파일 경로
    """
    result = {"videos": []}
    for vp in tqdm(video_predictions, desc="Building JSON"):
        vid_events = predictions_to_events(
            frame_nums=vp["frame_nums"],
            pred_binary=vp["pred_binary"],
            video_id=vp["video_id"]
        )
        result["videos"].append(vid_events)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"JSON 저장 완료: {output_path}")


def build_json_from_gt_csvs(labels_dir: str, output_path: str):
    """
    Ground truth CSV 파일들로부터 JSON 생성 (검증용).
    """
    labels_path = Path(labels_dir)
    result = {"videos": []}

    for csv_path in tqdm(sorted(labels_path.glob("*.csv")), desc="Processing GT CSVs"):
        video_id = csv_path.stem
        df = pd.read_csv(csv_path)
        df = df.sort_values("frame").reset_index(drop=True)

        label_cols = [c for c in ALL_LABELS if c in df.columns]
        frame_nums = df["frame"].values.astype(np.int64)
        labels = df[label_cols].values.astype(np.float32)

        # ALL_LABELS 순서로 맞추기
        full_labels = np.zeros((len(df), 17), dtype=np.float32)
        for i, lbl in enumerate(ALL_LABELS):
            if lbl in label_cols:
                full_labels[:, i] = labels[:, label_cols.index(lbl)]

        vid_events = predictions_to_events(frame_nums, full_labels.astype(int), video_id)
        result["videos"].append(vid_events)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"GT JSON 저장 완료: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_dir", type=str, required=True, help="CSV 라벨 폴더 경로")
    parser.add_argument("--output", type=str, default="gt_events.json")
    args = parser.parse_args()
    build_json_from_gt_csvs(args.labels_dir, args.output)
