"""
Convert per-frame binary predictions to challenge submission JSON.

Submission JSON format:
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

start/end are the 'frame' column values from the label CSV (actual frame numbers).
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
ALL_LABELS = ANATOMY_LABELS + PATHOLOGY_LABELS  # 17 classes, fixed order


def predictions_to_events(frame_nums: np.ndarray, pred_binary: np.ndarray, video_id: str) -> dict:
    """
    Convert per-frame binary prediction array to a list of events.

    Args:
        frame_nums  : [N] actual frame numbers (from CSV 'frame' column)
        pred_binary : [N, 17] binary predictions (0 or 1)
        video_id    : video ID string

    Returns:
        {"video_id": ..., "events": [...]}
    """
    assert len(frame_nums) == len(pred_binary), "frame_nums and pred_binary length mismatch"

    events = []
    N = len(frame_nums)

    # find contiguous active segments independently per label
    for cls_idx, label_name in enumerate(ALL_LABELS):
        active = pred_binary[:, cls_idx].astype(bool)
        if not active.any():
            continue

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

        if in_event:  # active until the last frame
            events.append({"start": start_frame, "end": int(frame_nums[-1]), "label": label_name})

    events.sort(key=lambda e: (e["start"], e["label"]))
    return {"video_id": video_id, "events": events}


def build_json_from_predictions(video_predictions: list, output_path: str):
    """
    Merge predictions from multiple videos into the final submission JSON.

    Args:
        video_predictions: list of {"video_id", "frame_nums", "pred_binary"} dicts
        output_path: path to save the .json file
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
    print(f"Saved: {output_path}")


def build_json_from_gt_csvs(labels_dir: str, output_path: str):
    """
    Build a GT-format JSON from label CSV files (for evaluation / debugging).
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

        # reorder to match ALL_LABELS order
        full_labels = np.zeros((len(df), 17), dtype=np.float32)
        for i, lbl in enumerate(ALL_LABELS):
            if lbl in label_cols:
                full_labels[:, i] = labels[:, label_cols.index(lbl)]

        vid_events = predictions_to_events(frame_nums, full_labels.astype(int), video_id)
        result["videos"].append(vid_events)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved GT JSON: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_dir", type=str, required=True, help="Directory containing label CSV files")
    parser.add_argument("--output", type=str, default="gt_events.json")
    args = parser.parse_args()
    build_json_from_gt_csvs(args.labels_dir, args.output)
