"""
Inference Script — GI Endoscopy Test Submission
=================================================
Runs sliding-window inference on pre-extracted features and saves
per-video JSON files for challenge submission.

Two modes:
  1. Pre-extracted features (recommended, fast):
       python predict.py \\
           --feat_dir   ./features_test \\
           --labels_dir ./test/Labels \\
           --checkpoint ./checkpoints/best_model.pth \\
           --config     configs/example.yaml \\
           --output_dir ./results

  2. On-the-fly extraction (slower, requires raw frames):
       python predict.py \\
           --frames_root ./test/Frames \\
           --labels_dir  ./test/Labels \\
           --checkpoint  ./checkpoints/best_model.pth \\
           --config      configs/example.yaml \\
           --output_dir  ./results

Output:
  results/{video_id}.json  — per-video event prediction for submission
  results/predictions.json — all videos combined (final submission file)
"""

import os
import json
import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from scipy.signal import medfilt
from tqdm import tqdm

from models.model import GalarModel
from utils.viterbi import viterbi_anatomy, smooth_anatomy_probs
from utils.cooccurrence import compute_or_load_cooccurrence, apply_cooccurrence_gating
from utils.gap_fill import fill_anatomy_gaps
from utils.make_json import ALL_LABELS

ANATOMY_LABELS = [
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus", "ileocecal valve",
]
PATHOLOGY_LABELS = [
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer",
]


# ── Feature extraction (on-the-fly mode) ─────────────────────────────────────

def get_transform(image_size=518):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_batch(model, imgs, device):
    tensor = torch.stack(imgs).to(device)
    out    = model.forward_features(tensor)
    cls    = out["x_norm_clstoken"]
    patch  = out["x_norm_patchtokens"].mean(dim=1)
    return torch.cat([cls, patch], dim=-1).cpu().float().numpy()


def extract_video_features(frame_paths, dinov2, transform, device, batch_size=32):
    all_feats, batch = [], []
    for path in tqdm(frame_paths, desc="  extracting", leave=False):
        try:
            batch.append(transform(Image.open(path).convert("RGB")))
        except Exception:
            continue
        if len(batch) == batch_size:
            all_feats.append(extract_batch(dinov2, batch, device))
            batch.clear()
    if batch:
        all_feats.append(extract_batch(dinov2, batch, device))
    return np.vstack(all_feats).astype(np.float32) if all_feats else np.zeros((0, 2048), np.float32)


# ── Sliding-window inference ──────────────────────────────────────────────────

@torch.no_grad()
def infer_video(model, features, config, device, co_matrix=None,
                threshold=None, infer_stride=None):
    N, D       = features.shape
    window     = config["data"]["window_size"]
    stride     = infer_stride if infer_stride is not None else config["data"]["infer_stride"]
    thr        = threshold if threshold is not None else config["inference"]["threshold"]
    smooth_w   = config["inference"]["smooth_window"]
    co_thr     = config["inference"].get("cooccurrence_threshold", 0.01)
    co_mode    = config["inference"].get("cooccurrence_mode", "hard")

    pred_sum   = np.zeros((N, 17), np.float64)
    pred_count = np.zeros(N, np.int32)

    starts = list(range(0, max(1, N - window + 1), stride))
    if N > window and (N - window) % stride != 0:
        starts.append(N - window)
    if not starts:
        starts = [0]

    for s in starts:
        e     = min(s + window, N)
        chunk = features[s:e].astype(np.float32)
        T_a   = len(chunk)
        if T_a < window:
            chunk = np.vstack([chunk, np.zeros((window - T_a, D), np.float32)])
        feat_t = torch.from_numpy(chunk).unsqueeze(0).to(device)
        vp_r   = torch.tensor([s / max(N - 1, 1)], dtype=torch.float32).to(device)
        a_l, p_l = model(features=feat_t, raw_features=feat_t, video_pos_ratio=vp_r)
        preds = torch.cat([torch.sigmoid(a_l), torch.sigmoid(p_l)], dim=-1).squeeze(0).cpu().numpy()
        pred_sum[s:s + T_a]   += preds[:T_a]
        pred_count[s:s + T_a] += 1

    pred_probs = (pred_sum / np.maximum(pred_count, 1)[:, None]).astype(np.float32)

    if smooth_w > 1:
        for c in range(8, 17):
            pred_probs[:, c] = medfilt(pred_probs[:, c], kernel_size=smooth_w)
    pred_probs[:, :8] = smooth_anatomy_probs(pred_probs[:, :8])
    pred_probs[:, :8] = viterbi_anatomy(pred_probs[:, :8])

    if co_matrix is not None:
        pred_probs = apply_cooccurrence_gating(pred_probs, co_matrix, co_thr, co_mode)

    return (pred_probs >= thr).astype(np.int32), pred_probs


# ── JSON event builder ────────────────────────────────────────────────────────

def to_events(frame_nums, pred_binary, video_id, min_seg_frames=5):
    """Convert per-frame binary predictions to event list."""
    events = []
    N = len(frame_nums)
    for c, label in enumerate(ALL_LABELS):
        in_seg, seg_start = False, None
        for i in range(N):
            if pred_binary[i, c] and not in_seg:
                in_seg, seg_start = True, int(frame_nums[i])
            elif not pred_binary[i, c] and in_seg:
                if int(frame_nums[i - 1]) - seg_start + 1 >= min_seg_frames:
                    events.append({"start": seg_start, "end": int(frame_nums[i - 1]), "label": label})
                in_seg = False
        if in_seg:
            if int(frame_nums[-1]) - seg_start + 1 >= min_seg_frames:
                events.append({"start": seg_start, "end": int(frame_nums[-1]), "label": label})
    events.sort(key=lambda e: (e["start"], e["label"]))
    return {"video_id": video_id, "events": events}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, required=True,
                        help="Path to config YAML (e.g. configs/example.yaml)")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to trained checkpoint (best_model.pth)")
    parser.add_argument("--labels_dir",  type=str, required=True,
                        help="Directory containing per-video CSV files with frame lists")
    parser.add_argument("--output_dir",  type=str, default="./results")

    # Feature source — use one of:
    parser.add_argument("--feat_dir",    type=str, default=None,
                        help="Pre-extracted feature directory ({video_id}_features.npy + _frames.npy)")
    parser.add_argument("--frames_root", type=str, default=None,
                        help="Root directory of raw frame images (on-the-fly extraction fallback)")

    parser.add_argument("--video_ids",   type=str, default=None,
                        help="Comma-separated video IDs to process. Default: all CSVs in labels_dir")
    parser.add_argument("--threshold",   type=float, default=None,
                        help="Override config inference threshold")
    parser.add_argument("--infer_stride",type=int,   default=None,
                        help="Override config inference stride")
    parser.add_argument("--min_seg_frames", type=int, default=20,
                        help="Minimum segment length in frames (shorter = noise, default 20)")
    parser.add_argument("--fill_gap",   action="store_true",
                        help="Apply anatomy gap filling post-processing")
    parser.add_argument("--output_name",type=str, default="predictions",
                        help="Output JSON filename without extension (default: predictions)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "config" in ckpt:
        config["model"] = ckpt["config"]["model"]  # use checkpoint model arch
    model = GalarModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.anatomy_proto_initialized = ckpt.get("anatomy_proto_initialized", False)
    model.normal_proto_initialized  = ckpt.get("normal_proto_initialized", False)
    model.eval()
    print(f"  epoch={ckpt.get('epoch', '?')}  val_tMAP={ckpt.get('val_tmap', 0):.4f}")

    # Co-occurrence gating
    co_matrix = None
    if config["inference"].get("use_cooccurrence_gating", False):
        co_matrix = compute_or_load_cooccurrence(
            config["inference"].get("cooccurrence_labels_dir", args.labels_dir),
            config["inference"].get("cooccurrence_cache", "./utils/cooccurrence_matrix.npy"),
        )
        print(f"Co-occurrence gating ON")

    # DINOv2 for on-the-fly extraction (lazy load)
    dinov2, transform = None, get_transform(518)

    labels_path = Path(args.labels_dir)
    video_ids = (
        [v.strip() for v in args.video_ids.split(",")]
        if args.video_ids
        else sorted([p.stem for p in labels_path.glob("*.csv")])
    )
    print(f"\nProcessing {len(video_ids)} videos...")
    print(f"threshold={args.threshold or config['inference']['threshold']}  "
          f"min_seg={args.min_seg_frames}  fill_gap={args.fill_gap}\n")

    all_results = []

    for vid_id in video_ids:
        print(f"{'='*50}")
        print(f"[{vid_id}]")

        # --- Load features ---
        features, frame_nums = None, None
        if args.feat_dir:
            fp = os.path.join(args.feat_dir, f"{vid_id}_features.npy")
            fnp = os.path.join(args.feat_dir, f"{vid_id}_frames.npy")
            if os.path.exists(fp) and os.path.exists(fnp):
                features   = np.load(fp).astype(np.float32)
                frame_nums = np.load(fnp).astype(np.int64)
                print(f"  Loaded from cache: {features.shape}")

        if features is None and args.frames_root:
            import pandas as pd
            label_csv = labels_path / f"{vid_id}.csv"
            if not label_csv.exists():
                print(f"  [SKIP] No label CSV: {label_csv}")
                continue
            # Support both train and test CSV formats
            try:
                df = pd.read_csv(str(label_csv), sep=";")
                frame_files = df["frame_file"].tolist()
                frame_nums  = np.array([int(f.replace("frame_","").split(".")[0]) for f in frame_files])
            except Exception:
                df = pd.read_csv(str(label_csv)).sort_values("frame")
                frame_nums  = df["frame"].values.astype(np.int64)
                frame_files = [f"frame_{n:06d}.PNG" for n in frame_nums]

            video_dir   = os.path.join(args.frames_root, str(vid_id))
            frame_paths = [os.path.join(video_dir, f) for f in frame_files]

            if dinov2 is None:
                print("  Loading DINOv2 ViT-L/14...")
                dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", verbose=False)
                dinov2.eval().to(device)
            features = extract_video_features(frame_paths, dinov2, transform, device)
            print(f"  Extracted on-the-fly: {features.shape}")

        if features is None or len(features) == 0:
            print(f"  [SKIP] No features available. Provide --feat_dir or --frames_root.")
            continue

        if frame_nums is None:
            frame_nums = np.arange(len(features), dtype=np.int64)

        # --- Inference ---
        pred_binary, pred_probs = infer_video(
            model, features, config, device, co_matrix,
            threshold=args.threshold, infer_stride=args.infer_stride,
        )

        # --- Post-processing ---
        if args.fill_gap:
            pred_binary = fill_anatomy_gaps(pred_binary, frame_nums)

        # --- Build events ---
        vid_result = to_events(frame_nums, pred_binary, str(vid_id), args.min_seg_frames)
        all_results.append(vid_result)
        print(f"  Events: {len(vid_result['events'])}")

        # Save per-video JSON
        per_vid_path = os.path.join(args.output_dir, f"{vid_id}.json")
        with open(per_vid_path, "w") as f:
            json.dump(vid_result, f, indent=2)

    # Save combined submission JSON
    out_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    with open(out_path, "w") as f:
        json.dump({"videos": all_results}, f, indent=2)
    print(f"\nSaved: {out_path}  ({len(all_results)} videos)")


if __name__ == "__main__":
    main()
