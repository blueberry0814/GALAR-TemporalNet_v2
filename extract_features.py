"""
Feature Extraction for GI Endoscopy Videos
============================================
Extracts per-frame features using DINOv2 (recommended) or DINOv3.
Each frame produces a 2048-d vector: [CLS token (1024) | patch mean (1024)].

Output files per video:
  {output_dir}/{video_id}_features.npy   [N_frames, 2048]  float32
  {output_dir}/{video_id}_frames.npy     [N_frames]        int64

Usage:
  # Training data
  python extract_features.py \\
      --model dinov2-vitl14 \\
      --labels_dir ./dataset/Labels \\
      --frames_root ./dataset \\
      --output_dir ./features

  # Test data
  python extract_features.py \\
      --model dinov2-vitl14 \\
      --labels_dir ./test/Labels \\
      --frames_root ./test \\
      --output_dir ./features_test \\
      --test_mode
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm


MODEL_CONFIGS = {
    "dinov2-vitl14": {
        "family": "dinov2", "hub_name": "dinov2_vitl14",
        "embed_dim": 1024, "feat_dim": 2048, "image_size": 518,
        "desc": "DINOv2 ViT-L/14 — recommended (no auth required)",
    },
    "dinov2-vitb14": {
        "family": "dinov2", "hub_name": "dinov2_vitb14",
        "embed_dim": 768, "feat_dim": 1536, "image_size": 518,
        "desc": "DINOv2 ViT-B/14 — faster, lower dim",
    },
    "dinov3-vitl16": {
        "family": "dinov3", "embed_dim": 1024, "feat_dim": 2048,
        "timm_names": [
            "vit_large_patch16_dinov3.lvd1689m",
            "vit_large_patch16_dinov3.sat493m",
            "vit_large_patch16_224.dinov3",
        ],
        "hf_names": ["facebook/dinov3-vitl16-pretrain-lvd1689m", "facebook/dinov3-large"],
        "desc": "DINOv3 ViT-L/16 (requires timm>=1.0.20 or HF token)",
    },
}


def load_model(model_key: str, device: torch.device):
    """Returns (extract_fn, embed_dim). extract_fn(imgs) -> np.ndarray [B, 2*embed_dim]"""
    cfg = MODEL_CONFIGS[model_key]
    if cfg["family"] == "dinov2":
        return _load_dinov2(cfg, device)
    return _load_dinov3(cfg, device)


def _load_dinov2(cfg, device):
    hub_name  = cfg["hub_name"]
    embed_dim = cfg["embed_dim"]
    img_size  = cfg["image_size"]

    print(f"  Loading via torch.hub: facebookresearch/dinov2 / {hub_name}")
    model = torch.hub.load("facebookresearch/dinov2", hub_name, verbose=False)
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    @torch.no_grad()
    def extract(imgs):
        tensor = torch.stack([transform(img) for img in imgs]).to(device)
        out    = model.forward_features(tensor)
        cls    = out["x_norm_clstoken"]                 # [B, D]
        patch  = out["x_norm_patchtokens"].mean(dim=1)  # [B, D]
        return torch.cat([cls, patch], dim=-1).cpu().float().numpy()

    print(f"  Loaded {hub_name} (embed_dim={embed_dim}, feat_dim={embed_dim * 2})")
    return extract, embed_dim


def _load_dinov3(cfg, device):
    embed_dim = cfg["embed_dim"]

    for name in cfg.get("timm_names", []):
        try:
            import timm as _timm
            print(f"  Trying timm: {name}")
            m         = _timm.create_model(name, pretrained=True, num_classes=0)
            m.eval().to(device)
            data_cfg  = _timm.data.resolve_model_data_config(m)
            transform = _timm.data.create_transform(**data_cfg, is_training=False)
            n_prefix  = getattr(m, "num_prefix_tokens", 1)
            actual    = m.num_features

            @torch.no_grad()
            def extract_timm(imgs, _m=m, _t=transform, _np=n_prefix):
                tensor = torch.stack([_t(img) for img in imgs]).to(device)
                feats  = _m.forward_features(tensor)
                cls    = feats[:, 0, :]
                patch  = feats[:, _np:, :].mean(dim=1)
                return torch.cat([cls, patch], dim=-1).cpu().float().numpy()

            print(f"  Loaded {name} (embed_dim={actual})")
            return extract_timm, actual
        except Exception as e:
            print(f"  Failed {name}: {e}")

    for name in cfg.get("hf_names", []):
        try:
            from transformers import AutoImageProcessor, AutoModel
            hf_token  = os.environ.get("HF_TOKEN", None)
            print(f"  Trying HuggingFace: {name}")
            processor = AutoImageProcessor.from_pretrained(name, token=hf_token)
            hf_model  = AutoModel.from_pretrained(name, torch_dtype=torch.float16, token=hf_token)
            hf_model.eval().to(device)
            num_reg   = getattr(hf_model.config, "num_register_tokens", 0)

            @torch.no_grad()
            def extract_hf(imgs, _m=hf_model, _p=processor, _nr=num_reg):
                inputs = _p(images=imgs, return_tensors="pt").to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                    last = _m(**inputs).last_hidden_state.float()
                cls   = last[:, 0, :]
                patch = last[:, 1 + _nr:, :].mean(dim=1)
                return torch.cat([cls, patch], dim=-1).cpu().numpy()

            print(f"  Loaded HF {name}")
            return extract_hf, hf_model.config.hidden_size
        except Exception as e:
            print(f"  Failed {name}: {e}")

    raise RuntimeError(
        "DINOv3 load failed. Use --model dinov2-vitl14 (no auth needed), "
        "or install timm>=1.0.20, or set HF_TOKEN env variable."
    )


def find_frame_path(frames_root, video_id, frame_num):
    for pat in [
        f"{frames_root}/{video_id}/frame_{frame_num:06d}.PNG",
        f"{frames_root}/{video_id}/frame_{frame_num:07d}.png",
    ]:
        if os.path.exists(pat):
            return pat
    for d in sorted(Path(frames_root).glob("Galar_Frames_*")):
        p = d / f"recording_{video_id}" / f"frame_{frame_num:06d}.PNG"
        if p.exists():
            return str(p)
    return None


def extract_video(video_id, label_path, frames_root, output_dir, extract_fn, batch_size, test_mode):
    feat_out  = os.path.join(output_dir, f"{video_id}_features.npy")
    frame_out = os.path.join(output_dir, f"{video_id}_frames.npy")

    if os.path.exists(feat_out) and os.path.exists(frame_out):
        print(f"[SKIP] {video_id}: already extracted")
        return

    if test_mode:
        df = pd.read_csv(label_path, sep=";")
        df["frame"] = df["frame_file"].str.extract(r"(\d+)").astype(np.int64)
        df = df.sort_values("frame").reset_index(drop=True)
    else:
        df = pd.read_csv(label_path).sort_values("frame").reset_index(drop=True)
    frame_nums = df["frame"].values.astype(np.int64)

    all_feats, valid_frames = [], []
    batch_imgs, batch_fnums = [], []

    for fn in tqdm(frame_nums, desc=f"  [{video_id}]", leave=False):
        path = find_frame_path(frames_root, video_id, int(fn))
        if path is None:
            continue
        try:
            batch_imgs.append(Image.open(path).convert("RGB"))
            batch_fnums.append(fn)
        except Exception:
            continue

        if len(batch_imgs) == batch_size:
            all_feats.append(extract_fn(batch_imgs))
            valid_frames.extend(batch_fnums)
            batch_imgs.clear(); batch_fnums.clear()

    if batch_imgs:
        all_feats.append(extract_fn(batch_imgs))
        valid_frames.extend(batch_fnums)

    if not all_feats:
        print(f"  [WARNING] {video_id}: no valid frames found")
        return

    feats_arr = np.vstack(all_feats).astype(np.float32)
    frame_arr = np.array(valid_frames, dtype=np.int64)
    np.save(feat_out, feats_arr)
    np.save(frame_out, frame_arr)
    print(f"  Saved {video_id}: frames={len(frame_arr)}, shape={feats_arr.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="dinov2-vitl14",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--labels_dir",  type=str, required=True,
                        help="Directory containing per-video CSV label files")
    parser.add_argument("--frames_root", type=str, required=True,
                        help="Root directory of frame images")
    parser.add_argument("--output_dir",  type=str, default="./features")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--video_ids",   type=str, default=None,
                        help="Comma-separated video IDs. Default: all CSVs in labels_dir")
    parser.add_argument("--test_mode",   action="store_true",
                        help="Test data mode: semicolon-delimited CSV with frame_file column")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = MODEL_CONFIGS[args.model]
    print(f"\nModel: {args.model}  ({cfg['desc']})")
    print(f"feat_dim: {cfg['feat_dim']}")

    extract_fn, embed_dim = load_model(args.model, device)
    print(f"Actual feat_dim: {embed_dim * 2}  (CLS={embed_dim} + patch_mean={embed_dim})\n")

    labels_path = Path(args.labels_dir)
    video_ids = (
        [v.strip() for v in args.video_ids.split(",")]
        if args.video_ids
        else sorted([p.stem for p in labels_path.glob("*.csv")])
    )
    print(f"Processing {len(video_ids)} videos...\n")

    for vid_id in video_ids:
        label_path = labels_path / f"{vid_id}.csv"
        if not label_path.exists():
            print(f"[SKIP] {vid_id}: no label file")
            continue
        extract_video(
            video_id=vid_id, label_path=str(label_path),
            frames_root=args.frames_root, output_dir=args.output_dir,
            extract_fn=extract_fn, batch_size=args.batch_size,
            test_mode=args.test_mode,
        )

    print(f"\nDone! Features saved to: {args.output_dir}/")
    print(f"\nUpdate your config:")
    print(f"  data.features_dir: {args.output_dir}")
    print(f"  model.feat_dim:    {embed_dim * 2}")


if __name__ == "__main__":
    main()
