"""
Training Script — GalarModel
================================
Trains on sliding-window feature sequences extracted by extract_features.py.
Best model is selected by validation temporal mAP@0.5.

Usage:
  python train.py --config configs/example.yaml
  python train.py --config configs/example.yaml --eval_interval 10
"""

import os
import csv
import time
import yaml
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from scipy.signal import medfilt
from tqdm import tqdm

from models.model import GalarModel
from data.dataset import (
    GalarWindowDataset, compute_pos_weights,
    make_weighted_sampler, stratified_video_split, preload_features,
)
from utils.losses import AsymmetricLoss
from utils.viterbi import viterbi_anatomy, smooth_anatomy_probs
from utils.cooccurrence import compute_or_load_cooccurrence, apply_cooccurrence_gating


ANATOMY_LABELS = [
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus", "ileocecal valve",
]
PATHOLOGY_LABELS = [
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer",
]

_ANATOMY_MODULES = {
    "gc_sim1", "gc_sim2", "gc_dis1", "gc_dis2", "anatomy_gcn_proj",
    "boundary_proj", "anatomy_head", "mamba_fwd", "mamba_bwd",
    "anat_input_proj", "anat_input_ln", "pos_embed", "motion_proj_cls",
    "local_blocks", "video_pos_proj",
}
_PATHOLOGY_MODULES = {
    "gc_p_sim", "gc_p_dis", "path_gcn_proj", "pathology_head",
    "recen_proj", "patch_proj", "patch_ln", "motion_proj_patch",
    "path_conv", "mamba_path", "fusion_proj", "anatomy_cond_proj",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Evaluation helpers ────────────────────────────────────────────────────────

def compute_frame_map(all_preds, all_labels):
    ap_list = []
    for i in range(17):
        y_true = all_labels[:, i]
        if y_true.sum() == 0:
            ap_list.append(None)
            continue
        try:
            ap_list.append(average_precision_score(y_true, all_preds[:, i]))
        except Exception:
            ap_list.append(None)
    valid = [x for x in ap_list if x is not None]
    return {
        "anatomy_mAP":   float(np.mean([x for x in ap_list[:8]  if x is not None] or [0])),
        "pathology_mAP": float(np.mean([x for x in ap_list[8:]  if x is not None] or [0])),
        "overall_mAP":   float(np.mean(valid) if valid else 0.0),
    }


@torch.no_grad()
def infer_video(model, features, config, device, co_matrix=None):
    N, D       = features.shape
    window     = config["data"]["window_size"]
    stride     = config["data"]["infer_stride"]
    threshold  = config["inference"]["threshold"]
    smooth_w   = config["inference"]["smooth_window"]
    co_thr     = config["inference"].get("cooccurrence_threshold", 0.01)
    co_mode    = config["inference"].get("cooccurrence_mode", "hard")

    pred_sum   = np.zeros((N, 17), dtype=np.float64)
    pred_count = np.zeros(N, dtype=np.int32)

    starts = list(range(0, max(1, N - window + 1), stride))
    if N > window and (N - window) % stride != 0:
        starts.append(N - window)
    if not starts:
        starts = [0]

    for s in starts:
        e    = min(s + window, N)
        chunk = features[s:e].astype(np.float32)
        T_a  = len(chunk)
        if T_a < window:
            chunk = np.vstack([chunk, np.zeros((window - T_a, D), dtype=np.float32)])
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
    return (pred_probs >= threshold).astype(np.int32), pred_probs


def _load_gt_segments(labels_dir, video_id, label_names):
    segments = {c: [] for c in range(len(label_names))}
    csv_path = os.path.join(labels_dir, f"{video_id}.csv")
    try:
        with open(csv_path) as f:
            rows = sorted(csv.DictReader(f), key=lambda r: int(r["frame"]))
    except FileNotFoundError:
        return segments
    for c, label in enumerate(label_names):
        if label not in rows[0]:
            continue
        in_seg, seg_start, prev = False, None, None
        for r in rows:
            active = r.get(label, "0").strip() == "1"
            frame  = int(r["frame"])
            if active and not in_seg:
                in_seg, seg_start = True, frame
            elif not active and in_seg and prev is not None:
                segments[c].append((seg_start, prev))
                in_seg = False
            prev = frame
        if in_seg and prev is not None:
            segments[c].append((seg_start, prev))
    return segments


def _frames_to_segments(frame_nums, pred_probs, threshold):
    segments = {c: [] for c in range(17)}
    binary   = (pred_probs >= threshold).astype(np.int32)
    for c in range(17):
        in_s, s_start, s_probs = False, None, []
        for i in range(len(frame_nums)):
            if binary[i, c]:
                if not in_s:
                    in_s, s_start, s_probs = True, int(frame_nums[i]), [pred_probs[i, c]]
                else:
                    s_probs.append(pred_probs[i, c])
            elif in_s:
                segments[c].append((s_start, int(frame_nums[i - 1]), float(np.mean(s_probs))))
                in_s = False
        if in_s:
            segments[c].append((s_start, int(frame_nums[-1]), float(np.mean(s_probs))))
    return segments


def _temporal_ap(pred_segs, gt_segs, iou_thr=0.5):
    if not gt_segs:
        return 0.0 if pred_segs else 1.0
    if not pred_segs:
        return 0.0
    matched, tp = set(), []
    for (ps, pe, _) in pred_segs:
        hit = False
        for j, (gs, ge) in enumerate(gt_segs):
            if j in matched:
                continue
            inter = max(0, min(pe, ge) - max(ps, gs) + 1)
            union = max(pe, ge) - min(ps, gs) + 1
            if inter / union >= iou_thr:
                matched.add(j); hit = True; break
        tp.append(1 if hit else 0)
    cum_tp, prev_r, ap = 0, 0.0, 0.0
    for i, v in enumerate(tp):
        cum_tp += v
        r = cum_tp / len(gt_segs)
        p = cum_tp / (i + 1)
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


@torch.no_grad()
def compute_temporal_map(model, val_ids, features_dir, labels_dir, config, device, co_matrix=None):
    model.eval()
    label_names = ANATOMY_LABELS + PATHOLOGY_LABELS
    threshold   = config["inference"]["threshold"]
    video_ap_50, video_ap_95 = [], []

    for vid_id in val_ids:
        feat_path  = os.path.join(features_dir, f"{vid_id}_features.npy")
        frame_path = os.path.join(features_dir, f"{vid_id}_frames.npy")
        if not os.path.exists(feat_path):
            continue
        features   = np.load(feat_path).astype(np.float32)
        frame_nums = np.load(frame_path).astype(np.int64)

        _, pred_probs = infer_video(model, features, config, device, co_matrix)
        gt_segs       = _load_gt_segments(labels_dir, vid_id, label_names)
        pred_segs     = _frames_to_segments(frame_nums, pred_probs, threshold)

        video_ap_50.append([_temporal_ap(pred_segs[c], gt_segs[c], 0.50) for c in range(17)])
        video_ap_95.append([_temporal_ap(pred_segs[c], gt_segs[c], 0.95) for c in range(17)])

    def agg(aps):
        if not aps:
            return {"mAP": 0.0, "anat": 0.0, "path": 0.0}
        arr = np.array(aps)
        return {
            "mAP":  float(arr.mean(axis=1).mean()),
            "anat": float(arr[:, :8].mean()),
            "path": float(arr[:, 8:].mean()),
        }

    r50, r95 = agg(video_ap_50), agg(video_ap_95)
    return {
        "tMAP_50": r50["mAP"], "anat_50": r50["anat"], "path_50": r50["path"],
        "tMAP_95": r95["mAP"], "anat_95": r95["anat"], "path_95": r95["path"],
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def temporal_label_smooth(labels, kernel_size):
    if kernel_size <= 1:
        return labels
    B, T, C = labels.shape
    half   = kernel_size // 2
    x      = torch.arange(-half, half + 1, dtype=torch.float32, device=labels.device)
    kernel = torch.exp(-x ** 2 / (2 * (half / 2.0) ** 2))
    kernel = kernel / kernel.sum()
    inp    = labels.permute(0, 2, 1).reshape(B * C, 1, T)
    out    = F.conv1d(inp, kernel.view(1, 1, -1), padding=half)
    return out.reshape(B, C, T).permute(0, 2, 1).clamp(0.0, 1.0)


def train_one_epoch(model, loader, optimizer, anatomy_loss_fn, pathology_loss_fn,
                    config, device, epoch, log_writer, path_boost=None):
    model.train()
    total_loss, steps = 0.0, 0
    t = config["training"]
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for batch in pbar:
        features        = batch["features"].to(device)
        labels          = batch["labels"].to(device)
        valid_len       = batch["valid_len"]
        video_pos_ratio = batch["video_pos_ratio"].to(device)

        anatomy_labels   = labels[:, :, :8]
        pathology_labels = labels[:, :, 8:]

        if t.get("label_smooth_window", 1) > 1:
            pathology_labels = temporal_label_smooth(pathology_labels, t["label_smooth_window"])

        # Temporal masking
        features_input = features
        if t.get("mask_ratio", 0) > 0:
            B, T_s, D = features.shape
            tmask = (torch.rand(B, T_s, device=device) < t["mask_ratio"]).unsqueeze(-1)
            noise = torch.randn(B, T_s, D, device=device) * 0.1
            features_input = torch.where(tmask.expand_as(features), noise, features)
        if t.get("feat_noise_std", 0) > 0:
            features_input = features_input + torch.randn_like(features_input) * t["feat_noise_std"]

        anatomy_logits, pathology_logits, x_a = model(
            features=features_input, raw_features=features,
            video_pos_ratio=video_pos_ratio, return_features=True,
        )

        B, T_s, _ = anatomy_logits.shape
        mask = torch.zeros(B, T_s, device=device)
        for i, vl in enumerate(valid_len):
            mask[i, :vl] = 1.0

        # Anatomy loss with boundary emphasis
        a_loss_raw   = anatomy_loss_fn(anatomy_logits, anatomy_labels)
        anat_diff    = (anatomy_labels[:, 1:] - anatomy_labels[:, :-1]).abs().any(dim=-1).float()
        boundary_w   = torch.zeros(B, T_s, device=device)
        boundary_w[:, 1:]  += anat_diff
        boundary_w[:, :-1] += anat_diff
        boundary_w   = F.max_pool1d(boundary_w.unsqueeze(1), 5, 1, 2).squeeze(1)
        boundary_w   = 1.0 + t.get("transition_loss_boost", 1.3) * boundary_w
        a_loss = (a_loss_raw * mask.unsqueeze(-1) * boundary_w.unsqueeze(-1)).sum() \
                 / (mask.sum() * 8 + 1e-8)

        # Pathology loss with boundary emphasis
        p_loss_raw    = pathology_loss_fn(pathology_logits, pathology_labels)
        if path_boost is not None:
            p_loss_raw = p_loss_raw * path_boost.view(1, 1, 9)
        path_diff     = (pathology_labels[:, 1:] - pathology_labels[:, :-1]).abs().any(dim=-1).float()
        path_bw       = torch.zeros(B, T_s, device=device)
        path_bw[:, 1:]  += path_diff
        path_bw[:, :-1] += path_diff
        path_bw       = F.max_pool1d(path_bw.unsqueeze(1), 5, 1, 2).squeeze(1)
        path_bw       = 1.0 + t.get("path_transition_loss_boost", 2.57) * path_bw
        p_loss = (p_loss_raw * mask.unsqueeze(-1) * path_bw.unsqueeze(-1)).sum() \
                 / (mask.sum() * 9 + 1e-8)

        # Smoothness loss (anatomy)
        smooth_w, smooth_loss = t.get("smooth_loss_weight", 0.0), torch.tensor(0.0, device=device)
        if smooth_w > 0:
            anat_p     = torch.sigmoid(anatomy_logits)
            pair_mask  = mask[:, :-1] * mask[:, 1:]
            cls_w      = torch.tensor([1., 2., 2., 2., 2., 0., 0., 0.], device=device)
            smooth_loss = (
                (anat_p[:, 1:] - anat_p[:, :-1]).abs() * pair_mask.unsqueeze(-1) * cls_w.view(1, 1, 8)
            ).sum() / (pair_mask.sum() * cls_w.sum() + 1e-8)

        # Anatomy cluster loss
        cluster_w, cluster_loss = t.get("anatomy_cluster_loss_weight", 0.0), torch.tensor(0.0, device=device)
        if cluster_w > 0 and model.anatomy_proto_initialized:
            n_xa  = F.normalize(x_a, dim=-1)
            n_pr  = F.normalize(model.anatomy_prototypes.detach(), dim=-1)
            sim   = n_xa @ n_pr.T
            no_p  = (pathology_labels.sum(-1) == 0).float()
            has_l = (anatomy_labels.sum(-1) > 0).float() * mask * no_p
            cluster_loss = ((1.0 - (sim * anatomy_labels).sum(-1)) * has_l).sum() / (has_l.sum() + 1e-8)

        # Monotonicity loss
        mono_w, mono_loss = t.get("mono_loss_weight", 0.0), torch.tensor(0.0, device=device)
        if mono_w > 0:
            order    = torch.tensor([0., 1., 3., 5., 7., 2., 4., 6.], device=device)
            anat_p   = torch.sigmoid(anatomy_logits)
            exp_pos  = (anat_p * order.view(1, 1, 8)).sum(-1)
            pair_m   = mask[:, :-1] * mask[:, 1:]
            mono_loss = (F.relu(exp_pos[:, :-1] - exp_pos[:, 1:]) * pair_m).sum() \
                        / (pair_m.sum() + 1e-8)

        loss = (t["anatomy_loss_weight"] * a_loss
                + t["pathology_loss_weight"] * p_loss
                + smooth_w * smooth_loss
                + cluster_w * cluster_loss
                + mono_w * mono_loss)

        if not torch.isfinite(loss):
            optimizer.zero_grad(); continue

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        if any(not torch.isfinite(p.grad).all()
               for p in model.parameters() if p.grad is not None):
            optimizer.zero_grad(); continue
        optimizer.step()

        model.update_normal_patch_prototypes(
            patch_features=features[:, :, model.cls_dim:].detach(),
            anatomy_labels=anatomy_labels.detach(),
            pathology_labels=pathology_labels.detach(),
            momentum=t.get("anatomy_proto_momentum", 0.99),
        )
        model.update_anatomy_prototypes(
            x_a=x_a.detach(),
            anatomy_labels=anatomy_labels.detach(),
            pathology_labels=pathology_labels.detach(),
            momentum=t.get("anatomy_proto_momentum", 0.99),
        )

        total_loss += loss.item(); steps += 1
        if steps % t["log_interval"] == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "gn": f"{grad_norm:.1f}"})
            log_writer.writerow([epoch, steps, f"{loss.item():.6f}",
                                  f"{a_loss.item():.6f}", f"{p_loss.item():.6f}", f"{grad_norm:.4f}"])

    return total_loss / max(steps, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="  [val]", leave=False):
        features        = batch["features"].to(device)
        valid_len       = batch["valid_len"]
        video_pos_ratio = batch["video_pos_ratio"].to(device)
        a_l, p_l = model(features=features, raw_features=features, video_pos_ratio=video_pos_ratio)
        preds = torch.cat([torch.sigmoid(a_l), torch.sigmoid(p_l)], dim=-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        for i, vl in enumerate(valid_len):
            all_preds.append(preds[i, :vl])
            all_labels.append(labels[i, :vl])
    return compute_frame_map(np.vstack(all_preds), np.vstack(all_labels))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        type=str, default="configs/example.yaml")
    parser.add_argument("--eval_interval", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labels_dir   = config["data"]["labels_dir"]
    features_dir = config["data"]["features_dir"]

    all_ids = sorted([p.stem for p in Path(labels_dir).glob("*.csv")])
    val_ratio = config["training"]["val_ratio"]
    if val_ratio > 0:
        train_ids, val_ids = stratified_video_split(all_ids, labels_dir, val_ratio,
                                                     config["training"]["seed"])
    else:
        train_ids, val_ids = all_ids, []
    print(f"Train: {len(train_ids)} videos  |  Val: {len(val_ids)} videos")

    print("Preloading features...", flush=True)
    preloaded = preload_features(all_ids, features_dir, labels_dir)
    print(f"Loaded {len(preloaded)} videos", flush=True)

    ws = config["data"]["window_size"]
    st = config["data"]["stride"]
    train_ds = GalarWindowDataset(train_ids, features_dir, labels_dir, ws, st, preloaded)
    val_ds   = GalarWindowDataset(val_ids,   features_dir, labels_dir, ws, st, preloaded)
    print(f"Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")

    nw = config["training"]["num_workers"]
    bs = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=make_weighted_sampler(train_ds),
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    model = GalarModel(config).to(device)

    warmstart = config["training"].get("warmstart_checkpoint", "")
    if warmstart and os.path.exists(warmstart):
        ws_ckpt = torch.load(warmstart, map_location=device)
        model.load_state_dict(ws_ckpt["model_state_dict"], strict=False)
        model.anatomy_proto_initialized = ws_ckpt.get("anatomy_proto_initialized", False)
        model.normal_proto_initialized  = ws_ckpt.get("normal_proto_initialized", False)
        print(f"Warm-started from: {warmstart}")

    # Loss functions
    pos_weights = compute_pos_weights(train_ds).to(device)
    rare_boost  = torch.ones(8, device=device)
    t = config["training"]
    rare_boost[0] = t.get("mouth_boost",     4.0)
    rare_boost[1] = t.get("esoph_boost",     4.0)
    rare_boost[5] = t.get("zline_boost",     4.0)
    rare_boost[6] = t.get("pylorus_boost",   4.0)
    rare_boost[7] = t.get("ileocecal_boost", 4.0)
    anatomy_loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weights[:8] * rare_boost, reduction="none")
    pathology_loss_fn = AsymmetricLoss(
        gamma_pos=t.get("asl_gamma_pos", 1.0),
        gamma_neg=t.get("asl_gamma_neg", 4.0),
        clip=t.get("asl_clip", 0.05),
    )
    path_boost = torch.ones(9, device=device)
    path_boost[3] = t.get("erosion_boost", 1.0)
    path_boost[2] = t.get("blood_boost",   1.0)
    path_boost[1] = t.get("angio_boost",   1.0)

    # Optimizer with per-branch LR
    base_lr = t["lr"]
    anat_params, path_params, shared_params = [], [], []
    for name, param in model.named_parameters():
        m = name.split(".")[0]
        if m in _ANATOMY_MODULES:
            anat_params.append(param)
        elif m in _PATHOLOGY_MODULES:
            path_params.append(param)
        else:
            shared_params.append(param)
    optimizer = torch.optim.AdamW([
        {"params": shared_params, "lr": base_lr},
        {"params": anat_params,   "lr": base_lr * t.get("lr_anatomy_mul",   0.5)},
        {"params": path_params,   "lr": base_lr * t.get("lr_pathology_mul", 2.0)},
    ], weight_decay=t["weight_decay"])

    warmup_ep = t.get("warmup_epochs", 5)
    n_epochs  = t["num_epochs"]
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
        torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, warmup_ep),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(n_epochs - warmup_ep, 1)),
    ], milestones=[warmup_ep])

    # Logging
    save_dir = t["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    log_f  = open("./logs/train_log.csv", "w", newline="")
    log_w  = csv.writer(log_f)
    log_w.writerow(["epoch", "step", "loss", "anatomy_loss", "pathology_loss", "grad_norm"])
    val_f  = open("./logs/val_log.csv", "w", newline="")
    val_w  = csv.writer(val_f)
    val_w.writerow(["epoch", "frame_mAP", "anatomy_mAP", "pathology_mAP",
                     "tMAP_50", "anat_50", "path_50", "tMAP_95", "anat_95", "path_95", "lr"])

    eval_interval = args.eval_interval or t.get("eval_interval", 5)
    use_co = config["inference"].get("use_cooccurrence_gating", False)
    co_matrix = None
    if use_co:
        co_matrix = compute_or_load_cooccurrence(
            config["inference"].get("cooccurrence_labels_dir", labels_dir),
            config["inference"].get("cooccurrence_cache", "./utils/cooccurrence_matrix.npy"),
        )

    best_tmap, best_epoch, ep_times = 0.0, 0, []

    for epoch in range(1, n_epochs + 1):
        t0       = time.time()
        avg_loss = train_one_epoch(model, train_loader, optimizer,
                                    anatomy_loss_fn, pathology_loss_fn,
                                    config, device, epoch, log_w, path_boost=path_boost)
        scheduler.step()
        ep_times.append(time.time() - t0)
        lr     = optimizer.param_groups[0]["lr"]
        eta    = np.mean(ep_times[-10:]) * (n_epochs - epoch)
        eta_s  = f"{eta/3600:.1f}h" if eta >= 3600 else f"{eta/60:.1f}min"

        frame_m = validate(model, val_loader, device) if val_ids else {"overall_mAP": 0, "anatomy_mAP": 0, "pathology_mAP": 0}
        print(f"  ep {epoch:3d}/{n_epochs} | loss={avg_loss:.4f} | lr={lr:.2e} | "
              f"frame_mAP={frame_m['overall_mAP']:.4f} | ETA {eta_s}", flush=True)

        do_temporal = val_ids and (epoch % eval_interval == 0 or epoch == n_epochs)
        if do_temporal:
            tm = compute_temporal_map(model, val_ids, features_dir, labels_dir, config, device, co_matrix)
            obj = (tm["tMAP_50"] + tm["tMAP_95"]) / 2
            mark = "★" if obj > best_tmap else " "
            print(f"  {mark} tMAP@0.50={tm['tMAP_50']:.4f} (A={tm['anat_50']:.4f} P={tm['path_50']:.4f})"
                  f"  tMAP@0.95={tm['tMAP_95']:.4f}  best={best_tmap:.4f}", flush=True)
            val_w.writerow([epoch, frame_m["overall_mAP"], frame_m["anatomy_mAP"], frame_m["pathology_mAP"],
                             tm["tMAP_50"], tm["anat_50"], tm["path_50"],
                             tm["tMAP_95"], tm["anat_95"], tm["path_95"], lr])
            val_f.flush()
            if obj > best_tmap:
                best_tmap, best_epoch = obj, epoch
                torch.save({
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_tmap": obj, "config": config,
                    "anatomy_proto_initialized": model.anatomy_proto_initialized,
                    "normal_proto_initialized":  model.normal_proto_initialized,
                }, os.path.join(save_dir, "best_model.pth"))
                print(f"    ✓ Best model saved (epoch={epoch}, tMAP@0.50={tm['tMAP_50']:.4f})")
        elif not val_ids:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "anatomy_proto_initialized": model.anatomy_proto_initialized,
                "normal_proto_initialized":  model.normal_proto_initialized,
            }, os.path.join(save_dir, "best_model.pth"))
        log_f.flush()

    log_f.close(); val_f.close()
    print(f"\nDone! Best epoch={best_epoch}, best val tMAP={best_tmap:.4f}")


if __name__ == "__main__":
    main()
