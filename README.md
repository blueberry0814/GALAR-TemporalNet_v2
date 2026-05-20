# GALAR TemporalNet v2

Temporal event detection in GI endoscopy videos — ICPR 2026 RARE-VISION Challenge.

This repository implements **GalarModel v2**, a dual-branch sequence model that jointly
classifies anatomy sections (8 classes) and pathology events (9 classes) in capsule
endoscopy videos using pre-extracted DINOv2/DINOv3 features.

---

## Architecture Overview

```

<img width="4230" height="2484" alt="icpr_data_figure" src="https://github.com/user-attachments/assets/5bda37cd-a5f1-470d-b6fe-ff4c20b62af0" />

<img width="4389" height="2379" alt="icpr_figure_v3" src="https://github.com/user-attachments/assets/31216f79-67b9-4d80-aa78-318e98b62f4b" />
```

---

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** `mamba-ssm` requires CUDA. Install from source if the pip version fails:
> ```bash
> pip install mamba-ssm --no-build-isolation
> ```

---

## Dataset

| Split | Source | Link |
|---|---|---|
| Training | Figshare (GALAR dataset) | [Download](https://plus.figshare.com/articles/dataset/Galar_-_a_large_multi-label_video_capsule_endoscopy_dataset/25304616) |
| Test | Google Drive (ICPR 2026 RARE) | [Download](https://drive.google.com/drive/folders/17rGcIlR9QEXVJstOP57NKnuYIyW-UuYk) |

> **Evaluation:** Submit your `predictions.json` to the official scoring server at
> [https://scoringrarevision.streamlit.app/](https://scoringrarevision.streamlit.app/)
> to get the official temporal mAP score.

---

## Data Structure

Download the datasets above and place them anywhere on your machine.
The code only needs the paths — point the config or CLI args at wherever you put the data.

### Training data

```
dataset/
├── Labels/
│   ├── 1.csv
│   ├── 2.csv
│   └── ...         (one CSV per video, named {video_id}.csv)
├── 1/
│   ├── frame_000100.PNG
│   ├── frame_000105.PNG
│   └── ...
├── 2/
│   └── ...
└── ...
```

Training CSV format — comma-delimited, with a `frame` column (integer frame number):

```
index,z-line,pylorus,ileocecal valve,...,mouth,esophagus,stomach,...,frame
20,0,0,0,...,1,0,0,...,100
21,0,0,0,...,1,0,0,...,105
```

### Test data

```
test/
├── Labels/
│   ├── ukdd_navi_00051.csv
│   ├── ukdd_navi_00068.csv
│   └── ...
├── ukdd_navi_00051/
│   ├── frame_0000049.png
│   └── ...
└── ...
```

Test CSV format — **semicolon-delimited**, with a `frame_file` column (filename string).
Use `--test_mode` flag when extracting features from test data:

```
frame_file;mouth;esophagus;stomach;...
frame_0000049.png;;;;;...
```

### Pre-extracted features (output of `extract_features.py`)

```
features/                           ← training features
├── 1_features.npy    [N_frames, 2048]  float32
├── 1_frames.npy      [N_frames]        int64 (frame numbers)
├── 2_features.npy
└── ...

features_test/                      ← test features
├── ukdd_navi_00051_features.npy
├── ukdd_navi_00051_frames.npy
└── ...
```

---

## Quick Start

### Step 1 — Extract Features

```bash
# Training data
python extract_features.py \
    --model       dinov2-vitl14 \
    --labels_dir  ./dataset/Labels \
    --frames_root ./dataset \
    --output_dir  ./features \
    --batch_size  32

# Test data  (note the --test_mode flag for semicolon-delimited CSVs)
python extract_features.py \
    --model       dinov2-vitl14 \
    --labels_dir  ./test/Labels \
    --frames_root ./test \
    --output_dir  ./features_test \
    --batch_size  32 \
    --test_mode
```

### Step 2 — Configure

Edit `configs/example.yaml` to match your paths:

```yaml
data:
  features_dir: ./features          # directory from Step 1
  labels_dir:   ./dataset/Labels    # training Labels directory

training:
  save_dir: ./checkpoints
  num_epochs: 140
  ...
```

### Step 3 — Train

```bash
python train.py --config configs/example.yaml
```

Outputs:
```
checkpoints/best_model.pth      — best val tMAP checkpoint
logs/train_log.csv              — per-step loss log
logs/val_log.csv                — per-epoch validation metrics
```

### Step 4 — Predict

```bash
python predict.py \
    --config     configs/example.yaml \
    --checkpoint checkpoints/best_model.pth \
    --feat_dir   ./features_test \
    --labels_dir ./test/Labels \
    --output_dir ./results \
    --min_seg_frames 20 \
    --fill_gap
```

Output:
```
results/predictions.json        — combined submission file
results/{video_id}.json         — per-video results
```

---

## File Structure

```
GALAR_TemporalNet_v2/
├── extract_features.py         Feature extraction (DINOv2 / DINOv3)
├── train.py                    Training with temporal mAP evaluation
├── predict.py                  Inference and JSON submission generation
├── requirements.txt
├── configs/
│   └── example.yaml            Config template with tuned defaults
├── models/
│   └── model.py                GalarModel architecture
├── data/
│   └── dataset.py              Sliding-window dataset + stratified split
└── utils/
    ├── layers.py                GraphConvolution, DistanceAdj
    ├── losses.py                AsymmetricLoss, FocalLoss
    ├── viterbi.py               Anatomy sequence constraint (Viterbi)
    ├── gap_fill.py              Anatomy gap filling
    ├── cooccurrence.py          Co-occurrence gating
    ├── postprocess.py           Anatomy-pathology soft gating, per-class thresholds
    ├── make_json.py             Label name constants and JSON builder
    └── cooccurrence_matrix.npy  Pre-computed training co-occurrence statistics
```

---



## Results

Evaluated on the ICPR 2026 RARE-VISION Challenge test set (3 videos) via the
[official scoring server](https://scoringrarevision.streamlit.app/).

| Metric | ukdd_navi_00051 | ukdd_navi_00068 | ukdd_navi_00076 | **Average** |
|---|---|---|---|---|
| Overall mAP @ 0.5  | 0.4782 | 0.1912 | 0.3533 | **0.3409** |
| Overall mAP @ 0.95 | 0.4706 | 0.1765 | 0.3529 | **0.3333** |

---

## Citation

If you use this code, please cite:
```
@misc{galartemporalnet2026,
  title  = {GALAR TemporalNet v2},
  year   = {2026},
}
```
