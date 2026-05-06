# Data Pipeline

This document describes how the public TexasPokerRobot data becomes batches for
policy training.

## Raw Dataset

The public data is hosted at:

```text
https://huggingface.co/datasets/Winniechen2002/TexasPokerRobot
```

Download with:

```bash
python workflow/download_data.py --local_dir data/TexasPokerRobot
```

Expected raw layout:

```text
data/TexasPokerRobot/
  pick_up_left/
    data_0001.npz
    data_0002.npz
  pick_up_right/
  push_5/
  push_10/
  push_50/
  push_100/
  pull_5/
  pull_10/
  pull_50/
  pull_100/
  put_down_left/
  put_down_right/
  show_left/
  show_right/
```

Each raw episode contains robot state arrays and up to three RGB/depth camera
streams.

## Instruction IDs

| ID | Operation | ID | Operation |
| --- | --- | --- | --- |
| 0 | `pick_up_left` | 7 | `pull_10` |
| 1 | `pick_up_right` | 8 | `pull_50` |
| 2 | `push_5` | 9 | `pull_100` |
| 3 | `push_10` | 10 | `put_down_left` |
| 4 | `push_50` | 11 | `put_down_right` |
| 5 | `push_100` | 12 | `show_left` |
| 6 | `pull_5` | 13 | `show_right` |

The source of truth is `workflow/instructions.json`.

## Organize Data

```bash
python workflow/organize_data.py \
  --source_dir data/TexasPokerRobot \
  --target_dir data/easy_mode \
  --eval_count 5
```

Output layout:

```text
data/easy_mode/
  instructions.json
  0/
    pick_up_card_train_<N>/
      data0001/
        images_cam0.npy
        depth_cam0.npy
        joint_positions.npy
        joint_efforts.npy
        joint_velocities.npy
    pick_up_card_test/
      data0001/
  ...
  13/
```

The organizer holds out the last `eval_count` episodes per task for testing and
explodes `.npz` files into `.npy` directories for faster lazy loading.

## Precompute Visual Features

Frozen-backbone models should precompute RGB features once.

DinoV2 CLS features for DP(DINO):

```bash
python workflow/precompute_features.py \
  --data_dir data/easy_mode \
  --feature_dir data/vitl14_features \
  --encoder dinov2_vitl14 \
  --gpu 0
```

SigLIP patch features for RDT_small and RDT_FT:

```bash
python workflow/precompute_features.py \
  --data_dir data/easy_mode \
  --feature_dir data/siglip_features \
  --encoder siglip_so400m \
  --gpu 0
```

Feature layout mirrors the data layout:

```text
data/vitl14_features/0/pick_up_card_train_<N>/data0001/
  rgb_features_cam0.npy
  rgb_features_cam1.npy
  rgb_features_cam2.npy
```

## Canonical Batch Format

The dataset and dataloader produce a dictionary with these keys when the
corresponding modalities are enabled:

| Key | Shape | Description |
| --- | --- | --- |
| `rgb` | `(B, T, cams, H, W, 3)` | RGB images, uint8 |
| `depth` | `(B, T, cams, H, W)` | depth images |
| `rgb_features` | `(B, T, cams, ...)` | precomputed visual features |
| `pos` | `(B, T, 30)` | full arm + hand joint positions |
| `eef` | `(B, T, 6)` | arm joints |
| `hand_pos` | `(B, T, 24)` | hand joints |
| `efforts` | `(B, T, 24)` | hand joint efforts |
| `velocity` | `(B, T, 30)` | joint velocities |
| `touch` | `(B, T, 60)` | tactile readings |
| `action` | `(B, pred_horizon, 30)` | target action chunk |
| `instruction` | `(B,)` | integer instruction ID |

Default temporal windows:

| Parameter | Value |
| --- | --- |
| `obs_horizon` | 1 for DP/ACT, 2 for RDT |
| `action_horizon` | 32 |
| `pred_horizon` | 64 |

## Loading and Normalization

`data_processing/loading.py` handles both source formats:

| Format | Example |
| --- | --- |
| `.npz` episode | `data0001.npz` |
| exploded `.npy` directory | `data0001/images_cam0.npy` |

The loader converts joint dictionaries to dense 30-d vectors using the fixed
UR arm + Shadow Hand joint order. Proprioception and actions are normalized to
`[-1, 1]` from training-set min/max statistics. Images, depth maps, and
precomputed visual features are not min-max normalized by the dataset.

Checkpoints save the normalization statistics, so deployment can reconstruct
the same preprocessing path.

## Combined Preparation Script

For the common full release setup:

```bash
bash scripts/prepare.sh \
  data/TexasPokerRobot data/easy_mode 5 4 \
  data/vitl14_features data/siglip_features
```

Arguments:

| Position | Meaning |
| --- | --- |
| 1 | raw source directory |
| 2 | organized target directory |
| 3 | held-out test episodes per task |
| 4 | number of GPUs for feature extraction |
| 5 | DinoV2 feature output directory, optional |
| 6 | SigLIP feature output directory, optional |
