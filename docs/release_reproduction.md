# Public Release Reproduction Guide

This guide reproduces the six public DexHoldem policy recipes:
DP(DINO), DP_transformer_resnet, DP_unet, ACT, RDT_small, and RDT_FT.

## 1. Environment

```bash
conda create -n texas python=3.10 -y
conda activate texas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

On the Shanghaiblue2 cluster, activate the project virtualenv instead:

```bash
source .venv/bin/activate
```

## 2. Download Data

The public data repo is:

```text
https://huggingface.co/datasets/Winniechen2002/TexasPokerRobot
```

Download the full dataset:

```bash
python workflow/download_data.py --local_dir data/TexasPokerRobot
```

The dataset is large. To verify the pipeline before a full download:

```bash
python workflow/download_data.py \
  --local_dir data/TexasPokerRobot_subset \
  --include "pick_up_left/**" "pick_up_right/**" "README.md"
```

## 3. Prepare Training Layout

Raw Hugging Face download layout:

```text
data/TexasPokerRobot/
  pick_up_left/
    data_0001.npz
    data_0002.npz
  pick_up_right/
  push_5/
  ...
```

Organize into multitask `easy_mode`:

```bash
python workflow/organize_data.py \
  --source_dir data/TexasPokerRobot \
  --target_dir data/easy_mode \
  --eval_count 5
```

Output:

```text
data/easy_mode/
  instructions.json
  0/pick_up_card_train_<N>/data0001/
  0/pick_up_card_test/data0001/
  ...
  13/pick_up_card_train_<N>/data0001/
```

Each `data0001/` directory contains `.npy` arrays exploded from the source
`.npz`, which enables memory-mapped lazy loading.

## 4. Precompute Features

DP(DINO) uses DinoV2 ViT-L/14 CLS features:

```bash
python workflow/precompute_features.py \
  --data_dir data/easy_mode \
  --feature_dir data/vitl14_features \
  --encoder dinov2_vitl14 \
  --gpu 0
```

RDT_small and RDT_FT use SigLIP-SO400M patch tokens:

```bash
python workflow/precompute_features.py \
  --data_dir data/easy_mode \
  --feature_dir data/siglip_features \
  --encoder siglip_so400m \
  --gpu 0
```

DP_unet and ACT do not need precomputed features.

For multi-GPU preparation, use:

```bash
bash scripts/prepare.sh \
  data/TexasPokerRobot data/easy_mode 5 4 \
  data/vitl14_features data/siglip_features
```

## 5. Train Release Models

```bash
# DP(DINO)
bash scripts/train_dp.sh data/easy_mode checkpoints/dp_dino 0 data/vitl14_features

# DP_transformer_resnet
bash scripts/train_dp_transformer.sh data/easy_mode checkpoints/dp_transformer_resnet 0

# DP_unet
bash scripts/train_dp_unet.sh data/easy_mode checkpoints/dp_unet 1

# ACT
bash scripts/train_act.sh data/easy_mode checkpoints/act 2

# RDT_small
bash scripts/train_rdt_small.sh data/easy_mode checkpoints/rdt_small 3 data/siglip_features
```

For RDT_FT, download the official RDT-1B checkpoint first:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('robotics-diffusion-transformer/rdt-1b', local_dir='checkpoints/rdt-1b-pretrained')"
```

Then finetune:

```bash
bash scripts/finetune_rdt.sh \
  data/easy_mode checkpoints/rdt_ft 3 \
  checkpoints/rdt-1b-pretrained data/siglip_features
```

## 6. Useful Overrides

All release scripts support common environment overrides:

```bash
BATCH_SIZE=64 EPOCHS=50 LR=5e-5 NUM_WORKERS=16 \
  bash scripts/train_dp_unet.sh data/easy_mode checkpoints/dp_unet_debug 0
```

Resume:

```bash
RESUME=checkpoints/dp_unet/latest.pt \
  bash scripts/train_dp_unet.sh data/easy_mode checkpoints/dp_unet 0
```

Enable W&B logging:

```bash
USE_WANDB=1 WANDB_PROJECT=DexHoldem WANDB_ENTITY=<entity> \
  bash scripts/train_act.sh data/easy_mode checkpoints/act 0
```

## 7. Verification

Fast local checks:

```bash
python train.py --list_models
python -m py_compile train.py workflow/download_data.py workflow/precompute_features.py
python test_code/test_training.py --run dp_pos_only
python test_code/test_training.py --run act_pos_only
```

Feature-based smoke checks:

```bash
python test_code/test_training.py --run dp_dinov2_transformer
python test_code/test_training.py --run rdt_precomputed_multitask
```

The RDT checks instantiate T5 and SigLIP code paths. They are slower and may
download model weights on first run.

## 8. Cluster Notes

Training and feature precomputation should run on GPU compute nodes. On
Shanghaiblue2, use `sbatch` and activate `.venv` inside the job script:

```bash
#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /public/home/winniechen/TexasPoker
source .venv/bin/activate
mkdir -p logs

bash scripts/train_dp.sh data/easy_mode checkpoints/dp_dino 0 data/vitl14_features
```
