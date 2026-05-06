# DexHoldem Policy

Imitation learning policies for a Shadow Hand + UR arm robot performing Texas
poker manipulation tasks. The public release focuses on six reproducible
training recipes:

| Release name | Script | Model code | Visual input |
| --- | --- | --- | --- |
| DP(DINO) | `scripts/train_dp.sh` | `learning/dp/` | DinoV2 ViT-L/14 CLS features |
| DP_transformer_resnet | `scripts/train_dp_transformer.sh` | `learning/dp/` | ResNet18 RGB by default |
| DP_unet | `scripts/train_dp_unet.sh` | `learning/dp/` | ResNet18 RGBD |
| ACT | `scripts/train_act.sh` | `learning/act/` | ResNet18 RGBD |
| RDT_small | `scripts/train_rdt_small.sh` | `learning/rdt/` | SigLIP-SO400M patch features |
| RDT_FT | `scripts/finetune_rdt.sh` | `learning/rdt/` | SigLIP-SO400M patch features + RDT-1B init |

The dataset is hosted at
[Winniechen2002/TexasPokerRobot](https://huggingface.co/datasets/Winniechen2002/TexasPokerRobot).

## Setup

```bash
conda create -n texas python=3.10 -y
conda activate texas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If you use a virtualenv instead of conda, activate it before running scripts.
The scripts use the active virtualenv first; otherwise they try `conda activate texas`.

## Data

Download the public dataset:

```bash
python workflow/download_data.py --local_dir data/TexasPokerRobot
```

For a small plumbing check, download only a subset:

```bash
python workflow/download_data.py \
  --local_dir data/TexasPokerRobot_subset \
  --include "pick_up_left/**" "pick_up_right/**" "README.md"
```

Prepare the training layout:

```bash
python workflow/organize_data.py \
  --source_dir data/TexasPokerRobot \
  --target_dir data/easy_mode \
  --eval_count 5
```

Precompute visual features for frozen-backbone models:

```bash
# DP(DINO)
python workflow/precompute_features.py \
  --data_dir data/easy_mode \
  --feature_dir data/vitl14_features \
  --encoder dinov2_vitl14 \
  --gpu 0

# RDT_small and RDT_FT
python workflow/precompute_features.py \
  --data_dir data/easy_mode \
  --feature_dir data/siglip_features \
  --encoder siglip_so400m \
  --gpu 0
```

You can also run the combined organizer and feature extraction pipeline:

```bash
bash scripts/prepare.sh \
  data/TexasPokerRobot data/easy_mode 5 4 \
  data/vitl14_features data/siglip_features
```

## Training

All scripts accept:

```bash
bash scripts/<name>.sh DATA_PATH SAVE_PATH GPU [FEATURE_DIR_OR_PRETRAINED]
```

Run the six public recipes:

```bash
# DP(DINO): DinoV2 CLS + high-capacity diffusion transformer
bash scripts/train_dp.sh data/easy_mode checkpoints/dp_dino 0 data/vitl14_features

# DP_transformer_resnet: ResNet18 RGB + configurable diffusion transformer baseline
bash scripts/train_dp_transformer.sh data/easy_mode checkpoints/dp_transformer_resnet 0

# DP_unet: ResNet18 RGBD + conditional 1D UNet
bash scripts/train_dp_unet.sh data/easy_mode checkpoints/dp_unet 1

# ACT: ResNet18 RGBD + CVAE transformer
bash scripts/train_act.sh data/easy_mode checkpoints/act 2

# RDT_small: 170M RDT ablation from scratch
bash scripts/train_rdt_small.sh data/easy_mode checkpoints/rdt_small 3 data/siglip_features

# RDT_FT: finetune from official RDT-1B weights
python -c "from huggingface_hub import snapshot_download; snapshot_download('robotics-diffusion-transformer/rdt-1b', local_dir='checkpoints/rdt-1b-pretrained')"
bash scripts/finetune_rdt.sh data/easy_mode checkpoints/rdt_ft 3 checkpoints/rdt-1b-pretrained data/siglip_features
```

Common overrides:

```bash
BATCH_SIZE=64 EPOCHS=50 LR=5e-5 NUM_WORKERS=16 bash scripts/train_act.sh data/easy_mode checkpoints/act_debug 0
```

Weights & Biases logging is off by default for public reproducibility. Enable it
with:

```bash
USE_WANDB=1 WANDB_PROJECT=DexHoldem WANDB_ENTITY=<entity> bash scripts/train_dp_unet.sh data/easy_mode checkpoints/dp_unet 0
```

## Testing

```bash
python train.py --list_models
python -m py_compile train.py workflow/download_data.py workflow/precompute_features.py
python test_code/test_training.py --run dp_pos_only
python test_code/test_training.py --run act_pos_only
```

The full image and RDT tests are more expensive because they instantiate large
vision/text backbones.

## Documentation

| Doc | Contents |
| --- | --- |
| `docs/release_reproduction.md` | End-to-end public reproduction guide |
| `docs/data_pipeline.md` | Dataset layout, preparation, and batch format |
| `docs/diffusion_policy.md` | DP(DINO), DP_transformer_resnet, and DP_unet details |
| `docs/act.md` | ACT architecture and training notes |
| `docs/rdt.md` | RDT_small and RDT_FT details |
| `docs/obs_encoder.md` | Shared observation encoder details |

## Deployment

Checkpoints are self-describing `.pt` files with model type, configs, and
normalization stats. A trained checkpoint can be served with:

```bash
python deploy_policy.py --ckpt checkpoints/dp_dino/latest.pt --port 13579
```

The robot client then connects with:

```bash
python robot_client.py --server_ip <GPU_SERVER_IP> --port 13579 --instruction 0
```
