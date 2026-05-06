# RDT

Registered model name: `rdt`

The public release exposes two RDT recipes:

| Release name | Script | Init | Size |
| --- | --- | --- | --- |
| RDT_small | `scripts/train_rdt_small.sh` | from scratch | 170M-style ablation |
| RDT_FT | `scripts/finetune_rdt.sh` | official RDT-1B checkpoint | 1B backbone |

`scripts/train_rdt_small.sh` is a public alias for the older
`scripts/train_rdt.sh` script.

## Architecture

RDT combines three condition streams:

| Stream | Source | Shape before projection |
| --- | --- | --- |
| Vision | SigLIP-SO400M patch tokens | `(B, T, cams, patches, 1152)` |
| State | robot joint positions | `(B, T, 30)` |
| Text | frozen T5 instruction tokens | `(B, tokens, raw_t5_dim)` |

The denoiser is an ACI transformer. Even layers cross-attend to language and
odd layers cross-attend to vision. The model predicts the clean action sample
(`prediction_type=sample`) and uses DPMSolver for fast inference.

## Feature Precomputation

RDT should use precomputed SigLIP patch tokens for full-data training:

```bash
python workflow/precompute_features.py \
  --data_dir data/easy_mode \
  --feature_dir data/siglip_features \
  --encoder siglip_so400m \
  --gpu 0
```

The feature directory mirrors the data directory:

```text
data/siglip_features/0/pick_up_card_train_<N>/data0001/
  rgb_features_cam0.npy
  rgb_features_cam1.npy
  rgb_features_cam2.npy
```

## RDT_small

```bash
bash scripts/train_rdt_small.sh \
  data/easy_mode checkpoints/rdt_small 0 data/siglip_features
```

Default settings:

| Setting | Value |
| --- | --- |
| Text encoder | T5-XXL embeddings |
| RGB encoder | SigLIP-SO400M, frozen |
| Observation | `img-pos` |
| `obs_horizon` | 2 |
| Hidden size | 1024 |
| Depth | 14 |
| Heads | 32 |
| Diffusion steps | 1000 |
| Inference steps | 5 |
| Batch size | 64 |
| Grad accumulation | 2 |
| Epochs | 100 |

For lower-memory experiments:

```bash
BATCH_SIZE=16 GRAD_ACCUM=4 GRAD_CKPT=1 \
  bash scripts/train_rdt_small.sh data/easy_mode checkpoints/rdt_small_mem 0 data/siglip_features
```

## RDT_FT

Download the official checkpoint:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('robotics-diffusion-transformer/rdt-1b', local_dir='checkpoints/rdt-1b-pretrained')"
```

Finetune:

```bash
bash scripts/finetune_rdt.sh \
  data/easy_mode checkpoints/rdt_ft 0 \
  checkpoints/rdt-1b-pretrained data/siglip_features
```

RDT_FT uses the 2048-d, 28-layer RDT-1B architecture. The loader imports
matching official weights and skips robot-specific shape mismatches such as the
30-d state adaptor and 30-d output head.

Default finetuning settings:

| Setting | Value |
| --- | --- |
| Batch size | 32 |
| Grad accumulation | 4 |
| LR | `5e-5` |
| Epochs | 100 |
| Save/eval freq | 1 |
| AMP | enabled |

## Instruction Text

RDT reads `workflow/instructions.json` by default. It encodes all instruction
strings once at model construction time, stores the frozen token embeddings as
buffers, and then performs a fast lookup by integer instruction ID during
training.

To use a custom instruction file:

```bash
INSTRUCTIONS_FILE=path/to/instructions.json \
  bash scripts/train_rdt_small.sh data/easy_mode checkpoints/rdt_small_custom 0 data/siglip_features
```

## Logging

W&B is disabled by default:

```bash
USE_WANDB=1 WANDB_PROJECT=DexHoldem_RDT WANDB_ENTITY=<entity> \
  bash scripts/train_rdt_small.sh data/easy_mode checkpoints/rdt_small 0 data/siglip_features
```

## Key Files

| File | Purpose |
| --- | --- |
| `learning/rdt/model.py` | RDT model, ACI decoder, T5 encoding, pretrained loading |
| `scripts/train_rdt_small.sh` | RDT_small public launch alias |
| `scripts/train_rdt.sh` | Backward-compatible RDT_small implementation script |
| `scripts/finetune_rdt.sh` | RDT_FT launch recipe |
