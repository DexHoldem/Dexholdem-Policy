# Diffusion Policy

Registered model name: `diffusion_policy`

The public release exposes three Diffusion Policy recipes from the same model
implementation in `learning/dp/`.

| Release name | Script | RGB path | Denoiser |
| --- | --- | --- | --- |
| DP(DINO) | `scripts/train_dp.sh` | frozen DinoV2 ViT-L/14 CLS features | Transformer |
| DP_transformer_resnet | `scripts/train_dp_transformer.sh` | trainable ResNet18 RGB by default | Transformer |
| DP_unet | `scripts/train_dp_unet.sh` | trainable ResNet18 RGBD | Conditional 1D UNet |

`scripts/train_dp_unet.sh` is a public alias for the older
`scripts/train_dp_light.sh` script.

## Architecture

All variants share:

1. `ObsEncoder` converts RGB/depth/state/instruction inputs into
   `ObsFeatures`.
2. A diffusion denoiser predicts action noise for a 30-d action sequence.
3. DDPM training uses 100 diffusion steps and an epsilon prediction loss.
4. EMA is maintained over all trainable parameters and used at inference.

Transformer variants use `obs.flat()` as cross-attention memory. The UNet
variant uses `obs.flat_time()` as global conditioning.

## DP(DINO)

```bash
python workflow/precompute_features.py \
  --data_dir data/easy_mode \
  --feature_dir data/vitl14_features \
  --encoder dinov2_vitl14 \
  --gpu 0

bash scripts/train_dp.sh data/easy_mode checkpoints/dp_dino 0 data/vitl14_features
```

Default settings:

| Setting | Value |
| --- | --- |
| RGB encoder | DinoV2 ViT-L/14, frozen |
| RGB feature | CLS token, 1024-d per camera |
| Observation | `img-depth-pos` |
| Denoiser | Transformer, hidden 768, depth 12, heads 12 |
| Diffusion steps | 100 |
| Batch size | 128 |
| Epochs | 300 |
| AMP | off |

If the feature directory argument is omitted, the script runs DinoV2 on the
fly. Precomputed features are recommended for full-data training.

## DP_transformer_resnet

```bash
bash scripts/train_dp_transformer.sh \
  data/easy_mode checkpoints/dp_transformer_resnet 0
```

This is the configurable transformer entry point. It defaults to trainable
ResNet18 RGB features, and exposes architecture overrides:

```bash
HIDDEN=512 DEPTH=8 HEADS=8 N_COND_LAYERS=2 \
  bash scripts/train_dp_transformer.sh data/easy_mode checkpoints/dp_transformer_resnet_small 0
```

For a frozen DinoV2 variant with precomputed features:

```bash
RGB_ENCODER=dinov2_vitl14 bash scripts/train_dp_transformer.sh \
  data/easy_mode checkpoints/dp_transformer_dino 0 data/vitl14_features
```

## DP_unet

```bash
bash scripts/train_dp_unet.sh data/easy_mode checkpoints/dp_unet 0
```

Default settings:

| Setting | Value |
| --- | --- |
| RGB/depth encoder | fused 4-channel ResNet18 |
| Observation | `img-depth-pos` |
| Denoiser | Conditional 1D UNet |
| Down dims | `[256, 512, 1024]` |
| Diffusion steps | 100 |
| Batch size | 256 |
| Epochs | 100 |
| AMP | bf16/fp16 when CUDA supports it |

## Instruction Conditioning

The release scripts use integer instruction IDs by default:

```bash
--use_instruction --num_instructions 32 --instruction_embed_dim 128
```

To use text instruction embeddings for the DP variants:

```bash
INSTR_MODE=text TEXT_ENCODER=clip \
  bash scripts/train_dp.sh data/easy_mode checkpoints/dp_dino_text 0 data/vitl14_features
```

The instruction mapping lives in `workflow/instructions.json` and is also
written to `data/easy_mode/instructions.json` by `workflow/organize_data.py`.

## Logging

W&B is disabled by default. Enable it explicitly:

```bash
USE_WANDB=1 WANDB_PROJECT=DexHoldem WANDB_ENTITY=<entity> \
  bash scripts/train_dp_unet.sh data/easy_mode checkpoints/dp_unet 0
```

## Key Files

| File | Purpose |
| --- | --- |
| `learning/dp/model.py` | `DiffusionPolicy`, config, loss, EMA, inference |
| `learning/dp/models.py` | UNet and transformer denoiser implementations |
| `learning/dp/nets.py` | Denoiser exports used by the policy |
| `scripts/train_dp.sh` | DP(DINO) launch recipe |
| `scripts/train_dp_transformer.sh` | Configurable transformer launch recipe |
| `scripts/train_dp_unet.sh` | DP_unet public launch alias |
