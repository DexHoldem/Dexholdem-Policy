# ACT

Registered model name: `act`

ACT is the Action Chunking with Transformers baseline. In this repository it
uses the shared `ObsEncoder`, a CVAE action encoder during training, and a
transformer encoder-decoder policy for action chunk prediction.

## Training

```bash
bash scripts/train_act.sh data/easy_mode checkpoints/act 0
```

Default settings:

| Setting | Value |
| --- | --- |
| RGB/depth encoder | fused 4-channel ResNet18 |
| Observation | `img-depth-pos` |
| Hidden dim | 512 |
| Heads | 8 |
| Encoder layers | 4 |
| Decoder layers | 7 |
| Latent dim | 32 |
| KL weight | 10 |
| Batch size | 512 |
| Epochs | 100 |
| AMP | enabled |

## Architecture

ACT treats each observation modality as tokens rather than flattening all
features into one vector:

1. `ObsEncoder` returns `ObsFeatures.by_modality`.
2. Each modality is projected to `hidden_dim`.
3. During training, a CVAE encoder consumes `[CLS, obs_tokens, action_tokens]`
   and predicts latent `mu` and `log_var`.
4. The policy encoder consumes `[z_token, obs_tokens]`.
5. The policy decoder uses action queries to predict a `(B, pred_horizon, 30)`
   action chunk.

At inference, `z` is set to zero, so ACT is deterministic.

## Instruction Conditioning

The release script enables integer instruction conditioning by default:

```bash
--use_instruction --num_instructions 32 --instruction_embed_dim 128
```

The instruction embedding is projected as its own modality token and attends
with RGB/depth/proprioception tokens in the policy transformer.

Text instruction embeddings are also supported:

```bash
INSTR_MODE=text \
  bash scripts/train_act.sh data/easy_mode checkpoints/act_text 0
```

## Useful Overrides

```bash
BATCH_SIZE=128 EPOCHS=50 LR=5e-5 NUM_WORKERS=16 \
  bash scripts/train_act.sh data/easy_mode checkpoints/act_debug 0
```

Disable fused RGBD and use RGB only:

```bash
REPR_TYPE=img-pos bash scripts/train_act.sh data/easy_mode checkpoints/act_rgb 0
```

Enable W&B:

```bash
USE_WANDB=1 WANDB_PROJECT=DexHoldem_ACT WANDB_ENTITY=<entity> \
  bash scripts/train_act.sh data/easy_mode checkpoints/act 0
```

## Key Files

| File | Purpose |
| --- | --- |
| `learning/act/model.py` | ACT policy, CVAE loss, inference |
| `scripts/train_act.sh` | Public ACT training recipe |
