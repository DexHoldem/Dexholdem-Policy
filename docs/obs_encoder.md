# Observation Encoder

The `ObsEncoder` is a shared module used by all models to process raw
sensor data into feature vectors. It is configured via `ObsEncoderConfig`
and produces `ObsFeatures` — a structured output that models can consume
in different ways.

## Architecture

```
batch dict
    |
    ├── "rgb"         → RGBEncoder (per camera) → rgb features
    ├── "depth"       → DepthEncoder (per camera) → depth features
    ├── "rgb_features"→ bypass backbone, project only
    ├── "pos"         → StateEncoder (2-layer MLP) → pos features
    ├── "eef"         → StateEncoder → eef features
    ├── "hand_pos"    → StateEncoder → hand_pos features
    ├── "efforts"     → StateEncoder → efforts features
    ├── "velocity"    → StateEncoder → velocity features
    ├── "touch"       → StateEncoder → touch features
    └── "instruction" → InstructionEncoder → instruction features
              |
              ▼
         ObsFeatures
```

## ObsFeatures Output

`ObsEncoder.forward(batch)` returns an `ObsFeatures` object with three
access patterns:

```python
obs = encoder(batch)

# Option 1: Flat concatenation per timestep (Transformer input)
x = obs.flat()        # (B, T, total_D)

# Option 2: Also flatten time (UNet global_cond, MLP input)
x = obs.flat_time()   # (B, T * total_D)

# Option 3: Per-modality access (ACT token building)
rgb = obs.by_modality["rgb"]   # (B, T, rgb_D)
pos = obs.by_modality["pos"]   # (B, T, pos_D)

# Metadata
obs.total_dim           # int — sum of all modality dims
obs.modality_dims       # dict[str, int] — per-modality output sizes
```

## RGB Encoders

### ResNet18 (trainable)

Default for ACT and DP Light. Processes images through a modified
ResNet18 with internal resize:

```
Input: (B, T, num_cams, 240, 320, 3) uint8
    → resize to 240×320
    → random crop 216×288 (train) / center crop 216×288 (eval)
    → ResNet18 → (B, T, num_cams, 512)
    → Linear(512, rgb_per_cam_output) per camera
    → concat cameras → (B, T, rgb_per_cam_output * num_cams)
```

### DinoV2 (frozen)

Used by DP and DP Max. Available variants:
- `dinov2_vits14` — 384-d CLS token
- `dinov2_vitb14` — 768-d CLS token
- `dinov2_vitl14` — 1024-d CLS token (default for DP)
- `dinov2_vitg14` — 1536-d CLS token
- `dinov2_vitl14_patch` — 1024-d × 256 patch tokens (for DP Max)

```
Input: (B, T, num_cams, 240, 320, 3) uint8
    → resize to 224×224 (DinoV2 native)
    → normalize with ImageNet stats
    → DinoV2 backbone (frozen)
    → CLS token (1024-d) or patch tokens (256 × 1024-d)
    → Linear projection per camera
```

### SigLIP-SO400M (frozen)

Used by RDT. Processes at 384x384 resolution:

```
Input: (B, T, num_cams, 240, 320, 3) uint8
    → resize to 384×384
    → normalize to [-1, 1]
    → SigLIP backbone (frozen)
    → 728 patch tokens × 1152-d (27×27 grid, first token dropped to match
      precomputed features)
```

### Precomputed Features

When `--precompute_rgb_features` is set, the RGB backbone is bypassed
entirely. Features are loaded from disk (`--feature_dir`):

```
batch["rgb_features"]: (B, T, num_cams, feat_dim)        # CLS tokens
                    or (B, T, num_cams, N_patches, feat_dim)  # patch tokens
    → Linear projection only (no backbone forward pass)
```

### Frozen Backbone Sharing

When `freeze_rgb_encoder=True` and multiple cameras are used, a single
frozen backbone is shared across all camera encoders instead of loading
N duplicate copies:

- DinoV2: saves memory when sharing one frozen ViT across cameras.
- SigLIP (RDT): saves memory when sharing one frozen SO400M backbone.

## Depth Encoder

ResNet18 modified for single-channel input:

```
Input: (B, T, num_cams, 240, 320) float32
    → unsqueeze channel → (B, T, num_cams, 1, 240, 320)
    → ResNet18 (1-channel) → (B, T, num_cams, 512)
    → Linear(512, depth_per_cam_output) per camera
```

## Fused RGBD Encoder

When `--fuse_rgbd` is set, RGB (3ch) + depth (1ch) are combined into
a single 4-channel ResNet18 per camera. This halves backbone passes
from 6 (3 RGB + 3 depth) to 3. Used by ACT and DP Light.

## State Encoders

Each proprioceptive modality has a 2-layer MLP:

```
Input: (B, T, raw_dim)
    → Linear(raw_dim, 256) → ReLU → Linear(256, output_size)
    → (B, T, output_size)
```

| Modality | Raw dim | Default output |
|----------|---------|----------------|
| `pos` | 30 | 128 |
| `eef` | 6 | 32 |
| `hand_pos` | 24 | 64 |
| `efforts` | 24 | 64 |
| `velocity` | 30 | 64 |
| `touch` | 60 | 64 |

## Instruction Encoder

ObsEncoder's instruction encoding is used by **DP**, **ACT**, and **BAKU**.
RDT has its own task-text path through the configured T5 encoder.

### Integer-ID mode (`--use_instruction`)

```
Input: (B,) int64 instruction IDs (0–13)
    → one_hot(num_instructions=14) → (B, 14)
    → Linear(14, 64, bias=False) + xavier_uniform init
    → LayerNorm(64)
    → (B, 64)
    → expand to (B, T, 64) in ObsFeatures.by_modality["instruction"]
```

### Text mode (`--use_text_instruction`)

Text is encoded once at construction time (CLIP or sentence-transformers),
stored as frozen buffers, then looked up and projected at runtime:

```
Construction:
  instructions.json → text encoder → (14, raw_dim) stored as frozen buffer
    "clip"                  → openai/clip-vit-base-patch32     → raw_dim=512
    "clip_large"            → openai/clip-vit-large-patch14    → raw_dim=768
    "sentence_transformers" → all-MiniLM-L6-v2                 → raw_dim=384

Forward:
  batch["instruction"] (B,) int64
    → buffer[ids] → (B, raw_dim)
    → Linear(raw_dim, 64) → LayerNorm(64)
    → (B, 64) → expand to (B, T, 64) in ObsFeatures.by_modality["instruction"]
```

### How Models Consume It

| Model | Fusion Strategy |
|-------|----------------|
| **DP (Transformer)** | Instruction is part of `obs.flat()` → cross-attention keys for denoiser |
| **DP (UNet)** | Instruction is part of `obs.flat_time()` → global FiLM conditioning |
| **ACT** | Instruction gets own `Linear→LN` projection, becomes a token in self-attention |

### When Instruction Is Missing

If `batch["instruction"]` is absent at forward time, a zero vector of shape
`(B, instruction_embed_dim)` is used as fallback.

## Representation Type

The `--representation_type` flag controls which modalities are active:

```
"img"                    → RGB only
"img-pos"                → RGB + full joint positions
"img-depth-pos"          → RGB + depth + positions (default)
"img-pos-efforts"        → RGB + positions + hand efforts
"img-depth-pos-efforts"  → all four
```

Camera selection via `--camera_indices "012"` (default: all 3 cameras).

## Config

```python
@dataclass
class ObsEncoderConfig:
    representation_type: list[str]  # parsed from "img-depth-pos"
    camera_indices: list[int]
    rgb_encoder_type: str = "resnet18"
    depth_encoder_type: str = "resnet18"
    freeze_rgb_encoder: bool = False
    precompute_rgb_features: bool = False
    fuse_rgbd: bool = False

    rgb_per_cam_output: int = 96
    depth_per_cam_output: int = 32
    pos_output_size: int = 128
    eef_output_size: int = 32
    hand_pos_output_size: int = 64
    efforts_output_size: int = 64
    velocity_output_size: int = 64
    touch_output_size: int = 64

    instruction_embed_dim: int = 64
    num_instructions: int = 14
    use_instruction: bool = False
    use_text_instruction: bool = False
    text_encoder: str = "clip"
```

## Key Source Files

- `learning/common/encoders.py` — `ObsEncoder`, `ObsFeatures`, all sub-encoders
- `learning/base.py` — `ModelConfig` (contains obs_horizon, action dimensions)
