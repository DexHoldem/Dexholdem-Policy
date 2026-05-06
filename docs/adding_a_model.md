# Adding a New Model

The public training stack keeps model integration small:

- `learning/base.py` defines `PolicyModel` and `ModelConfig`.
- `learning/registry.py` maps a string name to a policy class.
- `learning/common/encoders.py` converts robot batches into `ObsFeatures`.
- `train.py` builds the encoder, builds the selected policy, and owns data loading, normalization, checkpointing, validation, and logging.

The public repo ships four model families: Diffusion Policy (`diffusion_policy`), ACT (`act`), BAKU (`baku`), and RDT (`rdt`).

## 1. Create a Package

```text
learning/
  mymodel/
    __init__.py
    model.py
```

## 2. Implement the Policy

```python
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from learning.base import ModelConfig, PolicyModel
from learning.common.encoders import ObsEncoder
from learning.registry import register_model


@dataclass
class MyModelConfig(ModelConfig):
    hidden_dim: int = 256


@register_model("my_model")
class MyModel(PolicyModel):
    def __init__(self, obs_encoder: ObsEncoder, config: MyModelConfig):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.config = config
        self.head = torch.nn.Linear(
            obs_encoder.total_dim * config.obs_horizon,
            config.action_dim * config.pred_horizon,
        )
        self.norm_stats = {}

    @classmethod
    def build(cls, obs_encoder: ObsEncoder, config: MyModelConfig) -> "MyModel":
        return cls(obs_encoder=obs_encoder, config=config)

    def compute_loss(self, batch):
        obs = self.obs_encoder(batch)
        pred = self.head(obs.flat_time())
        pred = pred.reshape(-1, self.config.pred_horizon, self.config.action_dim)
        loss = F.mse_loss(pred, batch["action"].float())
        return {"loss": loss}

    @torch.no_grad()
    def predict_action(self, batch):
        obs = self.obs_encoder(batch)
        pred = self.head(obs.flat_time())
        pred = pred.reshape(-1, self.config.pred_horizon, self.config.action_dim)
        return pred[:, : self.config.action_horizon]
```

## 3. Register on Import

```python
# learning/mymodel/__init__.py
from learning.mymodel.model import MyModel, MyModelConfig

__all__ = ["MyModel", "MyModelConfig"]
```

Then import the package in `learning/__init__.py`:

```python
import learning.mymodel  # registers "my_model"
```

## 4. Add CLI Options

Add model-specific flags in `_parse_args()` and return a config in `_build_model_config()`:

```python
p.add_argument("--mymodel_hidden_dim", type=int, default=256)

elif model_name == "my_model":
    return MyModelConfig(**base, hidden_dim=args.mymodel_hidden_dim)
```

## 5. Train

```bash
python train.py \
  --model my_model \
  --train_path data/easy_mode/0/pick_up_left_train \
  --val_path data/easy_mode/0/pick_up_left_test \
  --save_path checkpoints/my_model
```

## Batch Contract

Common keys include:

| Key | Shape | Notes |
| --- | --- | --- |
| `rgb` | `(B, T, cams, H, W, 3)` | raw RGB frames |
| `depth` | `(B, T, cams, H, W)` | depth frames |
| `rgb_features` | `(B, T, cams, ...)` | precomputed DinoV2 or SigLIP features |
| `pos` | `(B, T, 30)` | arm and hand positions |
| `efforts` | `(B, T, 24)` | hand efforts |
| `velocity` | `(B, T, 30)` | joint velocities |
| `touch` | `(B, T, 60)` | tactile readings |
| `action` | `(B, pred_horizon, action_dim)` | normalized target action |
| `instruction` | `(B,)` | integer task id when instruction conditioning is enabled |

Use `obs_encoder.total_dim` or `obs_encoder.modality_dims` to size layers instead of hard-coding feature dimensions.
