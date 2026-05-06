"""
Model zoo: importing this package registers the public built-in models.

After ``import learning``, the registry contains:
  - "diffusion_policy"
  - "act"
  - "baku"
  - "rdt"

Additional models are registered by importing their sub-packages.
"""

from learning.common.hf_compat import ensure_transformers_deepspeed_attr

ensure_transformers_deepspeed_attr()

import learning.dp      # registers "diffusion_policy"
import learning.act     # registers "act"
import learning.baku    # registers "baku"
import learning.rdt     # registers "rdt"

from learning.base import PolicyModel, ModelConfig
from learning.registry import build_model, list_models, register_model

__all__ = [
    "PolicyModel",
    "ModelConfig",
    "build_model",
    "list_models",
    "register_model",
]
