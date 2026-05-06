"""
Diffusion backbone networks (UNet1D, Transformer1D).

This module re-exports the battle-tested network implementations from the
existing learning/dp/models.py so that the new DiffusionPolicy wrapper can
import them without duplicating complex architecture code.

In the future this file can be made fully self-contained by moving the
class definitions here directly.
"""

from __future__ import annotations

import os, sys

# Make the legacy models.py importable regardless of cwd.
_here = os.path.dirname(__file__)
if _here not in sys.path:
    sys.path.insert(0, _here)

from models import (          # noqa: E402  (legacy module)
    ConditionalUnet1D,
    ConditionalTransformerForDiffusion,
    SimpleBCModel,
)

# Alias used by the new DiffusionPolicy.
TransformerForDiffusion = ConditionalTransformerForDiffusion

__all__ = [
    "ConditionalUnet1D",
    "TransformerForDiffusion",
    "SimpleBCModel",
]
