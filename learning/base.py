"""
Abstract base class for all policy models.

Every model in this framework must subclass PolicyModel and implement
the three abstract methods: compute_loss, predict_action, and build.

See docs/adding_a_model.md for a step-by-step integration guide.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Base configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Shared hyperparameters that every policy model receives.

    Model-specific configs should inherit from this class and add their
    own fields.  All fields here are accessible to the training loop, so
    keep the names stable.
    """

    # --- Action space ---
    action_dim: int = 30            # total output dimensions (6 arm + 24 hand)

    # --- Temporal windows ---
    obs_horizon: int = 1            # how many past timesteps are fed as obs
    action_horizon: int = 32        # how many steps to execute per prediction
    pred_horizon: int = 64          # how many steps to predict in one forward pass

    # --- Instruction conditioning (multi-task) ---
    use_instruction: bool = False
    use_text_instruction: bool = False
    num_instructions: int = 14      # number of distinct task instructions
    instruction_embed_dim: int = 128 # dimension for the instruction embedding


# ---------------------------------------------------------------------------
# Abstract policy interface
# ---------------------------------------------------------------------------

class PolicyModel(nn.Module, ABC):
    """
    Abstract base class that every policy model must subclass.

    The training loop interacts with a model exclusively through four methods:

      build()            — construct from config (classmethod)
      compute_loss()     — supervised training step
      predict_action()   — roll-out / evaluation
      configure_optimizers() — (optional) custom optimizers / LR groups
      on_after_step()    — (optional) post-optimizer hook (e.g. EMA update)

    Both ``compute_loss`` and ``predict_action`` receive the **same batch
    dict** so the data pipeline is identical for all models.

    Batch format
    ------------
    All tensors live on the model's device.  Keys are present only when the
    corresponding modality is enabled in the experiment config.

    Visual observations
        "rgb"           (B, obs_horizon, num_cams, H, W, 3)  uint8, [0, 255]
        "depth"         (B, obs_horizon, num_cams, H, W)     float32
        "rgb_features"  (B, obs_horizon, num_cams, rgb_dim)  float32
                        present when precompute_rgb_features=True

    Proprioception — all normalized to [-1, 1]
        "pos"           (B, obs_horizon, 30)   full joint positions
        "eef"           (B, obs_horizon, 6)    arm joints only
        "hand_pos"      (B, obs_horizon, 24)   hand joints only
        "efforts"       (B, obs_horizon, 24)   joint efforts (hand)
        "velocity"      (B, obs_horizon, 30)   joint velocities
        "touch"         (B, obs_horizon, 60)   tactile sensors

    Target actions — normalized to [-1, 1]
        "action"        (B, pred_horizon, action_dim)

    Instruction (multi-task)
        "instruction"   (B,)   integer IDs in [0, num_instructions)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def build(
        cls,
        obs_encoder: "ObsEncoder",   # noqa: F821  (imported at runtime)
        config: ModelConfig,
    ) -> "PolicyModel":
        """
        Factory method used by the registry to instantiate a model.

        Args:
            obs_encoder: A fully constructed ObsEncoder whose
                ``total_dim`` property tells this model the size of the
                encoded observation vector it will receive.
            config: Model-specific config (a subclass of ModelConfig).

        Returns:
            A ready-to-train PolicyModel instance.
        """

    # ------------------------------------------------------------------
    # Core interface — must be implemented
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Compute the training loss for one batch.

        Args:
            batch: Dict following the batch format described in the class
                   docstring.  The "action" key is always present.

        Returns:
            A dict that **must** contain the key ``"loss"`` (a scalar Tensor
            with gradients).  Any additional keys (e.g. ``"mse"``, ``"kl"``)
            are automatically logged to WandB by the training loop.

        Example::

            def compute_loss(self, batch):
                obs = self.obs_encoder(batch)          # (B, T, D)
                pred = self.head(obs.flatten(1))       # (B, action_dim)
                loss = F.mse_loss(pred, batch["action"][:, 0])
                return {"loss": loss}
        """

    @abstractmethod
    def predict_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Predict actions from a batch of observations.

        The ``"action"`` key is **not** present in ``batch`` at inference
        time.

        Args:
            batch: Observation-only dict (same keys as training, minus
                   ``"action"``).

        Returns:
            actions: (B, action_horizon, action_dim) float Tensor in the
                     **original (un-normalized) action space**.  The caller
                     is responsible for executing ``action_horizon`` steps.
        """

    # ------------------------------------------------------------------
    # DataParallel compatibility
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Default forward delegates to compute_loss for DataParallel."""
        return self.compute_loss(batch)

    # ------------------------------------------------------------------
    # Optional hooks — override to customize behaviour
    # ------------------------------------------------------------------

    def configure_optimizers(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ) -> list[torch.optim.Optimizer]:
        """
        Return a list of optimizers for this model.

        The default creates a single AdamW optimizer over all parameters.
        Override to use per-group learning rates, frozen layers, etc.

        Returns:
            List of optimizers.  The training loop calls ``.step()`` and
            ``.zero_grad()`` on every optimizer each iteration.
        """
        return [
            torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        ]

    def on_after_step(self) -> None:
        """
        Called by the training loop **after** every optimizer step.

        Override for post-step operations such as EMA parameter averaging.
        The default implementation is a no-op.
        """

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _unnorm_action_mse(
        self, pred: Tensor, target: Tensor,
    ) -> dict[str, Tensor]:
        """Compute MSE in the original (unnormalized) action scale.

        Returns an empty dict if ``self.norm_stats`` lacks ``"action"``
        stats, so callers can safely ``dict.update()`` the result.
        Caches min/max/range tensors to avoid re-creating them every call.
        """
        stat = getattr(self, "norm_stats", {}).get("action")
        if not stat:
            return {}
        # Cache tensors on first call or device change.
        cache = getattr(self, "_unnorm_cache", None)
        if cache is None or cache[0] != pred.device:
            mn = torch.tensor(stat["min"], device=pred.device, dtype=torch.float32)
            mx = torch.tensor(stat["max"], device=pred.device, dtype=torch.float32)
            rng = torch.clamp(mx - mn, min=1e-8)
            self._unnorm_cache = (pred.device, mn, mx, rng)
        else:
            _, mn, mx, rng = cache
        # unnormalize: [-1,1] → original scale
        p = (pred  + 1) / 2 * rng + mn
        t = (target + 1) / 2 * rng + mn
        return {
            "unnorm_mse":      F.mse_loss(p, t).detach(),
            "unnorm_arm_mse":  F.mse_loss(p[:, :, :6],  t[:, :, :6]).detach(),
            "unnorm_hand_mse": F.mse_loss(p[:, :, 6:],  t[:, :, 6:]).detach(),
        }
