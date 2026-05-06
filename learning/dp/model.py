"""
Diffusion Policy — implements PolicyModel using DDPM/DDIM noise prediction.

Architecture
------------
  ObsEncoder  →  ObsFeatures
                  ↓ .flat()           (Transformer backbone)
                  ↓ .flat_time()      (UNet backbone)
  Noise predictor (ConditionalUnet1D or TransformerForDiffusion)
  DDPM loss = MSE(predicted_noise, true_noise)

At inference, DDPM (or optionally DDIM) reverse diffusion produces actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

from learning.base import ModelConfig, PolicyModel
from learning.registry import register_model
from learning.common.encoders import ObsEncoder, ObsFeatures


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DiffusionPolicyConfig(ModelConfig):
    """Hyper-parameters for DiffusionPolicy (extends the shared ModelConfig)."""

    # --- Diffusion scheduler ---
    num_diffusion_iters: int = 100
    num_inference_iters: int = 100
    use_ddim: bool = False

    # --- Backbone selection ---
    # "auto"        — Transformer if DinoV2/V3 encoder, else UNet
    # "transformer" — always TransformerForDiffusion
    # "unet"        — always ConditionalUnet1D
    diffusion_model_type: str = "auto"

    # Transformer-specific
    transformer_hidden_size: int = 256
    transformer_depth: int = 8
    transformer_num_heads: int = 4
    transformer_causal_attn: bool = True
    transformer_n_cond_layers: int = 4

    # UNet-specific
    unet_diffusion_step_embed_dim: int = 256
    unet_down_dims: list[int] = field(default_factory=lambda: [256, 512, 1024])
    unet_kernel_size: int = 5
    unet_n_groups: int = 8

    # Dedicated instruction token in the denoiser (Method 1).
    # When True and use_instruction is also True, instruction gets its own
    # condition token at full hidden_size instead of being concatenated
    # into the obs vector.  Only applies to the Transformer backbone.
    dedicated_instr_token: bool = False

    # Condition masking — randomly zero visual features (rgb/depth) to
    # force the model to rely on instruction conditioning.
    cond_mask_prob: float = 0.0

    # EMA
    ema_power: float = 0.75

    # LR schedule
    warmup_steps: int = 500


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@register_model("diffusion_policy")
class DiffusionPolicy(PolicyModel):
    """
    Diffusion Policy with DDPM training and optional DDIM inference.

    Observation encoding
    --------------------
    The default ``ObsEncoder`` is used.  If a model variant needs a different
    observation representation (e.g. raw patch tokens), pass a custom
    ``nn.Module`` that returns ``ObsFeatures`` and has ``total_dim``.

    Backbone selection
    ------------------
    UNet backbone receives ``obs.flat_time()`` — (B, T*D) — as global
    conditioning.  Transformer backbone receives ``obs.flat()`` — (B, T, D) —
    as cross-attention keys.  The choice is made automatically based on the
    RGB encoder type (DinoV2/V3 → Transformer) or forced via
    ``diffusion_model_type``.
    """

    def __init__(
        self,
        obs_encoder: ObsEncoder,
        noise_pred_net: nn.Module,
        config: DiffusionPolicyConfig,
    ):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.noise_pred_net = noise_pred_net
        self.config = config

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.inference_scheduler = (
            DDIMScheduler(
                num_train_timesteps=config.num_diffusion_iters,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
            )
            if config.use_ddim
            else self.noise_scheduler
        )

        # EMA covers ALL parameters (encoder + denoiser), matching TexasPoker.
        self.ema = EMAModel(
            parameters=self.parameters(),
            power=config.ema_power,
        )

        self.norm_stats: dict = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        obs_encoder: ObsEncoder,
        config: DiffusionPolicyConfig,
    ) -> "DiffusionPolicy":
        from learning.dp.nets import ConditionalUnet1D, TransformerForDiffusion

        enc_type = obs_encoder.config.rgb_encoder_type
        use_transformer = config.diffusion_model_type == "transformer" or (
            config.diffusion_model_type == "auto"
            and ("dinov2" in enc_type or "dinov3" in enc_type)
        )

        if use_transformer:
            # When dedicated_instr_token is enabled, instruction gets its own
            # condition token and is excluded from the obs conditioning dim.
            num_instr = 0
            obs_dim_for_denoiser = obs_encoder.total_dim
            if config.dedicated_instr_token and config.use_instruction:
                num_instr = config.num_instructions
                obs_dim_for_denoiser -= config.instruction_embed_dim

            noise_pred_net = TransformerForDiffusion(
                input_dim=config.action_dim,
                global_cond_dim=obs_dim_for_denoiser,
                hidden_dim=config.transformer_hidden_size,
                num_layers=config.transformer_depth,
                num_heads=config.transformer_num_heads,
                max_seq_len=config.pred_horizon,
                obs_horizon=config.obs_horizon,
                causal_attn=config.transformer_causal_attn,
                n_cond_layers=config.transformer_n_cond_layers,
                num_instructions=num_instr,
            )
        else:
            # UNet receives the flattened (T * D) conditioning vector.
            global_cond_dim = obs_encoder.total_dim * config.obs_horizon
            noise_pred_net = ConditionalUnet1D(
                input_dim=config.action_dim,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=config.unet_diffusion_step_embed_dim,
                down_dims=config.unet_down_dims,
                kernel_size=config.unet_kernel_size,
                n_groups=config.unet_n_groups,
            )

        return cls(
            obs_encoder=obs_encoder,
            noise_pred_net=noise_pred_net,
            config=config,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _uses_transformer(self) -> bool:
        from learning.dp.nets import TransformerForDiffusion
        return isinstance(self.noise_pred_net, TransformerForDiffusion)

    @property
    def _has_dedicated_instr_token(self) -> bool:
        """True when the transformer denoiser has its own instruction projection."""
        return (self._uses_transformer
                and hasattr(self.noise_pred_net, 'instruction_proj')
                and self.noise_pred_net.instruction_proj is not None)

    def _condition(self, obs: ObsFeatures) -> Tensor:
        """Return the conditioning tensor in the format the backbone expects.

        When the transformer has a dedicated instruction token, the instruction
        modality is excluded from the obs condition (it is passed separately).
        """
        if self._uses_transformer:
            if self._has_dedicated_instr_token and "instruction" in obs.by_modality:
                # Exclude instruction — it will be a dedicated condition token
                filtered = {k: v for k, v in obs.by_modality.items()
                            if k != "instruction"}
                return torch.cat(list(filtered.values()), dim=-1)  # (B, T, D')
            return obs.flat()       # (B, T, D)   — cross-attention keys
        else:
            return obs.flat_time()  # (B, T*D)    — UNet global_cond

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    _VISUAL_KEYS = frozenset({"rgb", "depth"})

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        obs = self.obs_encoder(batch)           # ObsFeatures
        action = batch["action"].float()        # (B, pred_horizon, action_dim)
        B = action.shape[0]

        # Condition masking: randomly zero visual features so the model must
        # rely on instruction (and proprioception) alone.
        if self.training and self.config.cond_mask_prob > 0:
            for key in self._VISUAL_KEYS:
                if key in obs.by_modality:
                    mask = torch.rand(B, device=action.device) < self.config.cond_mask_prob
                    obs.by_modality[key] = obs.by_modality[key] * (~mask)[:, None, None].float()

        cond = self._condition(obs)

        # Extract instruction IDs for dedicated token (if applicable)
        instr_ids = batch.get("instruction") if self._has_dedicated_instr_token else None

        timesteps = torch.randint(
            0, self.config.num_diffusion_iters, (B,), device=action.device
        ).long()
        noise = torch.randn_like(action)
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)

        if self._uses_transformer:
            noise_pred = self.noise_pred_net(
                noisy_action, timesteps, cond=cond, instruction=instr_ids)
        else:
            noise_pred = self.noise_pred_net(noisy_action, timesteps, global_cond=cond)

        loss = F.mse_loss(noise_pred, noise)
        return {
            "loss":     loss,
            "arm_mse":  F.mse_loss(noise_pred[:, :, :6],  noise[:, :, :6]).detach(),
            "hand_mse": F.mse_loss(noise_pred[:, :, 6:],  noise[:, :, 6:]).detach(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(self, batch: dict[str, Tensor]) -> Tensor:
        # EMA covers ALL parameters (encoder + denoiser), matching TexasPoker.
        self.ema.store(self.parameters())
        self.ema.copy_to(self.parameters())

        try:
            obs = self.obs_encoder(batch)
            cond = self._condition(obs)
            B = cond.shape[0]
            device = cond.device

            instr_ids = batch.get("instruction") if self._has_dedicated_instr_token else None

            scheduler = self.inference_scheduler
            if self.config.use_ddim:
                scheduler.set_timesteps(self.config.num_inference_iters)

            action = torch.randn(
                B, self.config.pred_horizon, self.config.action_dim, device=device
            )
            for t in scheduler.timesteps:
                t_batch = t.unsqueeze(0).expand(B).to(device)
                if self._uses_transformer:
                    noise_pred = self.noise_pred_net(
                        action, t_batch, cond=cond, instruction=instr_ids)
                else:
                    noise_pred = self.noise_pred_net(action, t_batch, global_cond=cond)
                action = scheduler.step(noise_pred, t, action).prev_sample
        finally:
            self.ema.restore(self.parameters())

        action = action[:, : self.config.action_horizon]
        if self.norm_stats:
            action = self._unnormalize(action)
        return action

    def _unnormalize(self, action: Tensor) -> Tensor:
        stat = self.norm_stats.get("action", {})
        if not stat:
            return action
        mn = torch.tensor(stat["min"], device=action.device, dtype=action.dtype)
        mx = torch.tensor(stat["max"], device=action.device, dtype=action.dtype)
        action = (action + 1) / 2
        return action * (mx - mn + 1e-8) + mn

    # ------------------------------------------------------------------
    # Validation (full reverse diffusion, matches TexasPoker eval)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Validation metrics for DP.

        ``loss``/``arm_mse``/``hand_mse`` use the same noise-prediction
        objective as training, so validation curves are comparable with train
        loss. Full reverse-diffusion action error is still reported under the
        ``sample_*`` keys for deployment-oriented diagnostics.
        """
        result = {k: v.detach() for k, v in self.compute_loss(batch).items()}

        obs = self.obs_encoder(batch)
        cond = self._condition(obs)
        action = batch["action"].float()        # (B, pred_horizon, action_dim)
        B = action.shape[0]
        device = action.device
        instr_ids = batch.get("instruction") if self._has_dedicated_instr_token else None

        # Start from pure noise and denoise
        pred_action = torch.randn_like(action)
        self.noise_scheduler.set_timesteps(self.config.num_diffusion_iters)
        for t in self.noise_scheduler.timesteps:
            t_batch = t.unsqueeze(0).expand(B).to(device)
            if self._uses_transformer:
                noise_pred = self.noise_pred_net(
                    pred_action, t_batch, cond=cond, instruction=instr_ids)
            else:
                noise_pred = self.noise_pred_net(pred_action, t_batch, global_cond=cond)
            pred_action = self.noise_scheduler.step(noise_pred, t, pred_action).prev_sample

        result.update({
            "sample_mse": F.mse_loss(pred_action, action).detach(),
            "sample_arm_mse": F.mse_loss(
                pred_action[:, :, :6], action[:, :, :6]).detach(),
            "sample_hand_mse": F.mse_loss(
                pred_action[:, :, 6:], action[:, :, 6:]).detach(),
        })
        for k, v in self._unnorm_action_mse(pred_action, action).items():
            result[f"sample_{k}"] = v
        return result

    # ------------------------------------------------------------------
    # Optimizer / EMA
    # ------------------------------------------------------------------

    def configure_optimizers(
        self, lr: float = 1e-4, weight_decay: float = 1e-5
    ) -> list[torch.optim.Optimizer]:
        # Uniform weight_decay on all params (matches TexasPoker).
        trainable = [p for p in self.parameters() if p.requires_grad]
        return [
            torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        ]

    def on_after_step(self) -> None:
        # EMA covers ALL parameters (encoder + denoiser), matching TexasPoker.
        self.ema.step(self.parameters())
