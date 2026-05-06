"""
ACT — Action Chunking with Transformers.

Reference: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
           (Zhao et al., RSS 2023)

Encoder strategy
----------------
ACT benefits from treating each sensor modality as a separate token stream
rather than concatenating everything into a flat vector.  The
``ObsFeatures.by_modality`` dict makes this straightforward:

  obs.by_modality["rgb"]   → (B, T, rgb_dim)   — visual tokens
  obs.by_modality["pos"]   → (B, T, pos_dim)   — proprioception tokens
  obs.by_modality["instruction"] → (B, T, D)   — task token

Each modality is projected to a shared ``hidden_dim`` and the resulting
tokens are concatenated into a sequence for the Transformer to attend over.
This is qualitatively different from DiffusionPolicy, which flattens
everything before conditioning.

Architecture
------------
Training (CVAE):
  [CLS, per-modality obs tokens, action tokens]
        → CVAE Encoder Transformer
        → mean, log_var  →  z (re-parameterisation)

  [per-modality obs tokens, z token]
        → Policy Encoder  →  memory
        → Policy Decoder (action queries × memory)
        → action predictions

  Loss = MSE(pred, target) + kl_weight * KL(z || N(0, I))

Inference:
  z = 0  (prior mean, deterministic)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from learning.base import ModelConfig, PolicyModel
from learning.registry import register_model
from learning.common.encoders import ObsEncoder, ObsFeatures


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ACTConfig(ModelConfig):
    """Hyper-parameters for ACT (extends the shared ModelConfig)."""

    # Transformer dimensions
    hidden_dim: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 4      # policy encoder
    num_decoder_layers: int = 7      # policy decoder
    dim_feedforward: int = 3200

    # CVAE
    latent_dim: int = 32
    kl_weight: float = 10.0

    # Dropout
    dropout: float = 0.1

    # Condition masking: randomly zero visual tokens during training
    # to force instruction dependence (classifier-free guidance style)
    cond_mask_prob: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SinePosEnc(nn.Module):
    """Fixed sine/cosine positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.shape[1]]


def _make_tf_encoder(d_model, nhead, ffn_dim, dropout, n_layers) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
        dropout=dropout, batch_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


def _make_tf_decoder(d_model, nhead, ffn_dim, dropout, n_layers) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
        dropout=dropout, batch_first=True,
    )
    return nn.TransformerDecoder(layer, num_layers=n_layers)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@register_model("act")
class ACT(PolicyModel):
    """
    Action Chunking with Transformers.

    Uses ``ObsFeatures.by_modality`` to project each sensor stream to a
    shared ``hidden_dim`` before building the token sequence.  This avoids
    the information-destroying flat concatenation used by simpler models and
    lets the Transformer attend across heterogeneous sensor types.
    """

    def __init__(self, obs_encoder: ObsEncoder, config: ACTConfig):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.config = config
        D = config.hidden_dim

        # --- Per-modality input projections ---
        # Each modality gets its own linear layer projecting from its
        # native dimension to the shared hidden_dim.  New modalities are
        # handled automatically via modality_dims without touching the
        # Transformer core.
        modality_dims = obs_encoder.modality_dims
        self.modality_projs = nn.ModuleDict({
            key: nn.Sequential(nn.Linear(dim, D), nn.LayerNorm(D))
            for key, dim in modality_dims.items()
        })

        # --- CVAE encoder ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.action_proj = nn.Linear(config.action_dim, D)
        self.cvae_encoder = _make_tf_encoder(
            D, config.num_heads, config.dim_feedforward,
            config.dropout, config.num_encoder_layers,
        )
        self.mean_proj   = nn.Linear(D, config.latent_dim)
        self.logvar_proj = nn.Linear(D, config.latent_dim)

        # --- Policy transformer ---
        self.pos_enc       = _SinePosEnc(D)
        self.latent_proj   = nn.Linear(config.latent_dim, D)
        self.policy_enc    = _make_tf_encoder(
            D, config.num_heads, config.dim_feedforward,
            config.dropout, config.num_encoder_layers,
        )
        self.policy_dec    = _make_tf_decoder(
            D, config.num_heads, config.dim_feedforward,
            config.dropout, config.num_decoder_layers,
        )
        self.action_queries = nn.Embedding(config.pred_horizon, D)
        self.action_head    = nn.Linear(D, config.action_dim)

        self.norm_stats: dict = {}
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.action_queries.weight, std=0.02)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        obs_encoder: ObsEncoder,
        config: ACTConfig,
    ) -> "ACT":
        return cls(obs_encoder=obs_encoder, config=config)

    # ------------------------------------------------------------------
    # Observation token sequence
    # ------------------------------------------------------------------

    # Modalities to mask during cond_mask_prob dropout.
    # Instruction and proprioception are kept so the model must learn
    # to use them when visual info is absent.
    _VISUAL_KEYS = frozenset({"rgb", "depth"})

    def _obs_tokens(
        self, obs: ObsFeatures, apply_cond_mask: bool = False,
    ) -> Tensor:
        """
        Build a token sequence from per-modality features.

        Each modality contributes ``obs_horizon`` tokens.  All tokens share
        the same ``hidden_dim`` via modality-specific linear projections.

        When ``apply_cond_mask=True`` and ``cond_mask_prob > 0``, visual
        tokens (rgb/depth) are randomly zeroed per-sample to force the
        model to rely on instruction conditioning.

        Returns: (B, obs_horizon * num_modalities, hidden_dim)
        """
        token_groups = []
        B = None
        for key, proj in self.modality_projs.items():
            feat = obs.by_modality.get(key)
            if feat is None:
                continue
            # feat: (B, T, modality_dim)  →  projected: (B, T, D)
            projected = proj(feat)
            if B is None:
                B = projected.shape[0]

            # Condition masking: zero visual tokens with probability cond_mask_prob
            if (apply_cond_mask and self.config.cond_mask_prob > 0
                    and key in self._VISUAL_KEYS):
                mask = torch.rand(B, device=projected.device) < self.config.cond_mask_prob
                projected = projected * (~mask)[:, None, None].float()

            token_groups.append(projected)

        # Stack along sequence dimension: (B, T * M, D)
        return torch.cat(token_groups, dim=1)

    # ------------------------------------------------------------------
    # CVAE encoding
    # ------------------------------------------------------------------

    def _encode_cvae(
        self, obs_tokens: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Encode (obs_tokens, actions) → (z, mean, logvar).

        Args:
            obs_tokens: (B, seq_len, D) per-modality obs sequence.
            actions:    (B, pred_horizon, action_dim) ground-truth actions.
        """
        B = obs_tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        act = self.action_proj(actions.float())          # (B, A, D)
        seq = torch.cat([cls, obs_tokens, act], dim=1)  # (B, 1+seq+A, D)

        out    = self.cvae_encoder(seq)
        cls_out = out[:, 0]                              # (B, D) — CLS token
        mean   = self.mean_proj(cls_out)
        logvar = self.logvar_proj(cls_out)
        std    = torch.exp(0.5 * logvar)
        z      = mean + std * torch.randn_like(std)

        return z, mean, logvar

    # ------------------------------------------------------------------
    # Policy forward
    # ------------------------------------------------------------------

    def _policy_forward(self, obs_tokens: Tensor, z: Tensor) -> Tensor:
        """
        Predict actions from obs tokens and latent z.

        Args:
            obs_tokens: (B, seq_len, D)
            z:          (B, latent_dim)

        Returns: (B, pred_horizon, action_dim)
        """
        B = obs_tokens.shape[0]
        z_tok = self.latent_proj(z).unsqueeze(1)         # (B, 1, D)
        src   = torch.cat([z_tok, obs_tokens], dim=1)    # (B, 1+seq, D)
        src   = self.pos_enc(src)

        memory  = self.policy_enc(src)                   # (B, 1+seq, D)

        queries = self.action_queries.weight              # (pred_horizon, D)
        queries = self.pos_enc(queries.unsqueeze(0).expand(B, -1, -1))
        out     = self.policy_dec(queries, memory)        # (B, pred_horizon, D)

        return self.action_head(out)                     # (B, pred_horizon, action_dim)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        obs      = self.obs_encoder(batch)               # ObsFeatures
        tokens   = self._obs_tokens(obs, apply_cond_mask=self.training)
        actions  = batch["action"].float()               # (B, A, action_dim)

        z, mean, logvar = self._encode_cvae(tokens, actions)

        action_pred = self._policy_forward(tokens, z)    # (B, A, action_dim)

        l2_loss = F.mse_loss(action_pred, actions)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        total   = l2_loss + self.config.kl_weight * kl_loss

        metrics = {
            "loss":     total,
            "l2_loss":  l2_loss.detach(),
            "kl_loss":  kl_loss.detach(),
            "arm_mse":  F.mse_loss(action_pred[:, :, :6],  actions[:, :, :6]).detach(),
            "hand_mse": F.mse_loss(action_pred[:, :, 6:],  actions[:, :, 6:]).detach(),
        }
        metrics.update(self._unnorm_action_mse(action_pred.detach(), actions))
        return metrics

    # ------------------------------------------------------------------
    # Validation (deterministic forward, z=0)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Validation via deterministic forward pass (z=0, prior mean).

        Measures MSE between predicted and ground-truth actions in both
        normalized and unnormalized space.
        """
        obs     = self.obs_encoder(batch)
        tokens  = self._obs_tokens(obs)
        actions = batch["action"].float()
        B       = tokens.shape[0]
        z       = torch.zeros(B, self.config.latent_dim, device=tokens.device)

        pred = self._policy_forward(tokens, z)

        loss = F.mse_loss(pred, actions)
        result: dict[str, Tensor] = {
            "loss":     loss,
            "arm_mse":  F.mse_loss(pred[:, :, :6],  actions[:, :, :6]).detach(),
            "hand_mse": F.mse_loss(pred[:, :, 6:],  actions[:, :, 6:]).detach(),
        }
        result.update(self._unnorm_action_mse(pred, actions))
        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Deterministic inference: z = 0 (prior mean)."""
        obs    = self.obs_encoder(batch)
        tokens = self._obs_tokens(obs)
        B      = tokens.shape[0]
        z      = torch.zeros(B, self.config.latent_dim, device=tokens.device)

        actions = self._policy_forward(tokens, z)        # (B, pred_horizon, D)
        actions = actions[:, : self.config.action_horizon]

        if self.norm_stats:
            actions = self._unnormalize(actions)
        return actions

    def _unnormalize(self, action: Tensor) -> Tensor:
        stat = self.norm_stats.get("action", {})
        if not stat:
            return action
        mn = torch.tensor(stat["min"], device=action.device, dtype=action.dtype)
        mx = torch.tensor(stat["max"], device=action.device, dtype=action.dtype)
        return (action + 1) / 2 * (mx - mn + 1e-8) + mn

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(
        self, lr: float = 1e-4, weight_decay: float = 1e-4
    ) -> list[torch.optim.Optimizer]:
        # Lower lr for pre-trained backbones inside the obs_encoder.
        backbone_params, other_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "obs_encoder" in name and "backbone" in name:
                backbone_params.append(p)
            else:
                other_params.append(p)

        return [
            torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": lr * 0.1},
                    {"params": other_params,    "lr": lr},
                ],
                weight_decay=weight_decay,
            )
        ]
