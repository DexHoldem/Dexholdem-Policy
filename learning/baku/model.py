"""
BAKU — native Dexas adaptation of the BAKU action-token policy.

This implementation keeps the core idea from the original BAKU paper/code:

- build per-step observation tokens
- append a learned action token for each observation timestep
- run a causal Transformer over the token sequence
- decode actions from the action-token states

Differences from the original repo:

- uses this repo's canonical batch dict and normalization flow
- reuses the existing ObsEncoder sub-encoders instead of BAKU's agent stack
- predicts a chunked action sequence `(pred_horizon, action_dim)` so it fits
  the rest of Dexas-Policy
- only implements the deterministic head
- does not implement prompt demonstrations or runtime temporal aggregation
- phase 2 adds optional FiLM conditioning for ResNet-based visual branches
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from learning.base import ModelConfig, PolicyModel
from learning.common.encoders import ObsEncoder
from learning.registry import register_model


@dataclass
class BakuConfig(ModelConfig):
    """Hyper-parameters for the native BAKU variant."""

    hidden_size: int = 256
    depth: int = 8
    num_heads: int = 4
    ff_dim: int = 0
    dropout: float = 0.1
    use_film: bool = True
    film_stage_channels: tuple[int, int, int, int] = (64, 128, 256, 512)


def _make_transformer(
    d_model: int,
    nhead: int,
    ffn_dim: int,
    dropout: float,
    n_layers: int,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class _ResNetFiLM(nn.Module):
    """Zero-init FiLM adapters for the four ResNet stages."""

    def __init__(self, cond_dim: int, stage_channels: tuple[int, int, int, int]):
        super().__init__()
        self.stage_channels = tuple(int(c) for c in stage_channels)
        self.projs = nn.ModuleList([
            nn.Linear(cond_dim, channels * 2)
            for channels in self.stage_channels
        ])
        for proj in self.projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def apply_to(self, x: Tensor, cond: Tensor, stage_idx: int) -> Tensor:
        beta_gamma = self.projs[stage_idx](cond).reshape(cond.shape[0], -1, 1, 1)
        beta, gamma = torch.chunk(beta_gamma, 2, dim=1)
        return (1.0 + gamma) * x + beta


@register_model("baku")
class Baku(PolicyModel):
    """
    Native BAKU adaptation for Dexas-Policy.

    Token layout per sample:
      [task_token?] +
      for each obs timestep:
        [rgb_cam tokens] [depth_cam tokens] [state tokens] [action_token]

    The final action token summarizes all available context and is decoded into
    the full predicted action chunk.
    """

    _UNSUPPORTED_RGB_ENCODERS = {"dinov2_vitl14_patch"}

    def __init__(self, obs_encoder: ObsEncoder, config: BakuConfig):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.config = config
        self.norm_stats: dict = {}
        # Optional compatibility knob for checkpoints whose deploy-time camera
        # stream has the opposite channel order from the training data.
        self._swap_rgb_channels_on_predict = False

        enc_cfg = obs_encoder.config
        if enc_cfg.rgb_encoder_type in self._UNSUPPORTED_RGB_ENCODERS:
            raise ValueError(
                "BAKU does not support patch-token RGB encoders "
                f"({enc_cfg.rgb_encoder_type}). Use vector encoders such as "
                "'resnet18', 'dinov2_vitl14', or 'siglip_so400m'."
            )
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_heads ({config.num_heads})."
            )

        self.num_cams = len(enc_cfg.camera_indices)
        self.use_fused_rgbd = obs_encoder.rgbd_encoders is not None
        self.has_visual_branch = bool(
            self.use_fused_rgbd
            or obs_encoder.rgb_encoders is not None
            or obs_encoder.depth_encoders is not None
        )
        self.film_requested = bool(config.use_film)
        self.use_film = bool(
            self.film_requested
            and self.has_visual_branch
            and obs_encoder.instruction_encoder is not None
        )
        self.film_cond_dim = enc_cfg.instruction_embed_dim

        if (self.film_requested
                and self.has_visual_branch
                and enc_cfg.precompute_rgb_features
                and "img" in enc_cfg.representation_type):
            raise ValueError(
                "BAKU FiLM does not support precomputed RGB features because FiLM "
                "must modulate the live visual encoder."
            )
        if (self.film_requested and self.has_visual_branch and not self.use_fused_rgbd
                and obs_encoder.rgb_encoders is not None
                and enc_cfg.rgb_encoder_type != "resnet18"):
            raise ValueError(
                "BAKU FiLM currently supports only ResNet18-based RGB visual branches. "
                f"Got rgb_encoder_type={enc_cfg.rgb_encoder_type!r}."
            )

        hidden = config.hidden_size
        ff_dim = config.ff_dim if config.ff_dim > 0 else 4 * hidden

        self.task_proj: nn.Module | None = None
        if obs_encoder.instruction_encoder is not None:
            self.task_proj = nn.Sequential(
                nn.Linear(enc_cfg.instruction_embed_dim, hidden),
                nn.LayerNorm(hidden),
            )

        self.visual_proj: nn.Module | None = None
        self.rgb_proj: nn.Module | None = None
        self.depth_proj: nn.Module | None = None
        self.visual_film: _ResNetFiLM | None = None
        self.rgb_film: _ResNetFiLM | None = None
        self.depth_film: _ResNetFiLM | None = None

        if self.use_fused_rgbd:
            visual_stage_channels = self._resolve_film_stage_channels(
                obs_encoder.rgbd_encoders,
                config.film_stage_channels,
            )
            fused_dim = enc_cfg.rgb_per_cam_output + enc_cfg.depth_per_cam_output
            self.visual_proj = nn.Sequential(
                nn.Linear(fused_dim, hidden),
                nn.LayerNorm(hidden),
            )
            self.visual_camera_embed = nn.Embedding(self.num_cams, hidden)
            if self.use_film:
                self.visual_film = _ResNetFiLM(self.film_cond_dim, visual_stage_channels)
        else:
            rgb_stage_channels = self._resolve_film_stage_channels(
                obs_encoder.rgb_encoders,
                config.film_stage_channels,
            )
            depth_stage_channels = self._resolve_film_stage_channels(
                obs_encoder.depth_encoders,
                config.film_stage_channels,
            )
            if obs_encoder.rgb_encoders is not None:
                self.rgb_proj = nn.Sequential(
                    nn.Linear(enc_cfg.rgb_per_cam_output, hidden),
                    nn.LayerNorm(hidden),
                )
                self.rgb_camera_embed = nn.Embedding(self.num_cams, hidden)
                if self.use_film:
                    self.rgb_film = _ResNetFiLM(self.film_cond_dim, rgb_stage_channels)
            if obs_encoder.depth_encoders is not None:
                self.depth_proj = nn.Sequential(
                    nn.Linear(enc_cfg.depth_per_cam_output, hidden),
                    nn.LayerNorm(hidden),
                )
                self.depth_camera_embed = nn.Embedding(self.num_cams, hidden)
                if self.use_film:
                    self.depth_film = _ResNetFiLM(self.film_cond_dim, depth_stage_channels)

        self.state_projs = nn.ModuleDict({
            key: nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden))
            for key, dim in obs_encoder.modality_dims.items()
            if key not in {"rgb", "depth", "instruction"}
        })

        self.action_token = nn.Parameter(torch.zeros(1, 1, hidden))

        tokens_per_step = self._tokens_per_step()
        max_seq_len = config.obs_horizon * tokens_per_step + (1 if self.task_proj is not None else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden))
        self.transformer = _make_transformer(
            d_model=hidden,
            nhead=config.num_heads,
            ffn_dim=ff_dim,
            dropout=config.dropout,
            n_layers=config.depth,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.pred_horizon * config.action_dim),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.action_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @classmethod
    def build(cls, obs_encoder: ObsEncoder, config: BakuConfig) -> "Baku":
        return cls(obs_encoder=obs_encoder, config=config)

    def _tokens_per_step(self) -> int:
        count = 1  # action token
        if self.use_fused_rgbd:
            count += self.num_cams
        else:
            if self.rgb_proj is not None:
                count += self.num_cams
            if self.depth_proj is not None:
                count += self.num_cams
        count += len(self.state_projs)
        return count

    @staticmethod
    def _infer_batch_shape(batch: dict[str, Tensor]) -> tuple[int, int]:
        sample = next(v for v in batch.values() if isinstance(v, Tensor) and v.dim() >= 2)
        return int(sample.shape[0]), int(sample.shape[1])

    def _encode_task_features(self, batch: dict[str, Tensor], batch_size: int) -> Tensor | None:
        if self.obs_encoder.instruction_encoder is None:
            return None
        if "instruction" in batch:
            return self.obs_encoder.instruction_encoder(batch["instruction"])
        device = next(self.obs_encoder.parameters()).device
        return torch.zeros(
            batch_size,
            self.obs_encoder.config.instruction_embed_dim,
            device=device,
        )

    def _encode_task_token(self, task_features: Tensor | None) -> Tensor | None:
        if task_features is None or self.task_proj is None:
            return None
        return self.task_proj(task_features)

    @staticmethod
    def _resolve_film_stage_channels(
        encoders: nn.ModuleList | None,
        fallback: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        if encoders is not None and len(encoders) > 0:
            channels = getattr(encoders[0], "film_stage_channels", None)
            if channels is not None:
                return tuple(int(c) for c in channels)
        return tuple(int(c) for c in fallback)

    @staticmethod
    def _expand_task_features(task_features: Tensor | None, t_obs: int) -> Tensor | None:
        if task_features is None:
            return None
        return task_features[:, None, :].expand(-1, t_obs, -1).reshape(-1, task_features.shape[-1])

    def _run_resnet_with_film(
        self,
        backbone: nn.Module,
        x: Tensor,
        film: _ResNetFiLM | None,
        cond: Tensor | None,
    ) -> Tensor:
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        stage_modules = getattr(backbone, "stage_modules", None)
        if stage_modules is None:
            stage_modules = [
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            ]

        for stage_idx, block in enumerate(stage_modules):
            x = block(x)
            if film is not None and cond is not None:
                x = film.apply_to(x, cond, stage_idx)

        if getattr(backbone, "projection_type", "avgpool") == "avgpool":
            x = backbone.avgpool(x)
            x = torch.flatten(x, 1)
        return x

    def _crop_resnet_image(self, img: Tensor) -> Tensor:
        img = F.interpolate(img, size=(240, 320), mode="bilinear", align_corners=False)
        if self.training:
            i = torch.randint(0, 240 - 216 + 1, (1,), device=img.device).item()
            j = torch.randint(0, 320 - 288 + 1, (1,), device=img.device).item()
            img = img[:, :, i:i + 216, j:j + 288]
        else:
            img = img[:, :, 12:228, 16:304]
        return img

    def _encode_fused_visual_tokens(
        self,
        batch: dict[str, Tensor],
        batch_size: int,
        t_obs: int,
        task_features: Tensor | None,
    ) -> list[Tensor]:
        if self.visual_proj is None or self.obs_encoder.rgbd_encoders is None:
            return []
        if "rgb" not in batch or "depth" not in batch:
            return []

        task_bt = self._expand_task_features(task_features, t_obs)
        tokens: list[Tensor] = []
        for ci, enc in enumerate(self.obs_encoder.rgbd_encoders):
            rgb = batch["rgb"][:, :, ci]
            dep = batch["depth"][:, :, ci]
            rgb_flat = rgb.reshape(batch_size * t_obs, *rgb.shape[2:])
            dep_flat = dep.reshape(batch_size * t_obs, *dep.shape[2:])
            if self.visual_film is None or task_bt is None:
                feat = enc(rgb_flat, dep_flat).reshape(batch_size, t_obs, -1)
            else:
                h, w = rgb_flat.shape[-3], rgb_flat.shape[-2]
                rgb_f = rgb_flat.reshape(-1, h, w, 3).float() / 255.0
                rgb_chw = rgb_f.permute(0, 3, 1, 2)
                dep_chw = dep_flat.reshape(-1, 1, h, w).float()
                img = torch.cat([rgb_chw, dep_chw], dim=1)
                img = self._crop_resnet_image(img)
                raw = self._run_resnet_with_film(enc.backbone, img, self.visual_film, task_bt)
                feat = enc.proj(raw).reshape(batch_size, t_obs, -1)
            cam_bias = self.visual_camera_embed.weight[ci].view(1, 1, -1)
            tokens.append(self.visual_proj(feat) + cam_bias)
        return tokens

    def _encode_rgb_tokens(
        self,
        batch: dict[str, Tensor],
        batch_size: int,
        t_obs: int,
        task_features: Tensor | None,
    ) -> list[Tensor]:
        if self.rgb_proj is None or self.obs_encoder.rgb_encoders is None:
            return []

        tokens: list[Tensor] = []
        if self.obs_encoder.config.precompute_rgb_features:
            if "rgb_features" not in batch:
                raise RuntimeError(
                    "BAKU requested precomputed RGB features but 'rgb_features' is missing from the batch."
                )
            for ci, enc in enumerate(self.obs_encoder.rgb_encoders):
                feat = batch["rgb_features"][:, :, ci]
                if feat.dim() != 3:
                    raise ValueError(
                        "BAKU only supports vector-shaped precomputed RGB features "
                        "(B, T, num_cams, feat_dim). Patch-token features are not supported."
                    )
                out = enc(feat.reshape(batch_size * t_obs, feat.shape[-1])).reshape(batch_size, t_obs, -1)
                cam_bias = self.rgb_camera_embed.weight[ci].view(1, 1, -1)
                tokens.append(self.rgb_proj(out) + cam_bias)
            return tokens

        if "rgb" not in batch:
            return []

        task_bt = self._expand_task_features(task_features, t_obs)
        for ci, enc in enumerate(self.obs_encoder.rgb_encoders):
            img = batch["rgb"][:, :, ci]
            img_flat = img.reshape(batch_size * t_obs, *img.shape[2:])
            if self.rgb_film is None or task_bt is None:
                img_chw = img_flat.float().permute(0, 3, 1, 2)
                img_chw = self.obs_encoder._preprocess(img_chw)
                img_hw3 = img_chw.permute(0, 2, 3, 1)
                out = enc(img_hw3).reshape(batch_size, t_obs, -1)
            else:
                h, w = img_flat.shape[-3], img_flat.shape[-2]
                img_chw = img_flat.reshape(-1, h, w, 3).float().permute(0, 3, 1, 2)
                img_chw = self.obs_encoder._preprocess(img_chw)
                img_chw = self._crop_resnet_image(img_chw / 255.0)
                raw = self._run_resnet_with_film(enc.backbone, img_chw, self.rgb_film, task_bt)
                out = enc.proj(raw).reshape(batch_size, t_obs, -1)
            cam_bias = self.rgb_camera_embed.weight[ci].view(1, 1, -1)
            tokens.append(self.rgb_proj(out) + cam_bias)
        return tokens

    def _encode_depth_tokens(
        self,
        batch: dict[str, Tensor],
        batch_size: int,
        t_obs: int,
        task_features: Tensor | None,
    ) -> list[Tensor]:
        if self.depth_proj is None or self.obs_encoder.depth_encoders is None or "depth" not in batch:
            return []

        task_bt = self._expand_task_features(task_features, t_obs)
        tokens: list[Tensor] = []
        for ci, enc in enumerate(self.obs_encoder.depth_encoders):
            dep = batch["depth"][:, :, ci]
            dep_flat = dep.reshape(batch_size * t_obs, *dep.shape[2:]).float()
            if self.depth_film is None or task_bt is None:
                out = enc(dep_flat).reshape(batch_size, t_obs, -1)
            else:
                h, w = dep_flat.shape[-2], dep_flat.shape[-1]
                img = dep_flat.reshape(-1, 1, h, w)
                img = self._crop_resnet_image(img)
                raw = self._run_resnet_with_film(enc.backbone, img, self.depth_film, task_bt)
                out = enc.proj(raw).reshape(batch_size, t_obs, -1)
            cam_bias = self.depth_camera_embed.weight[ci].view(1, 1, -1)
            tokens.append(self.depth_proj(out) + cam_bias)
        return tokens

    def _encode_state_tokens(self, batch: dict[str, Tensor], batch_size: int, t_obs: int) -> dict[str, Tensor]:
        tokens: dict[str, Tensor] = {}
        for key, proj in self.state_projs.items():
            if key not in batch:
                continue
            enc = self.obs_encoder.state_encoders[key]
            feat = enc(batch[key].float().reshape(batch_size * t_obs, -1)).reshape(batch_size, t_obs, -1)
            tokens[key] = proj(feat)
        return tokens

    def _tokenize(self, batch: dict[str, Tensor]) -> tuple[Tensor, list[int]]:
        batch_size, t_obs = self._infer_batch_shape(batch)
        task_features = self._encode_task_features(batch, batch_size)
        task_tok = self._encode_task_token(task_features)
        fused_tokens = self._encode_fused_visual_tokens(batch, batch_size, t_obs, task_features)
        rgb_tokens = self._encode_rgb_tokens(batch, batch_size, t_obs, task_features)
        depth_tokens = self._encode_depth_tokens(batch, batch_size, t_obs, task_features)
        state_tokens = self._encode_state_tokens(batch, batch_size, t_obs)

        seq: list[Tensor] = []
        action_positions: list[int] = []
        if task_tok is not None:
            seq.append(task_tok.unsqueeze(1))

        for t in range(t_obs):
            for tok in fused_tokens:
                seq.append(tok[:, t:t + 1])
            for tok in rgb_tokens:
                seq.append(tok[:, t:t + 1])
            for tok in depth_tokens:
                seq.append(tok[:, t:t + 1])
            for key in self.state_projs:
                tok = state_tokens.get(key)
                if tok is not None:
                    seq.append(tok[:, t:t + 1])
            seq.append(self.action_token.expand(batch_size, -1, -1))
            action_positions.append(len(seq) - 1)

        x = torch.cat(seq, dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]
        return x, action_positions

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> Tensor:
        mask = torch.full((size, size), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def _predict_normalized(self, batch: dict[str, Tensor]) -> Tensor:
        tokens, action_positions = self._tokenize(batch)
        mask = self._causal_mask(tokens.shape[1], tokens.device)
        hidden = self.transformer(tokens, mask=mask)
        summary = hidden[:, action_positions[-1], :]
        action = self.head(summary)
        return action.view(-1, self.config.pred_horizon, self.config.action_dim)

    def enable_deploy_rgb_channel_swap(self, enabled: bool = True) -> None:
        """Enable a deploy-only RGB<->BGR channel swap for predict_action()."""
        self._swap_rgb_channels_on_predict = bool(enabled)

    def _prepare_predict_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self._swap_rgb_channels_on_predict or "rgb" not in batch:
            return batch
        # Reverse the last channel axis: RGB <-> BGR.
        prepared = dict(batch)
        prepared["rgb"] = batch["rgb"].flip(dims=(-1,))
        return prepared

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        pred = self._predict_normalized(batch)
        target = batch["action"].float()
        loss = F.mse_loss(pred, target)
        metrics = {
            "loss": loss,
            "mse": loss.detach(),
            "arm_mse": F.mse_loss(pred[:, :, :6], target[:, :, :6]).detach(),
            "hand_mse": F.mse_loss(pred[:, :, 6:], target[:, :, 6:]).detach(),
        }
        metrics.update(self._unnorm_action_mse(pred.detach(), target))
        return metrics

    @torch.no_grad()
    def predict_action(self, batch: dict[str, Tensor]) -> Tensor:
        batch = self._prepare_predict_batch(batch)
        action = self._predict_normalized(batch)
        action = action[:, : self.config.action_horizon]
        stat = self.norm_stats.get("action", {})
        if not stat:
            return action
        mn = torch.tensor(stat["min"], device=action.device, dtype=action.dtype)
        mx = torch.tensor(stat["max"], device=action.device, dtype=action.dtype)
        return (action + 1) / 2 * (mx - mn + 1e-8) + mn

    def configure_optimizers(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> list[torch.optim.Optimizer]:
        backbone_params, other_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "obs_encoder" in name and "backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        param_groups = [{"params": other_params, "lr": lr}]
        if backbone_params:
            param_groups.insert(0, {"params": backbone_params, "lr": lr * 0.1})
        return [
            torch.optim.AdamW(
                param_groups,
                weight_decay=weight_decay,
            )
        ]
