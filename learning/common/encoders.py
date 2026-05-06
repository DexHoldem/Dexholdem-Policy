"""
Observation encoder: converts raw sensor data into structured features.

Design
------
Raw sensors  →  per-modality sub-encoders  →  ObsFeatures
                 (RGBEncoder, DepthEncoder,
                  StateEncoder, InstructionEncoder)

The key design decision is that ``ObsEncoder.forward()`` returns an
``ObsFeatures`` object rather than a flat tensor.  This lets each policy
model choose its own fusion strategy:

  obs.flat()        → (B, T, D)     — concatenate all modalities per timestep
  obs.flat_time()   → (B, T*D)      — also flatten time  (UNet global_cond)
  obs.by_modality   → dict          — per-modality tensors for custom fusion

Custom encoders
---------------
Any nn.Module with the same interface can replace ``ObsEncoder``:

  forward(batch: dict) -> ObsFeatures
  modality_dims: dict[str, int]      # output dim per named modality
  total_dim: int                     # sum of modality_dims

This means a model can define its own encoder (e.g. raw patch tokens for a
ViT, or a sensor-specific GNN) without changing the data pipeline or the
training loop.

Supported RGB backends
----------------------
  "resnet18"        — torchvision ResNet18, BN→GN, 512-dim backbone
  "dinov2_vits14"   — DINOv2 ViT-S/14, 384 dim (frozen)
  "dinov2_vitb14"   — DINOv2 ViT-B/14, 768 dim (frozen)
  "dinov2_vitl14"   — DINOv2 ViT-L/14, 1024 dim (frozen)
  "dinov3_vitl16"   — DINOv3 ViT-L/16 from local hub (frozen)
  "siglip_so400m"   — SigLIP-SO400M ViT-L/14 @ 384px, 1152 dim (frozen)
                      Used by RDT as the primary visual backbone.
                      Requires: pip install transformers
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor


# ---------------------------------------------------------------------------
# ObsFeatures — structured encoder output
# ---------------------------------------------------------------------------

@dataclass
class ObsFeatures:
    """
    Structured output from ``ObsEncoder.forward()``.

    ``by_modality`` maps each active modality name to a tensor of shape
    ``(B, obs_horizon, modality_dim)``.  Canonical keys are:
        "rgb", "depth", "pos", "eef", "hand_pos",
        "efforts", "velocity", "touch", "instruction"

    Helper methods provide the flat views that most policy models need:

    .. code-block:: python

        obs = encoder(batch)

        obs.flat()        # (B, T, total_D)   — cat all modalities
        obs.flat_time()   # (B, T * total_D)  — also flatten time
        obs.total_dim     # scalar int

        # Per-modality access (for models like ACT that process them separately)
        rgb_feat = obs.by_modality["rgb"]   # (B, T, rgb_D)
    """

    by_modality: dict[str, Tensor]

    def flat(self) -> Tensor:
        """Concatenate all modality features → (B, T, total_D)."""
        return torch.cat(list(self.by_modality.values()), dim=-1)

    def flat_time(self) -> Tensor:
        """Flatten both modality and time → (B, T * total_D).

        Suitable as a UNet ``global_cond`` or MLP input.
        """
        f = self.flat()
        B, T, D = f.shape
        return f.reshape(B, T * D)

    @property
    def total_dim(self) -> int:
        """Sum of all modality output dimensions."""
        return sum(v.shape[-1] for v in self.by_modality.values())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ObsEncoderConfig:
    """
    Specifies which sensors to encode and how.

    This config is owned by ``ObsEncoder``.  Policy models receive a fully
    constructed ``ObsEncoder`` and query ``obs_encoder.modality_dims`` or
    ``obs_encoder.total_dim`` to size their own layers.
    """

    # Which modalities to include.  Each string must be one of:
    # "img", "depth", "pos", "eef", "hand_pos", "efforts", "velocity", "touch"
    representation_type: list[str] = field(default_factory=lambda: ["img", "pos"])

    # Camera indices to include (0-indexed, must exist in the dataset).
    camera_indices: list[int] = field(default_factory=lambda: [0, 1, 2])

    # --- Visual encoder backends ---
    rgb_encoder_type: str = "resnet18"   # see module docstring for options
    depth_encoder_type: str = "resnet18"
    freeze_rgb_encoder: bool = False
    freeze_depth_encoder: bool = False

    # When True the dataset pre-computes DinoV2/V3 features offline and
    # stores them in batch["rgb_features"].  The encoder then applies only
    # a linear projection, skipping the heavy ViT backbone at training time.
    precompute_rgb_features: bool = False

    # Fuse RGB (3ch) + depth (1ch) into a single 4-channel ResNet18 per camera.
    # Halves the number of backbone passes (3 RGBD vs 3 RGB + 3 depth).
    # Only works with ResNet18 encoder.  When True, depth_encoders are not created.
    fuse_rgbd: bool = False

    # --- Output dimensions ---
    # Visual: per camera (total = per_cam * len(camera_indices))
    rgb_per_cam_output: int = 96
    depth_per_cam_output: int = 32

    # Proprioception: per modality
    pos_output_size: int = 128
    eef_output_size: int = 32
    hand_pos_output_size: int = 96
    efforts_output_size: int = 64
    velocity_output_size: int = 64
    touch_output_size: int = 64

    # Instruction embedding (multi-task)
    # --- Integer-ID mode (default) ---
    use_instruction: bool = False
    num_instructions: int = 14
    instruction_embed_dim: int = 128

    # --- Text instruction mode (alternative to use_instruction) ---
    # When True, instruction IDs are looked up in a text encoder pre-seeded
    # from CLIP / sentence-transformers at init time.  The output dim is still
    # instruction_embed_dim (shared with the integer-ID mode).
    use_text_instruction: bool = False
    # "clip" → openai/clip-vit-base-patch32  (512-d pooler output)
    # "clip_large" → openai/clip-vit-large-patch14 (768-d pooler output)
    # "sentence_transformers" → all-MiniLM-L6-v2 (384-d)
    text_encoder_type: str = "clip"
    # Path to JSON with {"0": {"text": "..."}, ...}.
    # Only needed during training; deploy_policy.py loads embeddings from ckpt.
    instructions_file: str = ""

    # --- Image pre-processing (applied inside the encoder) ---
    # Both off by default: images are kept at original 240×320 (H×W).
    # Each backbone resizes internally (DinoV2 → 224, SigLIP → 384, ResNet18 → 240×320 + crop).
    enable_downsample: bool = False
    downsample_size: tuple[int, int] = (240, 320)   # (H, W)
    enable_crop: bool = False
    crop_size: tuple[int, int] = (216, 288)         # (H, W)

    # --- Deploy-time depth resize ---
    # Set by deploy_policy.py to resize camera depth frames to the training
    # resolution before feeding to the depth backbone.  None during training
    # (data is already at the correct resolution).
    deploy_depth_size: tuple[int, int] | None = None


# ---------------------------------------------------------------------------
# Backbone helpers
# ---------------------------------------------------------------------------

def _make_resnet18(in_channels: int = 3) -> nn.Module:
    """ResNet18 with BatchNorm replaced by GroupNorm and no final FC layer."""
    net = torchvision.models.resnet18(weights=None)
    net.fc = nn.Identity()
    if in_channels != 3:
        net.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    _replace_bn_with_gn(net)
    return net


def _replace_bn_with_gn(module: nn.Module, num_groups: int = 8) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(
                num_groups=min(num_groups, child.num_features),
                num_channels=child.num_features,
            )
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child, num_groups)


# ---------------------------------------------------------------------------
# Per-modality sub-encoders
# ---------------------------------------------------------------------------

class RGBEncoder(nn.Module):
    """
    Encodes a single camera's RGB (or pre-computed feature) to a 1-D vector.

    When ``precompute=True`` the input is a pre-computed backbone feature
    vector and only a linear projection is applied (the heavy ViT backbone
    runs offline).
    """

    # Raw feature dimension of each frozen backbone (CLS / pooled output).
    _BACKBONE_DIMS: dict[str, int] = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
        "dinov2_vitl14_patch": 1024,  # same backbone, patch tokens instead of CLS
        "siglip_so400m": 1152,   # google/siglip-so400m-patch14-384, pooler_output
    }
    # Keep the old name as an alias for backwards-compat with serialised configs.
    _DINOV2_DIMS = _BACKBONE_DIMS

    def __init__(
        self,
        encoder_type: str,
        output_size: int,
        freeze: bool = False,
        precompute: bool = False,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.output_size = output_size
        self.precompute = precompute

        if precompute and encoder_type == "resnet18":
            raise ValueError(
                "precompute_rgb_features is only valid for frozen ViT/SigLIP "
                "features. ResNet18 must receive raw RGB images; remove "
                "--precompute_rgb_features and any feature_dir/feature_dirs."
            )

        if precompute:
            raw_dim = self._BACKBONE_DIMS.get(encoder_type, 1024)
            # Bare Linear (no LayerNorm) to match TexasPoker's precomputed
            # feature projection: nn.Linear(dino_feature_dim, 96).
            self.proj = nn.Linear(raw_dim, output_size)
            self.backbone = None
        elif encoder_type == "resnet18":
            self.backbone = _make_resnet18(in_channels=3)
            self.proj = nn.Linear(512, output_size)
        elif encoder_type.startswith("dinov2"):
            # dinov2_vitl14_patch reuses the same backbone as dinov2_vitl14
            hub_name = encoder_type.replace("_patch", "")
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", hub_name, pretrained=True
            )
            raw_dim = self._DINOV2_DIMS[encoder_type]
            self.proj = nn.Sequential(
                nn.Linear(raw_dim, output_size),
                nn.LayerNorm(output_size),
            )
        elif encoder_type.startswith("dinov3"):
            import os
            repo = os.environ.get("DINOV3_REPO", "/home/user/dinov3")
            self.backbone = torch.hub.load(
                repo, encoder_type, source="local", pretrained=True
            )
            self.proj = nn.Sequential(
                nn.Linear(1024, output_size),
                nn.LayerNorm(output_size),
            )
        elif encoder_type == "siglip_so400m":
            try:
                from transformers import SiglipVisionModel
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for SigLIP. "
                    "Install with: pip install transformers"
                ) from exc
            self.backbone = SiglipVisionModel.from_pretrained(
                "google/siglip-so400m-patch14-384", use_safetensors=True
            )
            raw_dim = self._BACKBONE_DIMS["siglip_so400m"]   # 1152
            self.proj = nn.Sequential(
                nn.Linear(raw_dim, output_size),
                nn.LayerNorm(output_size),
            )
        else:
            raise ValueError(f"Unknown rgb_encoder_type: {encoder_type!r}")

        if freeze and self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def attach_backbone(self):
        """Load a pretrained backbone for deploy-time on-the-fly inference.

        Called by deploy_policy.py when the checkpoint was trained with
        precomputed features.  The backbone is frozen and only used to
        extract features at inference time — the proj layer (bare Linear)
        stays exactly as trained.
        """
        if self.backbone is not None:
            return  # already present
        if self.encoder_type == "resnet18":
            self.backbone = _make_resnet18(in_channels=3)
        elif self.encoder_type.startswith("dinov2"):
            hub_name = self.encoder_type.replace("_patch", "")
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", hub_name, pretrained=True
            )
        elif self.encoder_type.startswith("dinov3"):
            import os
            repo = os.environ.get("DINOV3_REPO", "/home/user/dinov3")
            self.backbone = torch.hub.load(
                repo, self.encoder_type, source="local", pretrained=True
            )
        elif self.encoder_type == "siglip_so400m":
            from transformers import SiglipVisionModel
            self.backbone = SiglipVisionModel.from_pretrained(
                "google/siglip-so400m-patch14-384", use_safetensors=True
            )
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor, crop_ij: tuple[int, int] | None = None) -> Tensor:
        """
        Args:
            x: precompute=False → (*, H, W, 3) uint8
               precompute=True  → (*, raw_dim) float pre-computed feature
               precompute=True + backbone attached (deploy) → (*, H, W, 3)
            crop_ij: Optional (i, j) crop offsets for spatial alignment with
                     depth encoder.  Generated by ObsEncoder for each camera.

        Returns: (*, output_size)
        """
        if self.precompute and self.backbone is None:
            if x.dim() > 2:
                raise RuntimeError(
                    f"RGBEncoder has precompute=True but received a {x.dim()}-D tensor "
                    f"(shape {tuple(x.shape)}). Expected a 2-D precomputed feature tensor "
                    f"(N, raw_dim). Pass --precompute_rgb_features only when rgb_features "
                    "are already in the batch."
                )
            return self.proj(x.float())

        leading = x.shape[:-3]
        h, w, _ = x.shape[-3], x.shape[-2], x.shape[-1]
        img = x.reshape(-1, h, w, 3).float()
        img = img.permute(0, 3, 1, 2)  # (N, 3, H, W) in [0, 255]

        if self.encoder_type == "siglip_so400m":
            img = img / 255.0                     # [0,255] → [0, 1]
            img = F.interpolate(img, size=(384, 384), mode="bilinear", align_corners=False)
            img = (img - 0.5) / 0.5              # [0,1] → [-1, 1] (SigLIP norm)
            feat = self.backbone(pixel_values=img).pooler_output   # (N, 1152)
        elif self.encoder_type.startswith("dino"):
            img = img / 255.0                     # [0,255] → [0, 1]
            img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
            feat = self.backbone.forward_features(img)
            if self.encoder_type.endswith("_patch"):
                # Extract patch tokens (N, N_patches, D) instead of CLS
                if isinstance(feat, dict):
                    feat = feat["x_norm_patchtokens"]  # (N, N_patches, D)
                elif feat.dim() == 3:
                    feat = feat[:, 1:]  # drop CLS → (N, N_patches, D)
            else:
                if isinstance(feat, dict):
                    feat = feat["x_norm_clstoken"]  # (N, D)
                elif feat.dim() == 3:
                    feat = feat[:, 0]  # CLS token
        elif self.encoder_type == "resnet18":
            img = (img - 128.0) / 128.0           # [0,255] → [-1, 1] (match TexasPoker)
            img = F.interpolate(img, size=(240, 320), mode="bilinear", align_corners=False)
            if self.training:
                if crop_ij is not None:
                    i, j = crop_ij
                else:
                    i = torch.randint(0, 240 - 216 + 1, (1,)).item()
                    j = torch.randint(0, 320 - 288 + 1, (1,)).item()
                img = img[:, :, i:i+216, j:j+288]
            else:
                # Center crop to 216×288
                img = img[:, :, 12:228, 16:304]
            feat = self.backbone(img)
        else:
            img = img / 255.0                     # [0,255] → [0, 1]
            feat = self.backbone.forward_features(img)
            if isinstance(feat, dict):
                feat = feat["x_norm_clstoken"]
            elif feat.dim() == 3:
                feat = feat[:, 0]  # CLS token

        return self.proj(feat).reshape(*leading, self.output_size)


class DepthEncoder(nn.Module):
    """Encodes a single camera's depth image (1-channel) to a 1-D vector.

    Depth is normalized to [-1, 1] using data-driven min/max bounds when
    available (set via :meth:`set_depth_stats`), or falls back to the
    legacy hardcoded 16-bit center (mean=35767, std=35767).
    """

    # Legacy fallback for checkpoints that don't have data-driven stats.
    _FALLBACK_MEAN = 35767.0
    _FALLBACK_STD = 35767.0

    def __init__(self, output_size: int, freeze: bool = False):
        super().__init__()
        self.output_size = output_size
        self.backbone = _make_resnet18(in_channels=1)
        self.proj = nn.Linear(512, output_size)
        # Data-driven bounds (set by ObsEncoder from norm_stats["depth"]).
        self.depth_min: float | None = None
        self.depth_max: float | None = None
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def set_depth_stats(self, d_min: float, d_max: float) -> None:
        """Set data-driven depth normalization bounds."""
        self.depth_min = d_min
        self.depth_max = d_max

    def forward(self, x: Tensor, crop_ij: tuple[int, int] | None = None) -> Tensor:
        """x: (*, H, W) float  →  (*, output_size)

        Args:
            crop_ij: Optional (i, j) crop offsets for spatial alignment with
                     RGB encoder.  Generated by ObsEncoder for each camera.
        """
        leading = x.shape[:-2]
        h, w = x.shape[-2], x.shape[-1]
        img = x.reshape(-1, 1, h, w).float()
        # Normalize depth to [-1, 1]
        if self.depth_min is not None and self.depth_max is not None:
            eps = 1e-8
            img = (img - self.depth_min) / (self.depth_max - self.depth_min + eps) * 2 - 1
        else:
            img = (img - self._FALLBACK_MEAN) / self._FALLBACK_STD
        img = F.interpolate(img, size=(240, 320), mode="bilinear", align_corners=False)
        if self.training:
            if crop_ij is not None:
                i, j = crop_ij
            else:
                i = torch.randint(0, 240 - 216 + 1, (1,)).item()
                j = torch.randint(0, 320 - 288 + 1, (1,)).item()
            img = img[:, :, i:i+216, j:j+288]
        else:
            img = img[:, :, 12:228, 16:304]
        feat = self.backbone(img)
        return self.proj(feat).reshape(*leading, self.output_size)


class RGBDEncoder(nn.Module):
    """Encodes a single camera's RGB+depth (4-channel) through one ResNet18."""

    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.backbone = _make_resnet18(in_channels=4)
        self.proj = nn.Linear(512, output_size)
        # Data-driven depth bounds (set by ObsEncoder from norm_stats).
        self.depth_min: float | None = None
        self.depth_max: float | None = None

    def set_depth_stats(self, d_min: float, d_max: float) -> None:
        """Set data-driven depth normalization bounds."""
        self.depth_min = d_min
        self.depth_max = d_max

    def forward(self, rgb: Tensor, depth: Tensor) -> Tensor:
        """
        Args:
            rgb:   (*, H, W, 3) uint8 or float
            depth: (*, H, W) float
        Returns: (*, output_size)
        """
        leading = rgb.shape[:-3]
        h, w = rgb.shape[-3], rgb.shape[-2]
        rgb_f = rgb.reshape(-1, h, w, 3).float() / 255.0
        rgb_chw = rgb_f.permute(0, 3, 1, 2)                    # (N, 3, H, W)
        dep_f = depth.reshape(-1, 1, h, w).float()              # (N, 1, H, W)
        # Normalize depth to [-1, 1] (same scale as RGB channels)
        if self.depth_min is not None and self.depth_max is not None:
            eps = 1e-8
            dep_f = (dep_f - self.depth_min) / (self.depth_max - self.depth_min + eps) * 2 - 1
        img = torch.cat([rgb_chw, dep_f], dim=1)                # (N, 4, H, W)
        img = F.interpolate(img, size=(240, 320), mode="bilinear", align_corners=False)
        if self.training:
            i = torch.randint(0, 240 - 216 + 1, (1,)).item()
            j = torch.randint(0, 320 - 288 + 1, (1,)).item()
            img = img[:, :, i:i+216, j:j+288]
        else:
            img = img[:, :, 12:228, 16:304]
        feat = self.backbone(img)
        return self.proj(feat).reshape(*leading, self.output_size)


class StateEncoder(nn.Module):
    """Two-layer MLP encoder for a single proprioceptive modality.

    Matches TexasPoker NewStateMLP: both layers have LayerNorm + ReLU,
    fixed hidden_size=256.
    """

    def __init__(self, input_dim: int, output_size: int, hidden_size: int = 256):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (*, input_dim)  →  (*, output_size)"""
        x = self.layer1(x)
        return self.layer2(x)


class InstructionEncoder(nn.Module):
    """Encodes an integer instruction ID via one-hot → Linear → LayerNorm.

    Matches TexasPoker: one-hot(num_instructions) → Linear(no bias) → LN,
    with xavier_uniform init.
    """

    def __init__(self, num_instructions: int, embed_dim: int):
        super().__init__()
        self.num_instructions = num_instructions
        self.encoder = nn.Sequential(
            nn.Linear(num_instructions, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
        )
        # Xavier init (matches TexasPoker)
        self._init_weights()

    def forward(self, ids: Tensor) -> Tensor:
        """ids: (B,) integer  →  (B, embed_dim)"""
        one_hot = F.one_hot(ids.long(), self.num_instructions).float()
        return self.encoder(one_hot)

    def _init_weights(self):
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class TextInstructionEncoder(nn.Module):
    """
    Encodes instruction IDs to dense vectors via pre-computed text embeddings.

    At construction time the instruction texts are encoded once with a frozen
    CLIP / sentence-transformers model and stored as a non-trainable buffer.
    A small trainable projection head maps the raw features to ``embed_dim``.

    At forward time the heavy text backbone is **not** called — only a buffer
    lookup and a linear projection happen, so training speed is identical to
    the integer-ID ``InstructionEncoder``.

    Args:
        num_instructions: Number of distinct instruction classes.
        embed_dim:        Output dimension (same as InstructionEncoder).
        encoder_type:     ``"clip"``, ``"clip_large"``, or
                          ``"sentence_transformers"``.
        texts:            Ordered list of instruction strings.  When ``None``
                          the buffer is zero-initialized (used when loading
                          from a checkpoint — ``load_state_dict`` fills the
                          actual values).
    """

    # Raw feature dim for each supported encoder type.
    _RAW_DIMS: dict[str, int] = {
        "clip":                  512,   # openai/clip-vit-base-patch32
        "clip_large":            768,   # openai/clip-vit-large-patch14
        "sentence_transformers": 384,   # all-MiniLM-L6-v2
    }

    _CLIP_MODELS: dict[str, str] = {
        "clip":       "openai/clip-vit-base-patch32",
        "clip_large": "openai/clip-vit-large-patch14",
    }

    def __init__(
        self,
        num_instructions: int,
        embed_dim: int,
        encoder_type: str = "clip",
        texts: Optional[list[str]] = None,
    ):
        super().__init__()
        raw_dim = self._RAW_DIMS.get(encoder_type, 512)

        if texts is not None:
            raw_features = self._encode_texts(texts, encoder_type)
        else:
            raw_features = torch.zeros(num_instructions, raw_dim)

        # Non-trainable: values come from the text backbone (or checkpoint).
        self.register_buffer("raw_embeddings", raw_features)

        # Trainable projection + norm.
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    @torch.no_grad()
    def _encode_texts(self, texts: list[str], encoder_type: str) -> Tensor:
        """Run the text backbone once and return float32 features (N, raw_dim)."""
        if encoder_type in ("clip", "clip_large"):
            try:
                from transformers import CLIPTokenizer, CLIPTextModel
            except ImportError as e:
                raise ImportError(
                    "transformers is required for text instruction encoding. "
                    "Install with: pip install transformers"
                ) from e
            model_id = self._CLIP_MODELS[encoder_type]
            print(f"[TextInstructionEncoder] Loading {model_id} …")
            tokenizer = CLIPTokenizer.from_pretrained(model_id)
            model = CLIPTextModel.from_pretrained(model_id, use_safetensors=True)
            model.eval()
            inputs = tokenizer(texts, padding=True, truncation=True,
                               return_tensors="pt")
            outputs = model(**inputs)
            # pooler_output: (N, raw_dim)
            return outputs.pooler_output.float().cpu()

        elif encoder_type == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                ) from e
            print("[TextInstructionEncoder] Loading all-MiniLM-L6-v2 …")
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            feats = st_model.encode(texts, convert_to_tensor=True,
                                    show_progress_bar=False)
            return feats.float().cpu()

        else:
            raise ValueError(
                f"Unknown text_encoder_type: {encoder_type!r}. "
                "Choose 'clip', 'clip_large', or 'sentence_transformers'."
            )

    def forward(self, ids: Tensor) -> Tensor:
        """ids: (B,)  →  (B, embed_dim)"""
        raw = self.raw_embeddings[ids]   # (B, raw_dim)
        return self.proj(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_instruction_texts(
    instructions_file: str,
    num_instructions: int,
) -> Optional[list[str]]:
    """
    Load instruction texts from a JSON file.

    Returns a list of ``num_instructions`` strings, or ``None`` if the file
    path is empty or the file does not exist (deploy-time loading from ckpt).
    """
    if not instructions_file:
        return None
    path = Path(instructions_file)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    try:
        return [data[str(i)]["text"] for i in range(num_instructions)]
    except KeyError as e:
        raise KeyError(
            f"instructions.json is missing key {e}. "
            f"Expected keys 0..{num_instructions - 1} with a 'text' field."
        ) from e


# ---------------------------------------------------------------------------
# ObsEncoder — combines all sub-encoders, returns ObsFeatures
# ---------------------------------------------------------------------------

class ObsEncoder(nn.Module):
    """
    Default observation encoder that handles all sensor modalities.

    Returns ``ObsFeatures`` rather than a flat tensor, giving each policy
    model the freedom to fuse modalities in whatever way it needs.

    To use a custom encoder for a specific model, subclass this class or
    write any ``nn.Module`` that satisfies:

    .. code-block:: python

        def forward(self, batch: dict) -> ObsFeatures: ...
        modality_dims: dict[str, int]   # property
        total_dim: int                  # property

    The data pipeline and training loop never call any other method on the
    encoder, so any module with this interface works transparently.
    """

    # Raw input dimensions for each proprioceptive modality.
    _STATE_INPUT_DIMS: dict[str, int] = {
        "pos":      30,
        "eef":      6,
        "hand_pos": 24,
        "efforts":  24,
        "velocity": 30,
        "touch":    60,
    }

    def __init__(self, config: ObsEncoderConfig):
        super().__init__()
        self.config = config

        # Build state output size map from config.
        self._state_output_sizes: dict[str, int] = {
            "pos":      config.pos_output_size,
            "eef":      config.eef_output_size,
            "hand_pos": config.hand_pos_output_size,
            "efforts":  config.efforts_output_size,
            "velocity": config.velocity_output_size,
            "touch":    config.touch_output_size,
        }

        # --- Visual encoders ---
        self.rgb_encoders: Optional[nn.ModuleList] = None
        self.depth_encoders: Optional[nn.ModuleList] = None
        self.rgbd_encoders: Optional[nn.ModuleList] = None

        use_fuse = (config.fuse_rgbd
                    and "img" in config.representation_type
                    and "depth" in config.representation_type
                    and config.rgb_encoder_type == "resnet18")

        if use_fuse:
            # Fused RGBD: one 4-channel ResNet18 per camera (replaces separate RGB + depth)
            rgbd_output = config.rgb_per_cam_output + config.depth_per_cam_output
            self.rgbd_encoders = nn.ModuleList([
                RGBDEncoder(output_size=rgbd_output)
                for _ in config.camera_indices
            ])
        else:
            if "img" in config.representation_type:
                self.rgb_encoders = nn.ModuleList([
                    RGBEncoder(
                        encoder_type=config.rgb_encoder_type,
                        output_size=config.rgb_per_cam_output,
                        freeze=config.freeze_rgb_encoder,
                        precompute=config.precompute_rgb_features,
                    )
                    for _ in config.camera_indices
                ])
                # Share a single frozen backbone across all camera encoders to
                # avoid loading N duplicate copies onto GPU.
                if (config.freeze_rgb_encoder
                        and not config.precompute_rgb_features
                        and len(self.rgb_encoders) > 1
                        and self.rgb_encoders[0].backbone is not None):
                    shared_backbone = self.rgb_encoders[0].backbone
                    for enc in self.rgb_encoders[1:]:
                        enc.backbone = shared_backbone

            if "depth" in config.representation_type:
                self.depth_encoders = nn.ModuleList([
                    DepthEncoder(
                        output_size=config.depth_per_cam_output,
                        freeze=config.freeze_depth_encoder,
                    )
                    for _ in config.camera_indices
                ])

        # --- State encoders ---
        self.state_encoders = nn.ModuleDict({
            key: StateEncoder(self._STATE_INPUT_DIMS[key], self._state_output_sizes[key])
            for key in config.representation_type
            if key in self._STATE_INPUT_DIMS
        })

        # --- Instruction encoder ---
        # Both modes produce a tensor with the same key ("instruction") and
        # the same output dim (instruction_embed_dim), so downstream models
        # never need to know which mode is active.
        self.instruction_encoder: Optional[nn.Module] = None
        if config.use_text_instruction:
            texts = _load_instruction_texts(
                config.instructions_file, config.num_instructions
            )
            self.instruction_encoder = TextInstructionEncoder(
                num_instructions=config.num_instructions,
                embed_dim=config.instruction_embed_dim,
                encoder_type=config.text_encoder_type,
                texts=texts,
            )
        elif config.use_instruction:
            self.instruction_encoder = InstructionEncoder(
                config.num_instructions, config.instruction_embed_dim
            )

    # ------------------------------------------------------------------
    # Dimension introspection
    # ------------------------------------------------------------------

    @property
    def modality_dims(self) -> dict[str, int]:
        """
        Output dimension of each active modality.

        Models use this at construction time to size their projections:

        .. code-block:: python

            for name, dim in obs_encoder.modality_dims.items():
                self.proj[name] = nn.Linear(dim, hidden_dim)
        """
        dims: dict[str, int] = {}
        n_cams = len(self.config.camera_indices)
        if self.rgbd_encoders is not None:
            rgbd_out = self.config.rgb_per_cam_output + self.config.depth_per_cam_output
            dims["rgb"] = rgbd_out * n_cams
        elif self.rgb_encoders is not None:
            dims["rgb"] = self.config.rgb_per_cam_output * n_cams
        if self.depth_encoders is not None:
            dims["depth"] = self.config.depth_per_cam_output * n_cams
        for key in self.state_encoders:
            dims[key] = self._state_output_sizes[key]
        if self.instruction_encoder is not None:
            dims["instruction"] = self.config.instruction_embed_dim
        return dims

    @property
    def total_dim(self) -> int:
        """Sum of all ``modality_dims`` values."""
        return sum(self.modality_dims.values())

    def set_depth_stats(self, depth_stat: dict[str, np.ndarray]) -> None:
        """Propagate data-driven depth normalization bounds to all depth-aware encoders.

        Args:
            depth_stat: ``{"min": np.array([v]), "max": np.array([v])}``
                        from ``norm_stats["depth"]``.
        """
        d_min = float(depth_stat["min"].item())
        d_max = float(depth_stat["max"].item())
        if self.depth_encoders is not None:
            for enc in self.depth_encoders:
                enc.set_depth_stats(d_min, d_max)
        if self.rgbd_encoders is not None:
            for enc in self.rgbd_encoders:
                enc.set_depth_stats(d_min, d_max)

    # ------------------------------------------------------------------
    # Image pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, img: Tensor) -> Tensor:
        """Downsample then center-crop.  img: (N, C, H, W) float."""
        if self.config.enable_downsample:
            img = F.interpolate(
                img, size=self.config.downsample_size,
                mode="bilinear", align_corners=False,
            )
        if self.config.enable_crop:
            h_out, w_out = self.config.crop_size
            _, _, h_in, w_in = img.shape
            top  = (h_in - h_out) // 2
            left = (w_in - w_out) // 2
            img  = img[:, :, top: top + h_out, left: left + w_out]
        return img

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> ObsFeatures:
        """
        Encode all active modalities and return a structured ``ObsFeatures``.

        Args:
            batch: Standard batch dict (see ``PolicyModel`` docstring).

        Returns:
            ObsFeatures with ``by_modality`` keyed by modality name.
            Each value is shaped ``(B, obs_horizon, modality_dim)``.
        """
        # Infer B and T from any 2-D-or-higher tensor in the batch.
        sample = next(
            v for v in batch.values()
            if isinstance(v, Tensor) and v.dim() >= 2
        )
        B, T = sample.shape[0], sample.shape[1]

        by_modality: dict[str, Tensor] = {}

        # ---- Per-camera crop offsets (shared between RGB & depth) ----
        # TexasPoker applies a single RandomCrop to the 4-ch (RGB+depth)
        # tensor, so both modalities see the same spatial region.  We
        # reproduce this by generating offsets once and passing them to
        # both encoders.
        _num_cams = max(
            len(self.depth_encoders) if self.depth_encoders else 0,
            len(self.rgb_encoders) if self.rgb_encoders else 0,
            len(self.rgbd_encoders) if self.rgbd_encoders else 0,
        )
        if self.training and _num_cams > 0:
            _crop_offsets: list[tuple[int, int] | None] = [
                (torch.randint(0, 240 - 216 + 1, (1,)).item(),
                 torch.randint(0, 320 - 288 + 1, (1,)).item())
                for _ in range(_num_cams)
            ]
        else:
            _crop_offsets = [None] * max(_num_cams, 1)

        # ---- Fused RGBD ----
        if self.rgbd_encoders is not None and "rgb" in batch and "depth" in batch:
            parts = []
            for ci, enc in enumerate(self.rgbd_encoders):
                rgb = batch["rgb"][:, :, ci]              # (B, T, H, W, 3)
                dep = batch["depth"][:, :, ci]            # (B, T, H, W)
                rgb_flat = rgb.reshape(B * T, *rgb.shape[2:])
                dep_flat = dep.reshape(B * T, *dep.shape[2:])
                feat = enc(rgb_flat, dep_flat)             # (B*T, out)
                parts.append(feat.reshape(B, T, -1))
            by_modality["rgb"] = torch.cat(parts, dim=-1)

        # ---- RGB ----
        elif self.rgb_encoders is not None:
            if self.config.precompute_rgb_features and "rgb_features" in batch:
                # batch["rgb_features"]: (B, T, num_cams, raw_dim)
                parts = []
                for ci, enc in enumerate(self.rgb_encoders):
                    feat = batch["rgb_features"][:, :, ci]          # (B, T, raw_dim)
                    feat_flat = feat.reshape(B * T, feat.shape[-1])  # (B*T, raw_dim)
                    out = enc(feat_flat)                              # (B*T, out_dim)
                    parts.append(out.reshape(B, T, -1))              # (B, T, out_dim)
            elif "rgb" in batch:
                # batch["rgb"]: (B, T, num_cams, H, W, 3)
                parts = []
                for ci, enc in enumerate(self.rgb_encoders):
                    img = batch["rgb"][:, :, ci]            # (B, T, H, W, 3)
                    img_flat = img.reshape(B * T, *img.shape[2:])
                    img_chw = img_flat.float().permute(0, 3, 1, 2)  # (B*T, 3, H, W) float [0,255]
                    img_chw = self._preprocess(img_chw)
                    img_back = img_chw.permute(0, 2, 3, 1) # (B*T, H', W', 3)
                    feat = enc(img_back, crop_ij=_crop_offsets[ci])  # (B*T, out)
                    parts.append(feat.reshape(B, T, -1))
            else:
                parts = []

            if parts:
                by_modality["rgb"] = torch.cat(parts, dim=-1)   # (B, T, rgb_total)

        # ---- Depth ----
        if self.depth_encoders is not None and "depth" in batch:
            # batch["depth"]: (B, T, num_cams, H, W)
            parts = []
            for ci, enc in enumerate(self.depth_encoders):
                dep = batch["depth"][:, :, ci]              # (B, T, H, W)
                dep_2d = dep.reshape(B * T, *dep.shape[2:]) # (B*T, H, W)
                # Deploy-time resize: camera depth frames may differ from
                # training resolution.  deploy_depth_size is set by
                # deploy_policy.py; None during training.
                _dep_sz = getattr(self.config, "deploy_depth_size", None)
                if _dep_sz is not None:
                    dep_4d = dep_2d.reshape(-1, 1, *dep_2d.shape[1:]).float()
                    dep_4d = F.interpolate(dep_4d, size=_dep_sz,
                                           mode="bilinear", align_corners=False)
                    dep_2d = dep_4d.squeeze(1)
                # Call enc.forward() so depth normalization, resize, and crop
                # are all applied.  Shared crop_ij ensures spatial alignment
                # with the RGB encoder for the same camera.
                feat = enc(dep_2d, crop_ij=_crop_offsets[ci])  # (B*T, out)
                parts.append(feat.reshape(B, T, -1))
            by_modality["depth"] = torch.cat(parts, dim=-1)

        # ---- Proprioception ----
        for key, enc in self.state_encoders.items():
            x = batch[key].float()                          # (B, T, state_dim)
            feat = enc(x.reshape(B * T, -1)).reshape(B, T, -1)
            by_modality[key] = feat

        # ---- Instruction (broadcast across timesteps) ----
        if self.instruction_encoder is not None:
            if "instruction" in batch:
                instr = self.instruction_encoder(batch["instruction"])  # (B, D)
            else:
                # Instruction missing (e.g. single-task data without path-encoded IDs).
                # Use zeros so flat_time() shape stays consistent with total_dim.
                dev = next(self.instruction_encoder.parameters()).device
                instr = torch.zeros(B, self.config.instruction_embed_dim, device=dev)
            by_modality["instruction"] = instr.unsqueeze(1).expand(-1, T, -1)

        return ObsFeatures(by_modality=by_modality)
