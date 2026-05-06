"""
RDT (Robotics Diffusion Transformer) — registered as "rdt".

Paper-faithful implementation of:
  Liu et al., "RDT-1B: a Diffusion Foundation Model for Bimanual
  Manipulation", ICLR 2025.  https://rdt-robotics.github.io/rdt-robotics/

Architecture (matching the paper)
---------------------------------
  Self-attention sequence:
    [timestep_token | state_token | action_1 | … | action_N]

  Cross-attention conditions (Alternating Condition Injection):
    Even layers → language (T5 text tokens)
    Odd  layers → image   (SigLIP patch tokens)

  Visual path:
    batch["rgb_features"]  ← precomputed SigLIP-SO400M patch tokens
        (B, T, num_cams, N_patches, 1152)
        → flatten to (B, T*num_cams*N_patches, 1152)
        → visual_proj (2-layer MLP)
        → img_cond (B, N_vis, H)

  Proprioceptive path:
    batch["pos"] (B, T, prop_dim)
        → state_proj (3-layer MLP, last timestep only)
        → state_token (B, 1, H)   [prepended to action sequence]

  Text path:
    batch["instruction"] (B,)  → T5TextEncoder (lookup only)
        → raw T5 tokens (B, seq_len, raw_T5_dim)
        → text_proj (2-layer MLP)
        → lang_cond (B, seq_len, H)

  Diffusion:
    prediction_type = "sample" (predicts clean x₀, NOT noise)
    Training: DDPM, 1000 steps, squaredcos_cap_v2
    Inference: DPMSolver, 5 steps

Key differences from DiffusionPolicy:
  • Visual conditioning uses full SigLIP patch-token sequence (not pooled).
  • Language conditioning via dedicated cross-attention layers (even layers).
  • RMSNorm + QK-normalization (not LayerNorm).
  • Single cross-attn module per layer, alternating KV source.
  • Timestep + state tokens in the action sequence (not in obs_memory).
  • Predicts clean sample, not noise.
  • DPMSolver for fast inference (5 steps).
  • Condition masking during training for robustness.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from diffusers.training_utils import EMAModel

from learning.base import ModelConfig, PolicyModel
from learning.registry import register_model
from learning.common.encoders import ObsEncoder, ObsFeatures


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RDTConfig(ModelConfig):
    """Hyper-parameters for the RDT model (paper-faithful defaults)."""

    # --- Text encoder ---
    text_encoder_type: str = "t5_xxl"
    text_token_max_len: int = 120
    instructions_file: str = "workflow/instructions.json"

    # --- Visual encoder (SigLIP) ---
    siglip_raw_dim: int = 1152
    siglip_resolution: int = 384  # Input resolution for SigLIP (must be 384 for SO400M)
    siglip_pool_patches: int = 0  # Pool patch tokens to this count (0 = no pooling, e.g. 64→8×8)

    # --- Image history (matching official RDT-1B: img_history_size=2) ---
    obs_horizon: int = 2  # 2 frames of image history (current + previous)

    # --- Proprioceptive state ---
    prop_dim: int = 30

    # --- Control frequency ---
    ctrl_freq: float = 1.0  # fixed control frequency (no ctrl_freq in dataset)

    # --- Condition positional embedding lengths ---
    max_lang_cond_len: int = 1024  # max language condition tokens (matching official RDT-1B)
    max_img_cond_len: int = 4368   # max image condition tokens (2 frames × 3 cams × 728 patches)

    # --- Transformer backbone (paper-faithful RDT-1B defaults) ---
    hidden_size: int = 2048
    depth: int = 28
    num_heads: int = 32
    ff_dim: int = 0        # 0 → same as hidden_size (paper: 1x expansion)
    dropout: float = 0.0
    causal_attn: bool = False   # paper does not use causal mask
    diffusion_step_embed_dim: int = 256

    # --- Diffusion ---
    num_diffusion_iters: int = 1000    # paper: 1000
    num_inference_iters: int = 5       # paper: 5 with DPMSolver
    prediction_type: str = "sample"    # paper: predict clean x₀
    inference_scheduler: str = "dpmsolver"  # "dpmsolver", "ddim", "ddpm"

    # --- Training ---
    cond_mask_prob: float = 0.0  # random condition masking probability

    # --- EMA ---
    ema_power: float = 0.75


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (paper uses RMSNorm throughout)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class _SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half = self.dim // 2
        emb = math.log(10000) / half
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([emb.cos(), emb.sin()], dim=-1)


def _get_1d_sincos_pos_embed(embed_dim: int, length: int) -> Tensor:
    """1-D sinusoidal positional embedding [cos, sin] (matching official RDT).

    Returns:
        (1, length, embed_dim) float tensor.
    """
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    omega = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float64) / half))
    pos = torch.arange(length, dtype=torch.float64)
    out = torch.einsum("m,d->md", pos, omega)  # (length, half)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)  # (length, embed_dim)
    return emb.float().unsqueeze(0)  # (1, length, embed_dim)


def _get_multimodal_pos_embed(
    embed_dim: int, mm_cond_lens: "OrderedDict[str, int]"
) -> Tensor:
    """Multimodal positional embedding (matching official RDT).

    First half of embed_dim encodes *modality identity* (sinusoidal over
    modality index), second half encodes *position within modality*.

    Args:
        embed_dim: output embedding dimension.
        mm_cond_lens: OrderedDict mapping modality name → token count.

    Returns:
        (1, total_tokens, embed_dim) float tensor.
    """
    from collections import OrderedDict as _OD  # noqa: F811

    num_modalities = len(mm_cond_lens)
    mod_embed_dim = embed_dim // 2
    pos_embed_dim = embed_dim // 2

    # Modality identity embeddings (one per modality)
    mod_half = mod_embed_dim // 2
    mod_omega = 1.0 / (10000 ** (torch.arange(mod_half, dtype=torch.float64) / mod_half))
    mod_idx = torch.arange(num_modalities, dtype=torch.float64)
    mod_out = torch.einsum("m,d->md", mod_idx, mod_omega)
    mod_sincos = torch.cat([torch.sin(mod_out), torch.cos(mod_out)], dim=-1)  # (num_mod, mod_embed_dim)

    all_embeds = []
    for idx, (_, cond_len) in enumerate(mm_cond_lens.items()):
        # Position embeddings within this modality
        pos_half = pos_embed_dim // 2
        pos_omega = 1.0 / (10000 ** (torch.arange(pos_half, dtype=torch.float64) / pos_half))
        pos_grid = torch.arange(cond_len, dtype=torch.float64)
        pos_out = torch.einsum("m,d->md", pos_grid, pos_omega)
        pos_sincos = torch.cat([torch.sin(pos_out), torch.cos(pos_out)], dim=-1)  # (cond_len, pos_embed_dim)

        # Combine: [modality_id_embed | position_embed]
        mod_broadcast = mod_sincos[idx].unsqueeze(0).expand(cond_len, -1)  # (cond_len, mod_embed_dim)
        combined = torch.cat([mod_broadcast, pos_sincos], dim=-1)  # (cond_len, embed_dim)
        all_embeds.append(combined)

    return torch.cat(all_embeds, dim=0).float().unsqueeze(0)  # (1, total, embed_dim)


def _load_instruction_texts(
    instructions_file: str, num_instructions: int
) -> Optional[list[str]]:
    """Load ordered instruction texts from JSON, or return None.

    If num_instructions exceeds the entries in the JSON, missing entries
    are filled with empty strings (the T5 buffer is still allocated at
    num_instructions size, and padding masks will handle them).
    """
    if not instructions_file:
        return None
    path = Path(instructions_file)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    texts = []
    for i in range(num_instructions):
        entry = data.get(str(i))
        texts.append(entry["text"] if entry is not None else "")
    return texts


def _mlp2x_gelu(in_dim: int, out_dim: int) -> nn.Sequential:
    """2-layer MLP with GELU (paper's visual / language adaptor)."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(out_dim, out_dim),
    )


def _mlp3x_gelu(in_dim: int, out_dim: int) -> nn.Sequential:
    """3-layer MLP with GELU (paper's state adaptor)."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(out_dim, out_dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(out_dim, out_dim),
    )


# ---------------------------------------------------------------------------
# Attention modules with QK normalization (paper-faithful)
# ---------------------------------------------------------------------------

class _SelfAttention(nn.Module):
    """Multi-head self-attention with QK-normalization via RMSNorm."""

    def __init__(
        self, hidden_size: int, num_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = dropout

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each (B, heads, N, head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class _CrossAttention(nn.Module):
    """Multi-head cross-attention with QK-normalization via RMSNorm."""

    def __init__(
        self, hidden_size: int, num_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, N, C = x.shape
        S = context.shape[1]

        q = (
            self.q_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        kv = (
            self.kv_proj(context)
            .reshape(B, S, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_mask: Optional[Tensor] = None
        if key_padding_mask is not None:
            # (B, S) bool True=padding → (B, 1, 1, S) additive float mask
            attn_mask = torch.where(
                key_padding_mask[:, None, None, :],
                torch.tensor(float("-inf"), device=x.device, dtype=x.dtype),
                torch.tensor(0.0, device=x.device, dtype=x.dtype),
            )

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


# ---------------------------------------------------------------------------
# T5TextEncoder
# ---------------------------------------------------------------------------

class T5TextEncoder(nn.Module):
    """
    Frozen T5 text encoder — stores raw T5 token embeddings as buffers.

    The heavy T5 backbone runs **once** at construction time and its outputs
    are stored as non-trainable buffers.  At forward time only a buffer lookup
    happens — T5 is never called again during training or inference.

    The downstream transformer applies its own adaptor MLP (``text_proj``)
    to project raw T5 embeddings to the hidden size.
    """

    _T5_DIMS: dict[str, int] = {
        "t5_small": 512,
        "t5_base":  768,
        "t5_large": 1024,
        "t5_xl":    2048,
        "t5_xxl":   4096,
    }

    _T5_HF_IDS: dict[str, str] = {
        "t5_small": "t5-small",
        "t5_base":  "google/t5-v1_1-base",
        "t5_large": "google/t5-v1_1-large",
        "t5_xl":    "google/t5-v1_1-xl",
        "t5_xxl":   "google/t5-v1_1-xxl",
    }

    def __init__(
        self,
        num_instructions: int,
        encoder_type: str = "t5_base",
        token_max_len: int = 32,
        texts: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.raw_dim = self._T5_DIMS.get(encoder_type, 768)
        self.encoder_type = encoder_type
        self.token_max_len = token_max_len

        if texts is not None:
            tok_feat, tok_mask = self._encode(texts, encoder_type, token_max_len)
        else:
            tok_feat = torch.zeros(num_instructions, token_max_len, self.raw_dim)
            tok_mask = torch.zeros(num_instructions, token_max_len, dtype=torch.bool)

        self.register_buffer("raw_tokens", tok_feat)   # (N, L, raw_dim)
        self.register_buffer("pad_mask",   tok_mask)    # (N, L)  True=padding

    @torch.no_grad()
    def _encode(
        self, texts: list[str], encoder_type: str, max_len: int
    ) -> tuple[Tensor, Tensor]:
        """Run T5 encoder once and return (N, max_len, raw_dim) + bool mask."""
        try:
            from transformers import T5Tokenizer, T5EncoderModel
        except ImportError as e:
            raise ImportError(
                "transformers is required for T5 text encoding.  "
                "Install with: pip install transformers"
            ) from e

        model_id = self._T5_HF_IDS[encoder_type]
        print(f"[T5TextEncoder] Loading {model_id} for one-time text encoding …")
        tokenizer = T5Tokenizer.from_pretrained(model_id)
        # Bypass transformers torch.load safety check for PyTorch < 2.6
        # (T5-v1.1-XXL only provides pytorch_model.bin, not safetensors)
        import transformers.utils.import_utils as _tiu
        import transformers.modeling_utils as _tmu
        _noop = lambda: None
        _orig1, _orig2 = _tiu.check_torch_load_is_safe, _tmu.check_torch_load_is_safe
        _tiu.check_torch_load_is_safe = _noop
        _tmu.check_torch_load_is_safe = _noop
        try:
            model = T5EncoderModel.from_pretrained(model_id)
        finally:
            _tiu.check_torch_load_is_safe = _orig1
            _tmu.check_torch_load_is_safe = _orig2
        model.eval()

        inputs = tokenizer(
            texts,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        outputs = model(**inputs)
        token_feats = outputs.last_hidden_state.float().cpu()   # (N, L, raw_dim)
        pad_mask    = (inputs["input_ids"] == tokenizer.pad_token_id).cpu()

        print(
            f"[T5TextEncoder] Encoded {len(texts)} instructions "
            f"→ ({token_feats.shape[0]}, {token_feats.shape[1]}, {token_feats.shape[2]})  "
            f"[stored as frozen buffer]"
        )
        return token_feats, pad_mask

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            ids: (B,) integer instruction IDs.
        Returns:
            tokens: (B, seq_len, raw_dim)   — raw T5 embeddings.
            mask:   (B, seq_len) bool — True marks padding tokens.
        """
        return self.raw_tokens[ids], self.pad_mask[ids]


# ---------------------------------------------------------------------------
# ACIDecoderLayer — single cross-attention, alternating KV source
# ---------------------------------------------------------------------------

class _ACIDecoderLayer(nn.Module):
    """
    Pre-RMSNorm decoder layer with a single cross-attention module.

    Each layer receives one condition tensor (either lang or img, determined
    by the decoder's alternation logic).  This matches the paper's design
    where a single cross-attention module sees alternating KV sources.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1      = RMSNorm(hidden_size)
        self.self_attn  = _SelfAttention(hidden_size, num_heads, dropout)
        self.norm2      = RMSNorm(hidden_size)
        self.cross_attn = _CrossAttention(hidden_size, num_heads, dropout)
        self.norm3      = RMSNorm(hidden_size)
        self.ffn        = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ff_dim, hidden_size),
        )

    def forward(
        self,
        x:                Tensor,
        cond:             Optional[Tensor] = None,
        cond_mask:        Optional[Tensor] = None,
        self_attn_mask:   Optional[Tensor] = None,
    ) -> Tensor:
        # Self-attention
        x = x + self.self_attn(self.norm1(x), attn_mask=self_attn_mask)
        # Cross-attention (skipped if no condition)
        if cond is not None:
            x = x + self.cross_attn(self.norm2(x), cond, key_padding_mask=cond_mask)
        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# ACIDecoder — alternating lang / img cross-attention
# ---------------------------------------------------------------------------

class _ACIDecoder(nn.Module):
    """
    Alternating Condition Injection decoder (paper-faithful).

    Layer assignment (matching the paper):
      even index (0, 2, 4, …) → language cross-attention  (T5 text tokens)
      odd  index (1, 3, 5, …) → image cross-attention     (SigLIP patch tokens)

    Falls back gracefully: if a condition is None, cross-attention is skipped
    for layers that would use it.
    """

    def __init__(
        self,
        depth:       int,
        hidden_size: int,
        num_heads:   int,
        ff_dim:      int,
        dropout:     float = 0.0,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList([
            _ACIDecoderLayer(hidden_size, num_heads, ff_dim, dropout)
            for _ in range(depth)
        ])

    def forward(
        self,
        tgt:              Tensor,
        lang_cond:        Optional[Tensor] = None,
        img_cond:         Optional[Tensor] = None,
        lang_mask:        Optional[Tensor] = None,
        img_mask:         Optional[Tensor] = None,
        tgt_mask:         Optional[Tensor] = None,
    ) -> Tensor:
        # When language conditioning is absent, all layers use image
        # conditioning instead of leaving even layers without cross-attention.
        if lang_cond is not None:
            conds = [lang_cond, img_cond]
            masks = [lang_mask,  img_mask]
        else:
            conds = [img_cond, img_cond]
            masks = [img_mask,  img_mask]
        x = tgt
        for i, layer in enumerate(self.layers):
            c = conds[i % 2]
            m = masks[i % 2]
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, c, m, tgt_mask, use_reentrant=False)
            else:
                x = layer(x, cond=c, cond_mask=m, self_attn_mask=tgt_mask)
        return x


# ---------------------------------------------------------------------------
# RDTTransformer — paper-faithful architecture
# ---------------------------------------------------------------------------

class _RDTTransformer(nn.Module):
    """
    Denoising transformer for RDT (aligned with official repo).

    Self-attention sequence:
      [timestep_token(1) | ctrl_freq_token(1) | state_token(1) | action_1 | … | action_N]

    Cross-attention conditions (Alternating Condition Injection):
      Even layers → lang_cond (T5 text tokens, projected)
      Odd  layers → img_cond  (SigLIP patch tokens, projected)

    Adaptors (matching official repo):
      - visual_proj:    2-layer MLP (siglip_raw_dim → hidden_size)
      - state_adaptor:  3-layer MLP (action_dim * 2 → hidden_size)
                        Input = [value, mask] concatenated. Both state and
                        action tokens share this adaptor (official pattern).
      - text_proj:      2-layer MLP (raw_T5_dim → hidden_size)
      - output_head:    2-layer MLP with zero-init final layer

    Positional embeddings:
      - x_pos_embed:          multimodal sinusoidal for self-attn sequence
      - lang_cond_pos_embed:  1-D sinusoidal for language condition
      - img_cond_pos_embed:   1-D sinusoidal for image condition
    """

    def __init__(
        self,
        action_dim:               int,
        siglip_raw_dim:           int,    # 1152 for siglip_so400m
        prop_dim:                 int,    # e.g., 30 for full joint pos
        text_raw_dim:             int,    # raw T5 dim (e.g. 768 for t5_base)
        pred_horizon:             int,
        hidden_size:              int   = 512,
        depth:                    int   = 12,
        num_heads:                int   = 8,
        ff_dim:                   int   = 512,
        dropout:                  float = 0.0,
        causal_attn:              bool  = False,
        diffusion_step_embed_dim: int   = 256,
        max_lang_cond_len:        int   = 32,
        max_img_cond_len:         int   = 4096,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.prop_dim = prop_dim

        # --- Timestep embedding (sinusoidal + MLP with SiLU, matching paper) ---
        self.time_emb = nn.Sequential(
            _SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # --- Control frequency embedding (matching official repo) ---
        self.freq_emb = nn.Sequential(
            _SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # --- Unified state adaptor: 3-layer MLP (matching official repo) ---
        # Input = [value, mask] concatenated → action_dim * 2
        # Both state tokens and action tokens share this adaptor.
        self.state_adaptor = _mlp3x_gelu(action_dim * 2, hidden_size)

        # --- Visual adaptor: 2-layer MLP (paper's image adaptor) ---
        self.visual_proj = _mlp2x_gelu(siglip_raw_dim, hidden_size)

        # --- Text adaptor: 2-layer MLP (paper's language adaptor) ---
        self.text_proj = _mlp2x_gelu(text_raw_dim, hidden_size)

        # --- Positional embeddings (matching official repo) ---
        # Self-attention: [timestep(1), ctrl_freq(1), state(1), action(pred_horizon)]
        self.x_pos_embed = nn.Parameter(
            _get_multimodal_pos_embed(
                hidden_size,
                OrderedDict([
                    ("timestep", 1),
                    ("ctrl_freq", 1),
                    ("state", 1),
                    ("action", pred_horizon),
                ]),
            )
        )  # (1, 3+pred_horizon, hidden_size)

        # Language condition positional embedding
        self.lang_cond_pos_embed = nn.Parameter(
            _get_1d_sincos_pos_embed(hidden_size, max_lang_cond_len)
        )  # (1, max_lang_cond_len, hidden_size)

        # Image condition positional embedding
        self.img_cond_pos_embed = nn.Parameter(
            _get_1d_sincos_pos_embed(hidden_size, max_img_cond_len)
        )  # (1, max_img_cond_len, hidden_size)

        # --- ACI decoder ---
        self.aci_decoder = _ACIDecoder(
            depth=depth,
            hidden_size=hidden_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # --- Output head: 2-layer MLP (paper uses zero-init on final layer) ---
        self.ln_out = RMSNorm(hidden_size)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, action_dim),
        )

        self.causal_attn = causal_attn
        self.pred_horizon = pred_horizon

        # --- Initialization ---
        # x_pos_embed, lang_cond_pos_embed, img_cond_pos_embed are already
        # initialized from sinusoidal values above. apply() will xavier-init
        # all Linear layers; we then zero-init the output head final layer.
        self.apply(self._init_weights)
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _causal_mask(self, T: int, device: torch.device) -> Optional[Tensor]:
        if not self.causal_attn:
            return None
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        return mask.masked_fill(mask.bool(), float("-inf"))

    def forward(
        self,
        noisy_action:  Tensor,                  # (B, T_pred, action_dim)
        timestep:      Tensor,                  # (B,)
        visual_tokens: Optional[Tensor] = None, # (B, N_vis, siglip_raw_dim)
        prop_state:    Optional[Tensor] = None, # (B, T_obs, prop_dim)
        text_tokens:   Optional[Tensor] = None, # (B, seq_len, raw_T5_dim)
        text_mask:     Optional[Tensor] = None, # (B, seq_len) True=padding
        ctrl_freq:     Optional[Tensor] = None, # (B,) scalar frequency values
    ) -> Tensor:
        B, T, _ = noisy_action.shape
        device = noisy_action.device

        # --- Build self-attention sequence ---
        tokens = []

        # Timestep token
        time_tok = self.time_emb(timestep).unsqueeze(1)         # (B, 1, H)
        tokens.append(time_tok)

        # Control frequency token (matching official repo)
        if ctrl_freq is None:
            ctrl_freq = torch.ones(B, device=device)
        freq_tok = self.freq_emb(ctrl_freq).unsqueeze(1)        # (B, 1, H)
        tokens.append(freq_tok)

        # State token via unified state_adaptor (matching official repo)
        if prop_state is not None:
            state_val = prop_state[:, -1:, :]                   # (B, 1, prop_dim)
            # Align prop_dim to action_dim (official: state_dim == action_dim)
            if state_val.shape[-1] < self.action_dim:
                state_val = F.pad(state_val, (0, self.action_dim - state_val.shape[-1]))
            elif state_val.shape[-1] > self.action_dim:
                state_val = state_val[:, :, :self.action_dim]
            state_mask = torch.ones(B, 1, self.action_dim, device=device)
            state_input = torch.cat([state_val, state_mask], dim=-1)  # (B, 1, action_dim*2)
            state_tok = self.state_adaptor(state_input)         # (B, 1, H)
        else:
            state_tok = torch.zeros(B, 1, self.hidden_size, device=device)
        tokens.append(state_tok)

        # Action tokens via unified state_adaptor (matching official repo)
        action_mask = torch.ones_like(noisy_action)             # (B, T, action_dim)
        action_input = torch.cat([noisy_action, action_mask], dim=-1)  # (B, T, action_dim*2)
        act_tok = self.state_adaptor(action_input)              # (B, T, H)
        tokens.append(act_tok)

        x = torch.cat(tokens, dim=1)                            # (B, 3+T, H)
        x = x + self.x_pos_embed[:, : x.shape[1]]

        # --- Build cross-attention conditions ---
        # img_cond: SigLIP patch tokens → visual_proj + positional embedding
        img_cond: Optional[Tensor] = None
        if visual_tokens is not None:
            img_cond = self.visual_proj(visual_tokens)          # (B, N_vis, H)
            img_cond = img_cond + self.img_cond_pos_embed[:, : img_cond.shape[1]]

        # lang_cond: T5 tokens → text_proj + positional embedding
        lang_cond: Optional[Tensor] = None
        lang_mask: Optional[Tensor] = None
        if text_tokens is not None:
            lang_cond = self.text_proj(text_tokens)             # (B, seq_len, H)
            lang_cond = lang_cond + self.lang_cond_pos_embed[:, : lang_cond.shape[1]]
            lang_mask = text_mask

        # --- ACI decoding ---
        # Even layers → lang, Odd layers → img (matching paper)
        seq_len_full = x.shape[1]
        output = self.aci_decoder(
            x, lang_cond, img_cond,
            lang_mask=lang_mask,
            img_mask=None,
            tgt_mask=self._causal_mask(seq_len_full, device),
        )

        # Output head on ALL tokens, then slice actions (matching official repo)
        output = self.output_head(self.ln_out(output))          # (B, 3+T, action_dim)
        return output[:, -self.pred_horizon:]                   # (B, T, action_dim)


# ---------------------------------------------------------------------------
# RDT PolicyModel
# ---------------------------------------------------------------------------

@register_model("rdt")
class RDT(PolicyModel):
    """
    Robotics Diffusion Transformer (RDT) — paper-faithful implementation.

    Visual backbone:  SigLIP-SO400M (frozen), patch tokens, 1152-d.
      Features must be precomputed once before training:
        python workflow/precompute_features.py \\
            --encoder siglip_so400m \\
            --data_dir data/easy_mode \\
            --feature_dir data/siglip_features

    Text backbone:    T5 (frozen buffers, looked up at runtime).
    Denoising:        ACI Transformer + DDPM training, DPMSolver inference.
    """

    def __init__(
        self,
        obs_encoder:   ObsEncoder,
        transformer:   _RDTTransformer,
        text_encoder:  T5TextEncoder,
        config:        RDTConfig,
    ) -> None:
        super().__init__()
        self.obs_encoder  = obs_encoder
        self.transformer  = transformer
        self.text_encoder = text_encoder
        self.config       = config

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
            prediction_type=config.prediction_type,
        )

        # Inference scheduler
        if config.inference_scheduler == "dpmsolver":
            self._inference_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=config.num_diffusion_iters,
                beta_schedule="squaredcos_cap_v2",
                prediction_type=config.prediction_type,
            )
        elif config.inference_scheduler == "ddim":
            self._inference_scheduler = DDIMScheduler(
                num_train_timesteps=config.num_diffusion_iters,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=False,
                prediction_type=config.prediction_type,
            )
        else:
            self._inference_scheduler = self.noise_scheduler

        self.ema = EMAModel(
            parameters=transformer.parameters(),
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
        config:      RDTConfig,
    ) -> "RDT":
        # T5 text encoder — runs T5 once if instructions_file exists.
        texts = _load_instruction_texts(
            config.instructions_file, config.num_instructions
        )
        text_encoder = T5TextEncoder(
            num_instructions=config.num_instructions,
            encoder_type=config.text_encoder_type,
            token_max_len=config.text_token_max_len,
            texts=texts,
        )

        # Resolve ff_dim: 0 → same as hidden_size (paper: 1x expansion)
        ff_dim = config.ff_dim if config.ff_dim > 0 else config.hidden_size

        # Denoising transformer
        transformer = _RDTTransformer(
            action_dim=config.action_dim,
            siglip_raw_dim=config.siglip_raw_dim,
            prop_dim=config.prop_dim,
            text_raw_dim=text_encoder.raw_dim,
            pred_horizon=config.pred_horizon,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            ff_dim=ff_dim,
            dropout=config.dropout,
            causal_attn=config.causal_attn,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            max_lang_cond_len=config.max_lang_cond_len,
            max_img_cond_len=config.max_img_cond_len,
        )

        return cls(
            obs_encoder=obs_encoder,
            transformer=transformer,
            text_encoder=text_encoder,
            config=config,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_siglip_tokens(self, batch: dict[str, Tensor]) -> Optional[Tensor]:
        """Run frozen SigLIP on raw RGB to extract patch tokens on-the-fly.

        Uses the SigLIP backbone already loaded inside ObsEncoder.

        Returns:
            (B, T*num_cams*N_patches, siglip_raw_dim) or None.
        """
        if "rgb" not in batch or self.obs_encoder.rgb_encoders is None:
            return None

        rgb = batch["rgb"]  # (B, T, num_cams, H, W, 3) uint8
        B, T, C, H, W, _ = rgb.shape
        imgs = rgb.reshape(B * T * C, H, W, 3).float() / 255.0
        imgs = imgs.permute(0, 3, 1, 2)  # (N, 3, H, W)
        imgs = F.interpolate(imgs, size=(384, 384), mode="bilinear", align_corners=False)
        imgs = (imgs - 0.5) / 0.5  # [0,1] → [-1,1] SigLIP normalization

        backbone = self.obs_encoder.rgb_encoders[0].backbone
        with torch.no_grad():
            out = backbone(pixel_values=imgs)
            # Drop the first token to match precompute_features.py, which saves
            # hidden[:, 1:] → 728 patches per camera. The trained
            # img_cond_pos_embed is sized for 728/cam, so deploy must match.
            feats = out.last_hidden_state[:, 1:]  # (N, 728, D)

        # Pool patch tokens to reduce cross-attention cost.
        pool_size = getattr(self.config, 'siglip_pool_patches', 0)
        if pool_size > 0 and feats.shape[1] > pool_size:
            N_img, N_p, D = feats.shape
            side = int(N_p ** 0.5)  # 27
            feats_2d = feats.reshape(N_img, side, side, D).permute(0, 3, 1, 2)
            pool_side = int(pool_size ** 0.5)
            feats_2d = F.adaptive_avg_pool2d(feats_2d, pool_side)
            feats = feats_2d.permute(0, 2, 3, 1).reshape(N_img, pool_side * pool_side, D)

        N_patches, D = feats.shape[1], feats.shape[2]
        return feats.reshape(B, T * C * N_patches, D)

    def _encode(self, batch: dict[str, Tensor]):
        """
        Return (visual_tokens, prop_state, text_tokens, text_mask).

        Visual tokens come from precomputed ``rgb_features`` if available,
        otherwise SigLIP is run on-the-fly from raw ``rgb`` images.
        """
        # --- Visual ---
        visual_tokens: Optional[Tensor] = None
        if "rgb_features" in batch:
            x = batch["rgb_features"].float()
            if x.dim() == 5:
                B, T, C, N, D = x.shape
                # Pool precomputed patch tokens if configured.
                pool_size = getattr(self.config, 'siglip_pool_patches', 0)
                if pool_size > 0 and N > pool_size:
                    side = int(N ** 0.5)
                    pool_side = int(pool_size ** 0.5)
                    x2d = x.reshape(B * T * C, side, side, D).permute(0, 3, 1, 2)
                    x2d = F.adaptive_avg_pool2d(x2d, pool_side)
                    x = x2d.permute(0, 2, 3, 1).reshape(B, T, C, pool_side * pool_side, D)
                    N = pool_side * pool_side
                visual_tokens = x.reshape(B, T * C * N, D)
            elif x.dim() == 4:
                B, T, C, D = x.shape
                visual_tokens = x.reshape(B, T * C, D)
        else:
            visual_tokens = self._extract_siglip_tokens(batch)

        # --- Proprioceptive ---
        prop_state: Optional[Tensor] = None
        for key in ("pos", "eef", "hand_pos"):
            if key in batch:
                prop_state = batch[key].float()
                break

        # --- Text ---
        text_tokens: Optional[Tensor] = None
        text_mask:   Optional[Tensor] = None
        if "instruction" in batch:
            text_tokens, text_mask = self.text_encoder(batch["instruction"])

        return visual_tokens, prop_state, text_tokens, text_mask

    def _unnormalize(self, action: Tensor) -> Tensor:
        stat = self.norm_stats.get("action", {})
        if not stat:
            return action
        mn = torch.tensor(stat["min"], device=action.device, dtype=action.dtype)
        mx = torch.tensor(stat["max"], device=action.device, dtype=action.dtype)
        range_ = torch.clamp(mx - mn, min=1e-8)
        return (action + 1) / 2 * range_ + mn

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        visual_tokens, prop_state, text_tokens, text_mask = self._encode(batch)
        action = batch["action"].float()
        B = action.shape[0]
        device = action.device

        # --- Condition masking (paper: randomly zero out per sample) ---
        if self.training and self.config.cond_mask_prob > 0:
            if visual_tokens is not None:
                vis_drop = torch.rand(B, device=device) < self.config.cond_mask_prob
                visual_tokens = visual_tokens * (~vis_drop)[:, None, None].float()
            if text_tokens is not None:
                txt_drop = torch.rand(B, device=device) < self.config.cond_mask_prob
                text_tokens = text_tokens * (~txt_drop)[:, None, None].float()

        # --- DDPM forward ---
        timesteps = torch.randint(
            0, self.config.num_diffusion_iters, (B,), device=device
        ).long()
        noise        = torch.randn_like(action)
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)

        ctrl_freq = torch.full((B,), self.config.ctrl_freq, device=device)
        model_output = self.transformer(
            noisy_action, timesteps,
            visual_tokens=visual_tokens,
            prop_state=prop_state,
            text_tokens=text_tokens,
            text_mask=text_mask,
            ctrl_freq=ctrl_freq,
        )

        # Paper: prediction_type="sample" → target is clean action
        if self.config.prediction_type == "sample":
            target = action
        else:
            target = noise

        loss = F.mse_loss(model_output, target)
        metrics = {
            "loss":     loss,
            "arm_mse":  F.mse_loss(model_output[:, :, :6],  target[:, :, :6]).detach(),
            "hand_mse": F.mse_loss(model_output[:, :, 6:],  target[:, :, 6:]).detach(),
        }
        # Unnormalized MSE only meaningful when predicting clean actions
        if self.config.prediction_type == "sample":
            metrics.update(self._unnorm_action_mse(model_output.detach(), target))
        return metrics

    # ------------------------------------------------------------------
    # Validation (full reverse diffusion)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Validation via full reverse diffusion.

        Uses the inference scheduler (DPMSolver by default) to denoise
        from pure noise, then measures MSE against ground-truth actions.
        """
        visual_tokens, prop_state, text_tokens, text_mask = self._encode(batch)
        action = batch["action"].float()
        B = action.shape[0]
        device = action.device

        scheduler = self._inference_scheduler
        scheduler.set_timesteps(self.config.num_inference_iters)

        ctrl_freq = torch.full((B,), self.config.ctrl_freq, device=device)
        pred_action = torch.randn_like(action)
        for t in scheduler.timesteps:
            t_batch = t.unsqueeze(0).expand(B).to(device)
            model_output = self.transformer(
                pred_action, t_batch,
                visual_tokens=visual_tokens,
                prop_state=prop_state,
                text_tokens=text_tokens,
                text_mask=text_mask,
                ctrl_freq=ctrl_freq,
            )
            pred_action = scheduler.step(model_output, t, pred_action).prev_sample

        loss = F.mse_loss(pred_action, action)
        result: dict[str, Tensor] = {
            "loss":     loss,
            "arm_mse":  F.mse_loss(pred_action[:, :, :6], action[:, :, :6]).detach(),
            "hand_mse": F.mse_loss(pred_action[:, :, 6:], action[:, :, 6:]).detach(),
        }
        result.update(self._unnorm_action_mse(pred_action, action))
        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.ema.store(self.transformer.parameters())
        self.ema.copy_to(self.transformer.parameters())

        try:
            visual_tokens, prop_state, text_tokens, text_mask = self._encode(batch)

            ref = visual_tokens if visual_tokens is not None else prop_state
            if ref is None:
                raise ValueError("RDT requires at least visual tokens or prop state.")
            B, device = ref.shape[0], ref.device

            scheduler = self._inference_scheduler
            scheduler.set_timesteps(self.config.num_inference_iters)

            ctrl_freq = torch.full((B,), self.config.ctrl_freq, device=device)
            action = torch.randn(
                B, self.config.pred_horizon, self.config.action_dim, device=device
            )
            for t in scheduler.timesteps:
                t_batch = t.unsqueeze(0).expand(B).to(device)
                model_output = self.transformer(
                    action, t_batch,
                    visual_tokens=visual_tokens,
                    prop_state=prop_state,
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                    ctrl_freq=ctrl_freq,
                )
                action = scheduler.step(model_output, t, action).prev_sample
        finally:
            self.ema.restore(self.transformer.parameters())

        action = action[:, : self.config.action_horizon]
        if self.norm_stats:
            action = self._unnormalize(action)
        return action

    # ------------------------------------------------------------------
    # Optimizer / EMA
    # ------------------------------------------------------------------

    def configure_optimizers(
        self, lr: float = 1e-4, weight_decay: float = 1e-5
    ) -> list[torch.optim.Optimizer]:
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() == 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            torch.optim.AdamW(
                [
                    {"params": decay,    "weight_decay": weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=lr,
            )
        ]

    def on_after_step(self) -> None:
        self.ema.step(self.transformer.parameters())


# ---------------------------------------------------------------------------
# Pretrained checkpoint loading (official RDT-1B → our RDT)
# ---------------------------------------------------------------------------

# Key mapping from official RDTRunner state_dict → our RDT state_dict.
# Official keys have two namespaces:
#   "model.*"          → the inner RDT transformer (blocks, embedders, final_layer)
#   "lang_adaptor.*"   → top-level language adaptor
#   "img_adaptor.*"    → top-level image adaptor
#   "state_adaptor.*"  → top-level state adaptor
#
# Our keys live under "transformer.*" inside the RDT PolicyModel.

_OFFICIAL_TO_OURS: list[tuple[str, str]] = [
    # --- Timestep & frequency embedders ---
    # Official: model.t_embedder.mlp.{0,2}  →  Ours: transformer.time_emb.{1,3}
    # (our Sequential has SinusoidalPosEmb at index 0, so Linear layers at 1,3)
    ("model.t_embedder.mlp.0.", "transformer.time_emb.1."),
    ("model.t_embedder.mlp.2.", "transformer.time_emb.3."),
    ("model.freq_embedder.mlp.0.", "transformer.freq_emb.1."),
    ("model.freq_embedder.mlp.2.", "transformer.freq_emb.3."),

    # --- Positional embeddings ---
    ("model.x_pos_embed", "transformer.x_pos_embed"),
    ("model.img_cond_pos_embed", "transformer.img_cond_pos_embed"),

    # --- Image adaptor (shape matches: 1152 → 2048) ---
    ("img_adaptor.", "transformer.visual_proj."),

    # --- Language adaptor (shape matches: 4096 → 2048 with t5_xxl) ---
    ("lang_adaptor.", "transformer.text_proj."),

    # --- Language positional embedding (shape matches: 1024 tokens) ---
    ("model.lang_cond_pos_embed", "transformer.lang_cond_pos_embed"),

    # --- ACI decoder blocks ---
    # Per-block pattern: model.blocks.{i}.X → transformer.aci_decoder.layers.{i}.Y
    # norm1, norm2, norm3 — direct match
    # attn.qkv           → self_attn.qkv
    # attn.q_norm         → self_attn.q_norm
    # attn.k_norm         → self_attn.k_norm
    # attn.proj           → self_attn.proj
    # cross_attn.q        → cross_attn.q_proj
    # cross_attn.kv       → cross_attn.kv_proj
    # cross_attn.q_norm   → cross_attn.q_norm
    # cross_attn.k_norm   → cross_attn.k_norm
    # cross_attn.proj     → cross_attn.proj
    # ffn.fc1             → ffn.0
    # ffn.fc2             → ffn.2
    # (block-level rules are applied via _remap_block_key below)

    # --- Final layer / output head ---
    ("model.final_layer.norm_final.", "transformer.ln_out."),
    ("model.final_layer.ffn_final.fc1.", "transformer.output_head.0."),
    # model.final_layer.ffn_final.fc2 → skip (shape mismatch: 128 vs 30)
]

# Keys to always skip (shape mismatch due to action_dim)
_SKIP_PREFIXES = [
    "model.final_layer.ffn_final.fc2.",  # output_dim 128 vs 30
    "state_adaptor.",                 # input 256 vs 60
]


def _remap_block_key(official_key: str) -> Optional[str]:
    """Remap a single model.blocks.{i}.* key to our naming convention."""
    import re
    m = re.match(r"model\.blocks\.(\d+)\.(.*)", official_key)
    if m is None:
        return None
    idx, suffix = m.group(1), m.group(2)
    prefix = f"transformer.aci_decoder.layers.{idx}."

    # Self-attention
    if suffix.startswith("attn.qkv."):
        return prefix + "self_attn.qkv." + suffix[len("attn.qkv."):]
    if suffix.startswith("attn.q_norm."):
        return prefix + "self_attn.q_norm." + suffix[len("attn.q_norm."):]
    if suffix.startswith("attn.k_norm."):
        return prefix + "self_attn.k_norm." + suffix[len("attn.k_norm."):]
    if suffix.startswith("attn.proj."):
        return prefix + "self_attn.proj." + suffix[len("attn.proj."):]

    # Cross-attention (note: official uses "q" and "kv", we use "q_proj" and "kv_proj")
    if suffix.startswith("cross_attn.q."):
        return prefix + "cross_attn.q_proj." + suffix[len("cross_attn.q."):]
    if suffix.startswith("cross_attn.kv."):
        return prefix + "cross_attn.kv_proj." + suffix[len("cross_attn.kv."):]
    if suffix.startswith("cross_attn.q_norm."):
        return prefix + "cross_attn.q_norm." + suffix[len("cross_attn.q_norm."):]
    if suffix.startswith("cross_attn.k_norm."):
        return prefix + "cross_attn.k_norm." + suffix[len("cross_attn.k_norm."):]
    if suffix.startswith("cross_attn.proj."):
        return prefix + "cross_attn.proj." + suffix[len("cross_attn.proj."):]

    # Norms — direct match
    if suffix.startswith("norm1."):
        return prefix + "norm1." + suffix[len("norm1."):]
    if suffix.startswith("norm2."):
        return prefix + "norm2." + suffix[len("norm2."):]
    if suffix.startswith("norm3."):
        return prefix + "norm3." + suffix[len("norm3."):]

    # FFN: fc1 → ffn.0, fc2 → ffn.2
    if suffix.startswith("ffn.fc1."):
        return prefix + "ffn.0." + suffix[len("ffn.fc1."):]
    if suffix.startswith("ffn.fc2."):
        return prefix + "ffn.2." + suffix[len("ffn.fc2."):]

    return None


def _remap_key(official_key: str) -> Optional[str]:
    """Map an official RDTRunner state_dict key to our RDT key, or None to skip."""
    # Check skip list first
    for skip in _SKIP_PREFIXES:
        if official_key.startswith(skip):
            return None

    # Try block-level remapping
    if official_key.startswith("model.blocks."):
        return _remap_block_key(official_key)

    # Try direct prefix remapping
    for src, dst in _OFFICIAL_TO_OURS:
        if official_key.startswith(src):
            return dst + official_key[len(src):]

    return None


def load_pretrained_rdt(
    model: "RDT",
    ckpt_path: str,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Load pretrained weights from an official RDT-1B checkpoint into our RDT model.

    Performs key remapping and shape validation. Only loads parameters that
    exist in both models with matching shapes. Prints a detailed summary.

    Args:
        model: Our RDT PolicyModel instance (already constructed).
        ckpt_path: Path to the pretrained checkpoint file (.pt / .bin / .safetensors)
                   or a directory containing pytorch_model.bin / model.safetensors.
    """
    ckpt_path_obj = Path(ckpt_path)

    # --- Load pretrained state dict ---
    if ckpt_path_obj.is_dir():
        # Try safetensors first, then pytorch_model.bin
        sf_path = ckpt_path_obj / "model.safetensors"
        pt_path = ckpt_path_obj / "pytorch_model.bin"
        if sf_path.exists():
            from safetensors.torch import load_file
            pretrained_sd = load_file(str(sf_path), device=str(device))
        elif pt_path.exists():
            pretrained_sd = torch.load(str(pt_path), map_location=device)
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {ckpt_path}. "
                f"Expected model.safetensors or pytorch_model.bin"
            )
    elif ckpt_path_obj.suffix == ".safetensors":
        from safetensors.torch import load_file
        pretrained_sd = load_file(str(ckpt_path_obj), device=str(device))
    else:
        raw = torch.load(str(ckpt_path_obj), map_location=device)
        # Support both raw state_dict and wrapped checkpoint formats
        if isinstance(raw, dict) and "model_state_dict" in raw:
            pretrained_sd = raw["model_state_dict"]
        elif isinstance(raw, dict) and "module" in raw:
            pretrained_sd = raw["module"]
        else:
            pretrained_sd = raw

    # --- Detect format: official RDTRunner vs our checkpoint ---
    has_model_prefix = any(k.startswith("model.") for k in pretrained_sd)
    has_transformer_prefix = any(k.startswith("transformer.") for k in pretrained_sd)

    if has_transformer_prefix and not has_model_prefix:
        # Already in our format — do a direct partial load
        print("[pretrained] Checkpoint appears to be in our format. Doing direct partial load.")
        _partial_load_direct(model, pretrained_sd)
        return

    if not has_model_prefix:
        print("[pretrained] WARNING: Checkpoint has neither 'model.' nor 'transformer.' prefix. "
              "Attempting direct partial load.")
        _partial_load_direct(model, pretrained_sd)
        return

    # --- Official RDTRunner format: remap keys ---
    print("[pretrained] Detected official RDT-1B checkpoint format. Remapping keys...")

    model_sd = model.state_dict()
    loaded, skipped_remap, skipped_shape, skipped_missing = [], [], [], []

    # Positional embeddings that can be truncated/padded along the sequence dim
    _POS_EMBED_KEYS = {"transformer.img_cond_pos_embed", "transformer.x_pos_embed"}

    for official_key, official_val in pretrained_sd.items():
        our_key = _remap_key(official_key)
        if our_key is None:
            skipped_remap.append(official_key)
            continue
        if our_key not in model_sd:
            skipped_missing.append(f"{official_key} → {our_key}")
            continue
        if model_sd[our_key].shape != official_val.shape:
            # For positional embeddings, truncate or pad along dim=1
            if our_key in _POS_EMBED_KEYS and official_val.dim() == 3:
                our_len = model_sd[our_key].shape[1]
                pre_len = official_val.shape[1]
                if pre_len >= our_len:
                    # Truncate pretrained → our length
                    model_sd[our_key] = official_val[:, :our_len, :].to(model_sd[our_key].dtype)
                else:
                    # Partial copy (pad rest stays as initialized)
                    model_sd[our_key][:, :pre_len, :] = official_val.to(model_sd[our_key].dtype)
                loaded.append(f"{official_key} → {our_key} (pos_embed {pre_len}→{our_len})")
                continue
            skipped_shape.append(
                f"{official_key} → {our_key}: "
                f"pretrained {list(official_val.shape)} vs ours {list(model_sd[our_key].shape)}"
            )
            continue
        model_sd[our_key] = official_val.to(model_sd[our_key].dtype)
        loaded.append(f"{official_key} → {our_key}")

    model.load_state_dict(model_sd)

    # Also initialize EMA with loaded weights
    if hasattr(model, "ema"):
        model.ema = EMAModel(
            parameters=model.transformer.parameters(),
            power=model.config.ema_power,
        )

    # --- Print summary ---
    total_params = sum(p.numel() for p in model.parameters())
    loaded_params = sum(
        pretrained_sd[k.split(" → ")[0]].numel()
        for k in loaded
    )

    print(f"\n{'='*60}")
    print(f"  Pretrained RDT-1B weight loading summary")
    print(f"{'='*60}")
    print(f"  Loaded:          {len(loaded):>4} params  ({loaded_params/1e6:.1f}M parameters)")
    print(f"  Skipped (remap): {len(skipped_remap):>4} (no mapping rule — action_dim/lang_dim mismatch)")
    print(f"  Skipped (shape): {len(skipped_shape):>4} (key mapped but shape mismatch)")
    print(f"  Skipped (miss):  {len(skipped_missing):>4} (mapped key not in our model)")
    print(f"  Total model:     {total_params/1e6:.1f}M parameters")
    print(f"{'='*60}")

    if skipped_remap:
        print(f"\n  Skipped (no mapping):")
        for k in skipped_remap:
            print(f"    {k}")
    if skipped_shape:
        print(f"\n  Skipped (shape mismatch):")
        for k in skipped_shape:
            print(f"    {k}")
    if skipped_missing:
        print(f"\n  Skipped (mapped key missing in model):")
        for k in skipped_missing:
            print(f"    {k}")
    print()


def _partial_load_direct(model: "RDT", pretrained_sd: dict) -> None:
    """Direct partial load when checkpoint is already in our key format."""
    model_sd = model.state_dict()
    loaded, skipped = [], []
    for k, v in pretrained_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v.to(model_sd[k].dtype)
            loaded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(model_sd)
    print(f"[pretrained] Loaded {len(loaded)} params, skipped {len(skipped)} (shape mismatch or missing)")
