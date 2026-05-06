"""
Unified training entry point.

Supports all registered models through a single --model argument.
Data loading, normalization, checkpointing, and logging are handled here
so that individual model implementations stay free of boilerplate.

Quick start
-----------
Single-task Diffusion Policy::

    python train.py \\
        --model diffusion_policy \\
        --train_path data_split/pick_up_card_train_20 \\
        --val_path   data_split/pick_up_card_val_20 \\
        --save_path  checkpoints/dp_exp1

Multi-task ACT::

    python train.py \\
        --model act \\
        --multitask \\
        --train_paths "new_data/easy_mode/0/pick_up_card_train_50;new_data/easy_mode/1/pick_up_card_train_50" \\
        --val_paths   "new_data/easy_mode/0/pick_up_card_test;new_data/easy_mode/1/pick_up_card_test" \\
        --use_instruction \\
        --save_path checkpoints/act_multitask

List all available models::

    python train.py --list_models
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch._dynamo
import torch.distributed as dist
from tqdm import tqdm

# TF32 matmul (A100+): ~30% faster matmuls with negligible precision loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- Register all built-in models before parsing args ---
import learning  # noqa: F401 — side-effect: registers dp + act

from learning.registry import build_model, list_models
from learning.common.encoders import ObsEncoder, ObsEncoderConfig
from learning.dp.model import DiffusionPolicyConfig
from learning.act.model import ACTConfig
from learning.baku.model import BakuConfig
from learning.rdt.model import RDTConfig
from data_processing.dataset import (
    DatasetConfig,
    build_dataset,
    build_multitask_dataset,
    build_dataset_lazy,
    build_multitask_dataset_lazy,
    create_dataloader,
)
from data_processing.normalization import stats_to_json, stats_from_json


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a policy model on robot demonstration data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Meta ---
    p.add_argument("--model", default="diffusion_policy",
                   help="Model name (see --list_models).")
    p.add_argument("--list_models", action="store_true",
                   help="Print available models and exit.")

    # --- Data paths ---
    p.add_argument("--train_path", default="",
                   help="Path to training data directory (single-task).")
    p.add_argument("--val_path", default="",
                   help="Path to validation data directory (single-task).")
    p.add_argument("--multitask", action="store_true",
                   help="Enable multi-task mode (use --train_paths / --val_paths).")
    p.add_argument("--train_paths", default="",
                   help="Semicolon-separated training directories (multi-task).")
    p.add_argument("--val_paths", default="",
                   help="Semicolon-separated validation directories (multi-task).")

    # --- Pre-computed feature directories (output of precompute_features.py) ---
    p.add_argument("--feature_dir", default="",
                   help="Leaf feature directory for training data (single-task). "
                        "Mirrors the train_path structure: feature_dir/dataXXXX.npz.")
    p.add_argument("--val_feature_dir", default="",
                   help="Leaf feature directory for validation data (single-task). "
                        "Defaults to --feature_dir when not specified.")
    p.add_argument("--feature_dirs", default="",
                   help="Semicolon-separated leaf feature directories (multi-task), "
                        "matching --train_paths order.")
    p.add_argument("--val_feature_dirs", default="",
                   help="Semicolon-separated leaf feature directories (multi-task), "
                        "matching --val_paths order.")

    # --- Observation modalities ---
    p.add_argument("--representation_type", default="img-pos",
                   help="Hyphen-separated modalities: img,depth,pos,eef,hand_pos,efforts,velocity,touch.")
    p.add_argument("--camera_indices", default="012",
                   help="Concatenated camera indices, e.g. '012' for cameras 0,1,2.")

    # --- Visual encoder ---
    p.add_argument("--rgb_encoder", default="resnet18",
                   choices=["resnet18", "dinov2_vits14", "dinov2_vitb14",
                            "dinov2_vitl14", "dinov2_vitl14_patch",
                            "dinov3_vitl16", "siglip_so400m"],
                   help="RGB backbone type.")
    p.add_argument("--depth_encoder", default="resnet18")
    p.add_argument("--freeze_rgb_encoder", action="store_true")
    p.add_argument("--freeze_depth_encoder", action="store_true")
    p.add_argument("--precompute_rgb_features", action="store_true",
                   help="Use pre-computed features (DinoV2 or SigLIP; requires dataset prep).")
    p.add_argument("--fuse_rgbd", action="store_true",
                   help="Fuse RGB+depth into 4-channel ResNet18 (halves backbone passes).")
    p.add_argument("--rgb_per_cam_output", type=int, default=96)
    p.add_argument("--depth_per_cam_output", type=int, default=32)
    p.add_argument("--pos_output_size", type=int, default=64)
    p.add_argument("--eef_output_size", type=int, default=32)
    p.add_argument("--hand_pos_output_size", type=int, default=96)
    p.add_argument("--efforts_output_size", type=int, default=64)
    p.add_argument("--velocity_output_size", type=int, default=64)
    p.add_argument("--touch_output_size", type=int, default=64)
    p.add_argument("--enable_crop", action=argparse.BooleanOptionalAction, default=False,
                   help="Center-crop images in the encoder (default: off, images kept at original resolution).")
    p.add_argument("--crop_size", default="216,288")
    p.add_argument("--enable_downsample", action=argparse.BooleanOptionalAction, default=False,
                   help="Downsample images in the encoder (default: off, images kept at original resolution).")
    p.add_argument("--downsample_size", default="240,320")

    # --- Instruction conditioning ---
    # Integer-ID mode: embeds a task ID (0..N-1) via nn.Embedding.
    p.add_argument("--use_instruction", action="store_true",
                   help="Condition on integer instruction IDs via nn.Embedding.")
    # Text mode: encodes instruction texts once via a frozen text backbone,
    # then does a lookup + trainable projection at each forward pass.
    # Mutually exclusive with --use_instruction.
    p.add_argument("--use_text_instruction", action="store_true",
                   help="Condition on natural-language instruction texts via a "
                        "pre-trained text encoder (CLIP or sentence-transformers).")
    p.add_argument("--text_encoder", default="clip",
                   choices=["clip", "clip_large", "sentence_transformers"],
                   help="Text backbone for --use_text_instruction. "
                        "'clip' (512-d), 'clip_large' (768-d), "
                        "'sentence_transformers' (384-d).")
    p.add_argument("--instructions_file",
                   default="workflow/instructions.json",
                   help="JSON file mapping instruction IDs to text strings. "
                        "Used only during training; embeddings are stored in "
                        "the checkpoint.")
    p.add_argument("--num_instructions", type=int, default=14)
    p.add_argument("--instruction_embed_dim", type=int, default=128)

    # --- Temporal windows ---
    p.add_argument("--obs_horizon", type=int, default=1)
    p.add_argument("--action_horizon", type=int, default=32)
    p.add_argument("--pred_horizon", type=int, default=64)
    p.add_argument("--action_dim", type=int, default=30)

    # --- Training ---
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Accumulate gradients over N micro-batches before stepping.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--load_img", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--isolate_episodes", action=argparse.BooleanOptionalAction, default=True,
                   help="Force each batch to contain only one episode's samples "
                        "via EpisodeAwareSampler. Default: on.")
    p.add_argument("--gpu", type=str, default="0",
                   help="GPU index or comma-separated list for DataParallel (e.g. '0,3,5')")
    p.add_argument("--lazy_loading", action=argparse.BooleanOptionalAction, default=True,
                   help="Lazy-load images from disk instead of holding everything in RAM. "
                        "Requires uncompressed NPZ files (run workflow/npz_to_npy.py). "
                        "Reduces RAM from ~329GB to ~1GB per process for the full dataset. "
                        "Default: on. Disable with --no-lazy_loading.")
    p.add_argument("--cache_on_gpu", action="store_true",
                   help="Pre-load entire dataset onto GPU. Eliminates CPU→GPU transfer "
                        "overhead. Best with precomputed features (small data).")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to reduce VRAM usage at the cost "
                        "of ~30%% slower compute. Allows larger batch sizes.")
    p.add_argument("--lr_schedule", default="none",
                   choices=["none", "cosine"],
                   help="Learning rate schedule. 'cosine' uses cosine decay with linear warmup.")
    p.add_argument("--warmup_steps", type=int, default=500,
                   help="Number of warmup steps for LR schedule (default: 500).")

    # --- Checkpointing ---
    p.add_argument("--save_path", default="checkpoints/exp")
    p.add_argument("--save_freq", type=int, default=30,
                   help="Save a checkpoint every N epochs.")
    p.add_argument("--eval_freq", type=int, default=30)
    p.add_argument("--resume", default="",
                   help="Path to checkpoint to resume from.")
    p.add_argument("--pretrained_ckpt", default="",
                   help="Path to a pretrained checkpoint (e.g. official RDT-1B from HuggingFace) "
                        "for partial weight loading. Only loads weights whose shapes match; "
                        "mismatched keys (action_dim, lang_dim, etc.) are skipped. "
                        "Supports both RDTRunner (official) and our checkpoint formats.")

    # --- Model-specific (Diffusion Policy) ---
    p.add_argument("--num_diffusion_iters", type=int, default=100)
    p.add_argument("--use_ddim", action="store_true")
    p.add_argument("--diffusion_model_type", default="auto",
                   choices=["auto", "transformer", "unet"])
    p.add_argument("--transformer_hidden_size", type=int, default=256,
                   help="Transformer hidden dim for DP (paper default: 256).")
    p.add_argument("--transformer_depth", type=int, default=8,
                   help="Transformer layers for DP (paper default: 8).")
    p.add_argument("--transformer_num_heads", type=int, default=4,
                   help="Transformer heads for DP (paper default: 4).")
    p.add_argument("--transformer_causal_attn", action=argparse.BooleanOptionalAction,
                   default=True, help="Enable causal attention in DP Transformer.")
    p.add_argument("--transformer_n_cond_layers", type=int, default=4)
    p.add_argument("--dp_cond_mask_prob", type=float, default=0.0,
                   help="Condition masking probability for DP visual features.")
    p.add_argument("--dp_dedicated_instr_token", action="store_true", default=False,
                   help="Give instruction its own condition token in the denoiser "
                        "(instead of concatenating into obs).")

    # --- Model-specific (ACT) ---
    p.add_argument("--act_hidden_dim", type=int, default=512)
    p.add_argument("--act_num_heads", type=int, default=8)
    p.add_argument("--act_latent_dim", type=int, default=32)
    p.add_argument("--act_kl_weight", type=float, default=10.0)
    p.add_argument("--act_cond_mask_prob", type=float, default=0.0,
                   help="Probability of masking visual tokens to force instruction use")

    # --- Model-specific (BAKU) ---
    p.add_argument("--baku_hidden_size", type=int, default=256)
    p.add_argument("--baku_depth", type=int, default=8)
    p.add_argument("--baku_num_heads", type=int, default=4)
    p.add_argument("--baku_ff_dim", type=int, default=0,
                   help="Feed-forward dim (0 = 4 * hidden_size).")
    p.add_argument("--baku_dropout", type=float, default=0.1)
    p.add_argument("--baku_use_film", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable instruction-conditioned FiLM for compatible ResNet visual branches. "
                        "Auto-inactive when no instruction encoder or visual branch is present.")

    # --- Model-specific (RDT) ---
    p.add_argument("--rdt_text_encoder", default="t5_xxl",
                   choices=["t5_small", "t5_base", "t5_large", "t5_xl", "t5_xxl"],
                   help="T5 variant for RDT instruction conditioning.")
    p.add_argument("--rdt_token_max_len", type=int, default=120,
                   help="Max T5 token sequence length for RDT.")
    p.add_argument("--rdt_hidden_size", type=int, default=512,
                   help="Transformer hidden size for RDT.")
    p.add_argument("--rdt_depth", type=int, default=12,
                   help="Number of ACI decoder layers in RDT.")
    p.add_argument("--rdt_num_heads", type=int, default=8,
                   help="Number of attention heads in RDT.")
    p.add_argument("--rdt_ff_dim", type=int, default=0,
                   help="Feed-forward dim in RDT (0 = same as hidden_size, paper: 1x).")
    p.add_argument("--rdt_dropout", type=float, default=0.0)
    p.add_argument("--rdt_num_diffusion_iters", type=int, default=1000,
                   help="Diffusion training steps for RDT (paper: 1000).")
    p.add_argument("--rdt_num_inference_iters", type=int, default=5,
                   help="Diffusion inference steps for RDT (paper: 5 with DPMSolver).")
    p.add_argument("--rdt_inference_scheduler", default="dpmsolver",
                   choices=["dpmsolver", "ddim", "ddpm"],
                   help="Inference scheduler for RDT (paper: dpmsolver).")
    p.add_argument("--rdt_prediction_type", default="sample",
                   choices=["sample", "epsilon"],
                   help="Diffusion prediction type (paper: sample = predict clean x0).")
    p.add_argument("--rdt_cond_mask_prob", type=float, default=0.0,
                   help="Probability of masking conditions during RDT training (paper: 0.1).")
    p.add_argument("--rdt_siglip_raw_dim", type=int, default=1152,
                   help="Raw SigLIP patch token dimension. [default: 1152 for siglip_so400m]")
    p.add_argument("--rdt_siglip_resolution", type=int, default=384,
                   help="SigLIP input resolution. Must be 384 for SO400M.")
    p.add_argument("--rdt_siglip_pool_patches", type=int, default=0,
                   help="Pool SigLIP patches to this count (0=no pooling, 64→8×8). Speeds up cross-attn.")
    p.add_argument("--rdt_prop_dim", type=int, default=30,
                   help="Proprioceptive state dimension (6 arm + 24 hand). [default: 30]")
    p.add_argument("--rdt_ctrl_freq", type=float, default=1.0,
                   help="Control frequency token value (fixed, no ctrl_freq in data). [default: 1.0]")
    p.add_argument("--rdt_max_lang_cond_len", type=int, default=1024,
                   help="Max language condition positional embedding length. [default: 1024]")
    p.add_argument("--rdt_max_img_cond_len", type=int, default=4368,
                   help="Max image condition positional embedding length. [default: 4096]")

    # --- Performance ---
    p.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable automatic mixed precision (bf16 on Ampere+, fp16 otherwise). "
                        "Off by default to match TexasPoker fp32 training.")
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile() for potential speedup (requires PyTorch 2.0+).")

    # --- Logging ---
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="TexasPoker")
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--wandb_exp_name", default="")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _build_obs_enc_config(args: argparse.Namespace) -> ObsEncoderConfig:
    rep = [r for r in args.representation_type.split("-") if r]
    cams = [int(c) for c in args.camera_indices]
    crop_h, crop_w = (int(x) for x in args.crop_size.split(","))
    down_h, down_w = (int(x) for x in args.downsample_size.split(","))
    if args.use_instruction and args.use_text_instruction:
        raise ValueError("--use_instruction and --use_text_instruction are mutually exclusive.")
    if args.precompute_rgb_features and args.rgb_encoder == "resnet18":
        raise ValueError(
            "--precompute_rgb_features cannot be used with --rgb_encoder resnet18. "
            "ResNet18 is trainable/on-the-fly and needs raw RGB frames; unset "
            "feature_dir/feature_dirs for the ResNet DP Transformer run."
        )

    return ObsEncoderConfig(
        representation_type=rep,
        camera_indices=cams,
        rgb_encoder_type=args.rgb_encoder,
        depth_encoder_type=args.depth_encoder,
        freeze_rgb_encoder=args.freeze_rgb_encoder,
        freeze_depth_encoder=args.freeze_depth_encoder,
        precompute_rgb_features=args.precompute_rgb_features,
        fuse_rgbd=args.fuse_rgbd,
        rgb_per_cam_output=args.rgb_per_cam_output,
        depth_per_cam_output=args.depth_per_cam_output,
        pos_output_size=args.pos_output_size,
        eef_output_size=args.eef_output_size,
        hand_pos_output_size=args.hand_pos_output_size,
        efforts_output_size=args.efforts_output_size,
        velocity_output_size=args.velocity_output_size,
        touch_output_size=args.touch_output_size,
        use_instruction=args.use_instruction,
        use_text_instruction=args.use_text_instruction,
        text_encoder_type=args.text_encoder,
        instructions_file=args.instructions_file,
        num_instructions=args.num_instructions,
        instruction_embed_dim=args.instruction_embed_dim,
        enable_crop=args.enable_crop,
        crop_size=(crop_h, crop_w),
        enable_downsample=args.enable_downsample,
        downsample_size=(down_h, down_w),
    )


def _build_model_config(args: argparse.Namespace, model_name: str):
    base = dict(
        action_dim=args.action_dim,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        pred_horizon=args.pred_horizon,
        use_instruction=args.use_instruction,
        use_text_instruction=args.use_text_instruction,
        num_instructions=args.num_instructions,
        instruction_embed_dim=args.instruction_embed_dim,
    )
    if model_name == "diffusion_policy":
        return DiffusionPolicyConfig(
            **base,
            num_diffusion_iters=args.num_diffusion_iters,
            use_ddim=args.use_ddim,
            diffusion_model_type=args.diffusion_model_type,
            transformer_hidden_size=args.transformer_hidden_size,
            transformer_depth=args.transformer_depth,
            transformer_num_heads=args.transformer_num_heads,
            transformer_causal_attn=args.transformer_causal_attn,
            transformer_n_cond_layers=args.transformer_n_cond_layers,
            cond_mask_prob=getattr(args, "dp_cond_mask_prob", 0.0),
            dedicated_instr_token=getattr(args, "dp_dedicated_instr_token", False),
        )
    elif model_name == "act":
        return ACTConfig(
            **base,
            hidden_dim=args.act_hidden_dim,
            num_heads=args.act_num_heads,
            latent_dim=args.act_latent_dim,
            kl_weight=args.act_kl_weight,
            cond_mask_prob=args.act_cond_mask_prob,
        )
    elif model_name == "baku":
        return BakuConfig(
            **base,
            hidden_size=args.baku_hidden_size,
            depth=args.baku_depth,
            num_heads=args.baku_num_heads,
            ff_dim=args.baku_ff_dim,
            dropout=args.baku_dropout,
            use_film=args.baku_use_film,
        )
    elif model_name == "rdt":
        return RDTConfig(
            **base,
            text_encoder_type=args.rdt_text_encoder,
            text_token_max_len=args.rdt_token_max_len,
            instructions_file=args.instructions_file,
            hidden_size=args.rdt_hidden_size,
            depth=args.rdt_depth,
            num_heads=args.rdt_num_heads,
            ff_dim=args.rdt_ff_dim,
            dropout=args.rdt_dropout,
            num_diffusion_iters=args.rdt_num_diffusion_iters,
            num_inference_iters=args.rdt_num_inference_iters,
            inference_scheduler=args.rdt_inference_scheduler,
            prediction_type=args.rdt_prediction_type,
            cond_mask_prob=args.rdt_cond_mask_prob,
            siglip_raw_dim=args.rdt_siglip_raw_dim,
            siglip_resolution=args.rdt_siglip_resolution,
            siglip_pool_patches=args.rdt_siglip_pool_patches,
            prop_dim=args.rdt_prop_dim,
            ctrl_freq=args.rdt_ctrl_freq,
            max_lang_cond_len=args.rdt_max_lang_cond_len,
            max_img_cond_len=args.rdt_max_img_cond_len,
        )
    else:
        # Generic fallback — model must accept ModelConfig fields.
        from learning.base import ModelConfig
        return ModelConfig(**base)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _ema_scope(model) -> tuple:
    """Determine the correct parameter scope for EMA store/copy_to/restore.

    EMA shadow count must match the parameter iterator length. Models that
    register EMA over all parameters must pass `model.parameters()`. Models
    that scope EMA to a sub-network must pass the matching sub-network params.

    Returns ``(params_iterator, scope_name)`` where *scope_name* is "all" or
    the sub-network attribute name (used for logging/ema_model.pt metadata).
    """
    n_ema = len(model.ema.shadow_params)
    n_all = sum(1 for _ in model.parameters())
    if n_ema == n_all:
        return model.parameters(), "all"

    _ema_net_names = ("noise_pred_net", "transformer")
    for attr in _ema_net_names:
        if hasattr(model, attr):
            sub = getattr(model, attr)
            n_sub = sum(1 for _ in sub.parameters())
            if n_sub == n_ema:
                return sub.parameters(), attr
    # Fallback: all params (may warn if mismatch)
    return model.parameters(), "all"


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
    epoch: int,
    model_name: str,
    obs_enc_cfg: ObsEncoderConfig,
    model_cfg,
    norm_stats,
    args: argparse.Namespace | None = None,
) -> None:
    import json, pickle

    path.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_type": model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_states": [o.state_dict() for o in optimizers],
        "obs_encoder_config": dataclasses.asdict(obs_enc_cfg),
        "model_config": dataclasses.asdict(model_cfg),
        "norm_stats": stats_to_json(norm_stats),
    }
    # Save AMP dtype so deploy_policy can match training precision.
    if args is not None and getattr(args, "use_amp", False):
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ckpt["amp_dtype"] = str(amp_dtype).split(".")[-1]  # "bfloat16" or "float16"
    # Save EMA shadow parameters when available (used by DiffusionPolicy, RDT).
    has_ema = hasattr(model, "ema") and hasattr(model.ema, "state_dict")
    if has_ema:
        ckpt["ema_state_dict"] = model.ema.state_dict()
    torch.save(ckpt, path / f"epoch_{epoch:04d}.pt")
    # Always overwrite the "latest" pointer.
    torch.save(ckpt, path / "latest.pt")

    # ---- Save companion files (written once, overwritten each save) ----

    # args.json — full training arguments for reproducibility
    if args is not None:
        args_dict = vars(args)
        args_dict["_epoch"] = epoch
        with open(path / "args.json", "w") as f:
            json.dump(args_dict, f, indent=2, default=str)

    # stats.pkl — normalization statistics (pickle for numpy arrays)
    with open(path / "stats.pkl", "wb") as f:
        pickle.dump(norm_stats, f)

    # stats.json — human-readable normalization statistics
    with open(path / "stats.json", "w") as f:
        json.dump(stats_to_json(norm_stats), f, indent=2)

    # ema_model.pt — standalone EMA weights for easy loading
    if has_ema:
        ema_params, ema_net_name = _ema_scope(model)

        if ema_params is not None:
            # Swap in EMA weights, save, swap back
            model.ema.store(ema_params)
            model.ema.copy_to(ema_params)
            try:
                ema_ckpt = {
                    "epoch": epoch,
                    "model_type": model_name,
                    "ema_network": ema_net_name,
                    "model_state_dict": model.state_dict(),
                    "obs_encoder_config": dataclasses.asdict(obs_enc_cfg),
                    "model_config": dataclasses.asdict(model_cfg),
                    "norm_stats": stats_to_json(norm_stats),
                }
                if args is not None and getattr(args, "use_amp", False):
                    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    ema_ckpt["amp_dtype"] = str(amp_dtype).split(".")[-1]
                torch.save(ema_ckpt, path / "ema_model.pt")
            finally:
                model.ema.restore(ema_params)

    print(f"  Saved checkpoint → {path}/epoch_{epoch:04d}.pt")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(model, val_loader, device,
              task_names: dict[int, str] | None = None,
              amp_dtype=torch.float32, amp_enabled=False,
              data_on_gpu: bool = False,
              ) -> dict[str, float]:
    """Evaluate *model* on *val_loader* and return averaged metrics.

    When the model has EMA weights (DiffusionPolicy, RDT),
    validation is run **twice**: once with live weights and once with EMA
    weights.  EMA metrics are logged under ``ema/`` prefix so they can be
    compared directly with deployment behavior (which uses EMA weights).

    When batches contain an ``"instruction"`` key **and** *task_names* is
    provided, per-task losses are additionally computed and returned with
    keys like ``"task/pick_up_left/loss"``.
    """
    model.eval()
    torch_uint16 = getattr(torch, "uint16", None)

    def _run_val(desc: str) -> tuple[dict[str, float], int,
                                      dict[int, dict[str, float]],
                                      dict[int, int]]:
        totals: dict[str, float] = {}
        count = 0
        per_task_totals: dict[int, dict[str, float]] = {}
        per_task_counts: dict[int, int] = {}

        # Use full reverse-diffusion eval when the model provides it
        # (all models implement compute_val_loss).
        _use_val_loss = hasattr(model, "compute_val_loss")

        val_bar = tqdm(val_loader, desc=desc, unit="batch",
                       leave=False, dynamic_ncols=True)
        for batch in val_bar:
            if not data_on_gpu:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss_dict = (model.compute_val_loss(batch)
                             if _use_val_loss else model.compute_loss(batch))
            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            count += 1

            if task_names and "instruction" in batch:
                instr_ids = batch["instruction"]
                for tid in instr_ids.unique().tolist():
                    tid = int(tid)
                    mask = instr_ids == tid
                    sub_batch = {k: (v.to(torch.int32)[mask]
                                     if torch_uint16 is not None and v.dtype == torch_uint16
                                     else v[mask])
                                 for k, v in batch.items()}
                    with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                        sub_loss = (model.compute_val_loss(sub_batch)
                                    if _use_val_loss else model.compute_loss(sub_batch))
                    if tid not in per_task_totals:
                        per_task_totals[tid] = {}
                        per_task_counts[tid] = 0
                    per_task_counts[tid] += 1
                    for k, v in sub_loss.items():
                        per_task_totals[tid][k] = per_task_totals[tid].get(k, 0.0) + float(v)

            val_bar.set_postfix({k: f"{v / count:.4f}" for k, v in totals.items()
                                 if k in ("loss", "arm_mse", "hand_mse")})
        val_bar.close()
        return totals, count, per_task_totals, per_task_counts

    # ---- Run with live weights ----
    totals, count, per_task_totals, per_task_counts = _run_val("  Val")

    # ---- Run with EMA weights (matches deployment behavior) ----
    has_ema = hasattr(model, "ema") and model.ema is not None
    ema_totals: dict[str, float] = {}
    ema_count = 0
    ema_task_totals: dict[int, dict[str, float]] = {}
    ema_task_counts: dict[int, int] = {}

    if has_ema:
        ema_params, _ = _ema_scope(model)

        model.ema.store(ema_params)
        model.ema.copy_to(ema_params)
        try:
            ema_totals, ema_count, ema_task_totals, ema_task_counts = _run_val("  Val(EMA)")
        finally:
            model.ema.restore(ema_params)

    model.train()

    # ---- Assemble metrics ----
    metrics = {k: v / max(count, 1) for k, v in totals.items()}

    if task_names:
        for tid, losses in per_task_totals.items():
            name = task_names.get(tid, f"task_{tid}")
            tc = max(per_task_counts.get(tid, 1), 1)
            for k, v in losses.items():
                metrics[f"task/{name}/{k}"] = v / tc

    if has_ema and ema_count > 0:
        for k, v in ema_totals.items():
            metrics[f"ema/{k}"] = v / max(ema_count, 1)
        if task_names:
            for tid, losses in ema_task_totals.items():
                name = task_names.get(tid, f"task_{tid}")
                tc = max(ema_task_counts.get(tid, 1), 1)
                for k, v in losses.items():
                    metrics[f"ema/task/{name}/{k}"] = v / tc

    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # ---- Device ----
    # DDP: launched via torchrun — LOCAL_RANK is set automatically.
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        # Use a long NCCL timeout so rank imbalance during validation
        # (full reverse diffusion for EMA val can take >10 min) does not
        # trigger the watchdog and abort training at the epoch boundary.
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        is_main = (rank == 0)
        if is_main:
            print(f"DDP: {world_size} GPUs, rank {rank}, device {device}")
    else:
        gpu_ids = [int(g) for g in args.gpu.split(",")]
        device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
        is_main = True
        world_size = 1
        print(f"Using device: {device}")
    multi_gpu = False  # DataParallel no longer used

    # ---- Dataset ----
    rep = [r for r in args.representation_type.split("-") if r]
    cams = [int(c) for c in args.camera_indices]
    ds_cfg = DatasetConfig(
        representation_type=rep,
        camera_indices=cams,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_horizon=args.action_horizon,
        load_img=args.load_img,
        isolate_episodes=args.isolate_episodes,
        n_load_workers=args.num_workers,
    )

    # Choose eager vs lazy dataset builders.
    if args.lazy_loading:
        print("[lazy_loading] Images will be loaded on-demand from disk (low RAM).")
        _build_single = build_dataset_lazy
        _build_multi  = build_multitask_dataset_lazy
    else:
        _build_single = build_dataset
        _build_multi  = build_multitask_dataset

    if args.multitask:
        train_dirs = [p for p in args.train_paths.split(";") if p]
        val_dirs   = [p for p in args.val_paths.split(";")   if p]

        # Optional per-task feature directories (None = no precomputed features).
        def _split_feat_dirs(s: str, n: int):
            parts = [p for p in s.split(";") if p]
            if not parts:
                return None
            return [Path(p) for p in parts] + [None] * (n - len(parts))

        train_feat_dirs = _split_feat_dirs(args.feature_dirs, len(train_dirs))
        val_feat_dirs   = _split_feat_dirs(args.val_feature_dirs or args.feature_dirs,
                                           len(val_dirs))

        print(f"Loading training data ({len(train_dirs)} tasks) …")
        train_ds = _build_multi(train_dirs, ds_cfg,
                                feature_dirs=train_feat_dirs)
        print(f"Loading validation data ({len(val_dirs)} tasks) …")
        val_ds   = _build_multi(val_dirs, ds_cfg,
                                norm_stats=train_ds.norm_stats,
                                feature_dirs=val_feat_dirs)
    else:
        train_feat = Path(args.feature_dir) if args.feature_dir else None
        val_feat   = Path(args.val_feature_dir or args.feature_dir) if \
                     (args.val_feature_dir or args.feature_dir) else None

        train_cfg = dataclasses.replace(ds_cfg, feature_dir=train_feat)
        print(f"Loading training data from {args.train_path} …")
        train_ds = _build_single(args.train_path, train_cfg)

        val_cfg = dataclasses.replace(ds_cfg, feature_dir=val_feat)
        print(f"Loading validation data from {args.val_path} …")
        val_ds  = _build_single(args.val_path, val_cfg, norm_stats=train_ds.norm_stats)

    norm_stats = train_ds.norm_stats
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    # DDP: use DistributedSampler so each GPU gets a unique data shard.
    train_sampler = None
    val_sampler = None
    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = create_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        use_episode_sampler=args.isolate_episodes if not ddp else False,
        sampler=train_sampler,
        pin_memory=args.pin_memory,
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=val_sampler,
        pin_memory=args.pin_memory,
    )

    # ---- Optional GPU caching ----
    # Pre-load all batches onto GPU to eliminate CPU→GPU transfer each step.
    gpu_cached_train = None
    gpu_cached_val = None
    if args.cache_on_gpu and device.type == "cuda":
        print("Caching training data on GPU ...")
        gpu_cached_train = [
            {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for batch in tqdm(train_loader, desc="  Cache train", leave=False)
        ]
        print("Caching validation data on GPU ...")
        gpu_cached_val = [
            {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for batch in tqdm(val_loader, desc="  Cache val", leave=False)
        ]
        torch.cuda.synchronize(device)
        mem_gb = torch.cuda.memory_allocated(device) / 1024**3
        print(f"GPU cache: {len(gpu_cached_train)} train + {len(gpu_cached_val)} val batches "
              f"({mem_gb:.1f} GB)")

    # ---- Per-task names for validation breakdown ----
    task_names: dict[int, str] | None = None
    if args.multitask and args.instructions_file:
        try:
            with open(args.instructions_file, encoding="utf-8") as f:
                instr_data = json.load(f)
            task_names = {int(k): v["name"] for k, v in instr_data.items()}
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            task_names = None

    # ---- Model ----
    obs_enc_cfg = _build_obs_enc_config(args)
    obs_encoder = ObsEncoder(obs_enc_cfg)
    model_cfg   = _build_model_config(args, args.model)
    model = build_model(args.model, obs_encoder=obs_encoder, config=model_cfg)
    model.norm_stats = norm_stats
    if "depth" in norm_stats:
        obs_encoder.set_depth_stats(norm_stats["depth"])

    # Enable gradient checkpointing if requested (reduces VRAM, allows larger batch).
    if args.gradient_checkpointing:
        from learning.rdt.model import _ACIDecoder
        for m in model.modules():
            if isinstance(m, _ACIDecoder):
                m.gradient_checkpointing = True
                if is_main:
                    print(f"Gradient checkpointing enabled on {m.__class__.__name__}")

    model = model.to(device)
    # EMAModel is not an nn.Module, so .to(device) above won't move its
    # shadow parameters — move them explicitly.
    if hasattr(model, "ema"):
        model.ema.to(device)

    # ---- Optimizers ----
    optimizers = model.configure_optimizers(lr=args.lr, weight_decay=args.weight_decay)

    # ---- LR scheduler (match TexasPoker: diffusers cosine scheduler) ----
    lr_schedulers = []
    if args.lr_schedule == "cosine":
        from diffusers.optimization import get_scheduler as _get_lr_scheduler
        steps_per_epoch = max(len(train_loader), 1)
        total_steps = steps_per_epoch * args.epochs
        for opt in optimizers:
            lr_schedulers.append(_get_lr_scheduler(
                name="cosine",
                optimizer=opt,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_steps,
            ))
        if is_main:
            print(f"LR schedule: cosine with {args.warmup_steps} warmup steps, "
                  f"{total_steps} total steps")

    # ---- AMP (automatic mixed precision) ----
    amp_enabled = args.use_amp and device.type == "cuda"
    if amp_enabled:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"AMP enabled: {amp_dtype}")
    else:
        amp_dtype = torch.float32
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=scaler_enabled)
    elif device.type == "cuda" and hasattr(torch.cuda, "amp"):
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    else:
        class _NoOpGradScaler:
            def scale(self, loss):
                return loss

            def unscale_(self, optimizer) -> None:
                return None

            def step(self, optimizer) -> None:
                optimizer.step()

            def update(self) -> None:
                return None

        scaler = _NoOpGradScaler()

    # ---- Multi-GPU wrapping ----
    raw_model = model  # keep reference to unwrapped model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)
        if is_main:
            print(f"Wrapped model in DDP on {world_size} GPUs")

    # ---- Optional torch.compile ----
    if args.compile and hasattr(torch, "compile"):
        torch._dynamo.config.optimize_ddp = False  # avoid higher-order-op issue with grad ckpt
        inner = raw_model
        compiled_parts = []
        # RDT: compile the denoising transformer (skip frozen SigLIP)
        if hasattr(inner, "transformer") and hasattr(inner.transformer, "aci_decoder"):
            inner.transformer = torch.compile(inner.transformer)
            compiled_parts.append("transformer")
        # DiffusionPolicy: compile the noise prediction network
        if hasattr(inner, "noise_pred_net"):
            inner.noise_pred_net = torch.compile(inner.noise_pred_net)
            compiled_parts.append("noise_pred_net")
        # ACT: compile encoder and decoder transformers
        if hasattr(inner, "policy_enc"):
            inner.policy_enc = torch.compile(inner.policy_enc)
            inner.policy_dec = torch.compile(inner.policy_dec)
            compiled_parts.append("policy_enc+policy_dec")
        if not compiled_parts:
            model = torch.compile(model)
            compiled_parts.append("full model")
        if is_main:
            print(f"torch.compile: compiled {', '.join(compiled_parts)}")

    # ---- Optional resume ----
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        for opt, st in zip(optimizers, ckpt.get("optimizer_states", [])):
            opt.load_state_dict(st)
        if "ema_state_dict" in ckpt and hasattr(raw_model, "ema"):
            raw_model.ema.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        # Fast-forward LR schedulers to the correct step
        if lr_schedulers:
            steps_done = start_epoch * max(len(train_loader), 1)
            for sched in lr_schedulers:
                for _ in range(steps_done):
                    sched.step()
        print(f"Resumed from epoch {start_epoch - 1}")

    # ---- Optional pretrained checkpoint (partial load) ----
    if getattr(args, "pretrained_ckpt", "") and is_main:
        from learning.rdt.model import load_pretrained_rdt
        load_pretrained_rdt(raw_model, args.pretrained_ckpt, device=device)

    # ---- WandB ----
    wandb_run = None
    if args.use_wandb and is_main:
        import wandb
        run_name = args.wandb_exp_name or f"{args.model}_{Path(args.save_path).name}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_name,
            config=vars(args),
        )

    # ---- Training loop ----
    save_path = Path(args.save_path)
    model.train()

    import random as _random

    epoch_bar = tqdm(range(start_epoch, args.epochs), desc="Epochs", unit="epoch",
                     disable=not is_main)
    for epoch in epoch_bar:
        # DDP: set epoch for proper shuffling across ranks.
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_totals: dict[str, float] = {}
        n_batches = 0

        # Use GPU-cached batches when available, otherwise DataLoader.
        if gpu_cached_train is not None:
            _random.shuffle(gpu_cached_train)
            train_iter = gpu_cached_train
        else:
            train_iter = train_loader

        accum_steps = args.gradient_accumulation_steps
        batch_bar = tqdm(train_iter, desc=f"  Train", unit="batch",
                         leave=False, dynamic_ncols=True, disable=not is_main)
        for opt in optimizers:
            opt.zero_grad()
        micro_step = 0
        for batch in batch_bar:
            if gpu_cached_train is None:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss_dict = model(batch)  # forward → compute_loss (works with DDP)
                loss = loss_dict["loss"] / accum_steps

            # Skip NaN/Inf losses to prevent model corruption
            if not torch.isfinite(loss):
                tqdm.write(f"  [WARNING] Non-finite loss detected ({float(loss):.4f}), skipping batch")
                for opt in optimizers:
                    opt.zero_grad()
                micro_step = 0
                continue

            scaler.scale(loss).backward()
            micro_step += 1

            if micro_step % accum_steps == 0:
                for opt in optimizers:
                    scaler.unscale_(opt)
                for opt in optimizers:
                    scaler.step(opt)
                scaler.update()
                for opt in optimizers:
                    opt.zero_grad()
                for sched in lr_schedulers:
                    sched.step()
                raw_model.on_after_step()

            for k, v in loss_dict.items():
                epoch_totals[k] = epoch_totals.get(k, 0.0) + float(v) * accum_steps
            n_batches += 1

            batch_bar.set_postfix({k: f"{float(v) * accum_steps:.4f}" for k, v in loss_dict.items()})

        # Flush leftover accumulated gradients at end of epoch.
        if micro_step % accum_steps != 0:
            for opt in optimizers:
                scaler.unscale_(opt)
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()
            for opt in optimizers:
                opt.zero_grad()
            for sched in lr_schedulers:
                sched.step()
            raw_model.on_after_step()

        batch_bar.close()

        # Epoch-level logging.
        epoch_means = {f"train/{k}": v / n_batches for k, v in epoch_totals.items()}
        if wandb_run:
            wandb_run.log({"epoch": epoch, **epoch_means})

        epoch_bar.set_postfix({k.replace("train/", ""): f"{v:.4f}"
                                for k, v in epoch_means.items()})

        # Validation (all ranks compute, but only main logs).
        if (epoch + 1) % args.eval_freq == 0 and is_main:
            val_metrics = _evaluate(raw_model,
                                    gpu_cached_val or val_loader,
                                    device,
                                    task_names=task_names,
                                    amp_dtype=amp_dtype,
                                    amp_enabled=amp_enabled,
                                    data_on_gpu=gpu_cached_val is not None)
            val_log = {f"val/{k}": v for k, v in val_metrics.items()
                       if not k.startswith("task/") and not k.startswith("ema/")}
            ema_log = {f"val/{k}": v for k, v in val_metrics.items()
                       if k.startswith("ema/") and not k.startswith("ema/task/")}
            task_log = {f"val/{k}": v for k, v in val_metrics.items()
                        if k.startswith("task/") or k.startswith("ema/task/")}
            if wandb_run:
                wandb_run.log({"epoch": epoch, **val_log, **ema_log, **task_log})
            val_str = "  ".join(f"{k}={v:.4f}" for k, v in val_log.items())
            tqdm.write(f"[Epoch {epoch:4d}]  Val: {val_str}")
            if ema_log:
                ema_str = "  ".join(f"{k}={v:.4f}" for k, v in ema_log.items())
                tqdm.write(f"              EMA: {ema_str}")
            if task_log:
                task_str = "  ".join(f"{k}={v:.4f}" for k, v in task_log.items())
                tqdm.write(f"              Per-task: {task_str}")

        # Checkpoint (only main rank saves).
        if is_main and ((epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1):
            _save_checkpoint(
                save_path, raw_model, optimizers, epoch,
                args.model, obs_enc_cfg, model_cfg, norm_stats,
                args=args,
            )

        # DDP barrier: wait for all ranks before next epoch.
        if ddp:
            dist.barrier()

    if wandb_run:
        wandb_run.finish()
    if ddp:
        dist.destroy_process_group()
    if is_main:
        tqdm.write("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.list_models:
        print("Available models:", list_models())
        sys.exit(0)

    # Validate required paths.
    if args.multitask:
        if not args.train_paths or not args.val_paths:
            print("ERROR: --multitask requires --train_paths and --val_paths.")
            sys.exit(1)
    else:
        if not args.train_path or not args.val_path:
            print("ERROR: Provide --train_path and --val_path (or use --multitask).")
            sys.exit(1)

    train(args)


if __name__ == "__main__":
    main()
