"""
ZeroMQ inference server for trained policy models.

Loads a checkpoint saved by train.py and serves action predictions to
robot_client.py over a REQ/REP socket on port 13579 by default.

Protocol
--------
1. Client sends:  {"type": "config_request", "timestamp": <float>}
   Server replies: {"config": {obs_horizon, action_horizon, use_instruction,
                               instruction_dim, camera_indices, predict_pos_delta,
                               clip_depth_max, depth_max_value}}

2. Client sends:  {"observation": [obs_t0, obs_t1, ...], "timestamp": <float>}
   Server replies: {"action": <list[list[float]]>}   shape (action_horizon, action_dim)
   On error:      {"error": <str>}

Each obs dict in the list has keys:
    images_cam{i}    float32 list (H, W, 3)   — BGR 0-255 range, matching training data
    depth_cam{i}     float32 list (H, W)
    joint_positions  list (30,)
    joint_efforts    list (30,)
    joint_velocities list (30,)
    instruction      list (num_instructions,)  — one-hot or zeros

Usage
-----
    python deploy_policy.py --ckpt checkpoints/dp_exp1/latest.pt --port 13579
    python deploy_policy.py --ckpt checkpoints/dp_exp1/latest.pt --capture-request-dir captured_robot_requests
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zmq

# Register all models before loading the checkpoint.
import learning  # noqa: F401

from learning.registry import build_model
from learning.common.encoders import ObsEncoder, ObsEncoderConfig, _make_resnet18
from data_processing.normalization import normalize_data, stats_from_json


# ---------------------------------------------------------------------------
# EMA shadow re-mapping for precomputed→on-the-fly deploy
# ---------------------------------------------------------------------------


class _NoOpEMA:
    """Drop-in replacement for EMA at deploy time.

    After baking EMA shadows into model weights, predict_action() still
    calls ema.store/copy_to/restore.  This no-op avoids touching every
    model's predict_action code.
    """

    def store(self, params):
        pass

    def copy_to(self, params):
        pass

    def restore(self, params):
        pass


class _LegacySpatialSoftmax(nn.Module):
    """Legacy spatial-softmax head used by older fused RGBD checkpoints."""

    def __init__(self, in_c: int, in_h: int, in_w: int, num_kp: int):
        super().__init__()
        self._spatial_conv = nn.Conv2d(in_c, num_kp, kernel_size=1)
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, in_w).float(),
            torch.linspace(-1, 1, in_h).float(),
            indexing="ij",
        )
        self.register_buffer("pos_x", pos_x.reshape(1, in_w * in_h))
        self.register_buffer("pos_y", pos_y.reshape(1, in_w * in_h))
        self._num_kp = num_kp
        self._in_c = in_c
        self._in_h = in_h
        self._in_w = in_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1:] != (self._in_c, self._in_h, self._in_w):
            raise ValueError(
                "Legacy spatial-softmax expected feature map shape "
                f"(*, {self._in_c}, {self._in_h}, {self._in_w}), got {tuple(x.shape)}."
            )
        h = self._spatial_conv(x).contiguous().view(-1, self._in_h * self._in_w)
        attention = F.softmax(h, dim=-1)
        keypoint_x = (self.pos_x * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        keypoint_y = (self.pos_y * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        return torch.cat([keypoint_x, keypoint_y], dim=1)


class _LegacySpatialProjection(nn.Module):
    """SpatialSoftmax + Linear projection used by older ResNet RGBD encoders."""

    def __init__(self, input_shape: tuple[int, int, int], out_dim: int):
        super().__init__()
        in_c, in_h, in_w = input_shape
        num_kp = out_dim // 2
        self.out_dim = out_dim
        self.spatial_softmax = _LegacySpatialSoftmax(in_c, in_h, in_w, num_kp=num_kp)
        self.projection = nn.Linear(num_kp * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.spatial_softmax(x))


class _LegacyRGBDBackbone(nn.Module):
    """Legacy 4-channel ResNet trunk that stops after the second ResNet stage."""

    def __init__(self):
        super().__init__()
        net = _make_resnet18(in_channels=4)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.block1 = net.layer1[0]
        self.block2 = net.layer1[1]
        self.block3 = net.layer2[0]
        self.block4 = net.layer2[1]
        self.stage_modules = [self.block1, self.block2, self.block3, self.block4]
        self.projection_type = "spatial"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for block in self.stage_modules:
            x = block(x)
        return x


class _LegacyRGBDEncoder(nn.Module):
    """Deploy-time compatibility encoder for older fused RGBD checkpoints."""

    film_stage_channels = (64, 64, 128, 128)

    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.backbone = _LegacyRGBDBackbone()
        self.depth_min: float | None = None
        self.depth_max: float | None = None

        with torch.no_grad():
            dummy = torch.zeros(1, 4, 216, 288)
            output_shape = tuple(self.backbone(dummy).shape[1:])
        self.proj = _LegacySpatialProjection(output_shape, output_size)

    def set_depth_stats(self, d_min: float, d_max: float) -> None:
        self.depth_min = d_min
        self.depth_max = d_max

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        leading = rgb.shape[:-3]
        h, w = rgb.shape[-3], rgb.shape[-2]
        rgb_f = rgb.reshape(-1, h, w, 3).float() / 255.0
        rgb_chw = rgb_f.permute(0, 3, 1, 2)
        dep_f = depth.reshape(-1, 1, h, w).float()
        if self.depth_min is not None and self.depth_max is not None:
            eps = 1e-8
            dep_f = (dep_f - self.depth_min) / (self.depth_max - self.depth_min + eps) * 2 - 1
        img = torch.cat([rgb_chw, dep_f], dim=1)
        img = F.interpolate(img, size=(240, 320), mode="bilinear", align_corners=False)
        if self.training:
            i = torch.randint(0, 240 - 216 + 1, (1,), device=img.device).item()
            j = torch.randint(0, 320 - 288 + 1, (1,), device=img.device).item()
            img = img[:, :, i:i + 216, j:j + 288]
        else:
            img = img[:, :, 12:228, 16:304]
        feat = self.backbone(img)
        return self.proj(feat).reshape(*leading, self.output_size)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _filter_dataclass_kwargs(cfg_cls, cfg_dict: dict, label: str) -> dict:
    """Drop legacy checkpoint keys that are not accepted by *cfg_cls*."""
    valid_fields = {f.name for f in dataclasses.fields(cfg_cls)}
    dropped = sorted(set(cfg_dict) - valid_fields)
    if dropped:
        print(f"  Ignoring legacy {label} keys not used by current code: {dropped}")
    return {k: v for k, v in cfg_dict.items() if k in valid_fields}


def _maybe_use_legacy_rgbd_encoder(obs_encoder: ObsEncoder, saved_cfg: dict) -> None:
    """Swap in the older spatial-projection RGBD encoder when required."""
    use_legacy = (
        saved_cfg.get("fuse_rgbd")
        and saved_cfg.get("rgb_encoder_type") == "resnet18"
        and saved_cfg.get("resnet_projection") == "spatial"
    )
    if not use_legacy:
        return

    output_size = (
        obs_encoder.config.rgb_per_cam_output + obs_encoder.config.depth_per_cam_output
    )
    obs_encoder.rgbd_encoders = nn.ModuleList([
        _LegacyRGBDEncoder(output_size=output_size)
        for _ in obs_encoder.config.camera_indices
    ])
    obs_encoder.rgb_encoders = None
    obs_encoder.depth_encoders = None
    print("  Using legacy fused RGBD encoder compatibility path (spatial projection).")


def _expand_shared_camera_encoder_state_dict(
    state_dict: dict[str, torch.Tensor],
    saved_cfg: dict,
) -> dict[str, torch.Tensor]:
    """Replicate camera-0 encoder weights for legacy shared-camera checkpoints."""
    if not saved_cfg.get("share_camera_encoders"):
        return state_dict

    camera_indices = saved_cfg.get("camera_indices") or []
    num_cams = len(camera_indices)
    if num_cams <= 1:
        return state_dict

    expanded = dict(state_dict)
    for prefix in ("rgbd_encoders", "rgb_encoders", "depth_encoders"):
        base = f"obs_encoder.{prefix}.0."
        if not any(k.startswith(base) for k in state_dict):
            continue
        if any(k.startswith(f"obs_encoder.{prefix}.1.") for k in state_dict):
            continue
        for cam_idx in range(1, num_cams):
            for key, value in state_dict.items():
                if key.startswith(base):
                    expanded[key.replace(".0.", f".{cam_idx}.", 1)] = value.clone()
        print(f"  Expanded shared {prefix} weights from camera 0 to {num_cams} cameras.")
    return expanded


def load_checkpoint(ckpt_path: str, device: torch.device):
    """
    Load a checkpoint written by train.py and reconstruct the model.

    Accepts either:
    - ``latest.pt`` / ``epoch_XXXX.pt`` — standard checkpoint (live weights +
      EMA shadow params loaded separately).
    - ``ema_model.pt`` — standalone EMA checkpoint where model_state_dict
      already contains EMA-smoothed weights.  predict_action() works
      directly without swapping weights.

    Returns:
        model:        PolicyModel in eval mode
        obs_enc_cfg:  ObsEncoderConfig (for introspecting camera_indices etc.)
        norm_stats:   NormStats dict for proprioception normalization
        amp_dtype:    torch.dtype used during training (bfloat16/float16/float32)
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # --- Reconstruct ObsEncoder ---
    saved_obs_enc_cfg = ckpt["obs_encoder_config"].copy()
    enc_cfg_dict = saved_obs_enc_cfg.copy()
    # Clear instructions_file so TextInstructionEncoder is built with zero
    # embeddings rather than re-running CLIP.  The actual embeddings are
    # restored from the checkpoint's model_state_dict via load_state_dict.
    enc_cfg_dict["instructions_file"] = ""
    # Keep precompute_rgb_features as-is from training so the model
    # architecture (proj layers etc.) is identical to what was trained.
    # After loading weights we attach the backbone for on-the-fly inference.
    # Remove deploy_depth_size from saved config if present (old checkpoints
    # won't have it; avoid passing None explicitly to the dataclass).
    enc_cfg_dict.pop("deploy_depth_size", None)
    enc_cfg_dict = _filter_dataclass_kwargs(
        ObsEncoderConfig, enc_cfg_dict, "obs_encoder_config",
    )
    obs_enc_cfg = ObsEncoderConfig(**enc_cfg_dict)
    obs_encoder = ObsEncoder(obs_enc_cfg)
    _maybe_use_legacy_rgbd_encoder(obs_encoder, saved_obs_enc_cfg)
    obs_encoder = obs_encoder.to(device)

    # --- Reconstruct model ---
    model_type = ckpt["model_type"]
    sd = _expand_shared_camera_encoder_state_dict(
        ckpt["model_state_dict"], saved_obs_enc_cfg,
    )
    model_cfg_dict = ckpt["model_config"].copy()
    if model_type == "baku" and "film_stage_channels" not in model_cfg_dict:
        film_stage_channels = _infer_baku_film_stage_channels(sd)
        if film_stage_channels is not None:
            model_cfg_dict["film_stage_channels"] = film_stage_channels

    # Import model-specific config class via the registry.
    from learning.registry import _REGISTRY  # noqa: PLC2701
    model_cls = _REGISTRY[model_type]
    cfg_cls = _get_config_class(model_cls)
    model_cfg_dict = _filter_dataclass_kwargs(cfg_cls, model_cfg_dict, "model_config")
    model_cfg = cfg_cls(**model_cfg_dict)

    model = build_model(model_type, obs_encoder=obs_encoder, config=model_cfg)
    was_precomputed = ckpt["obs_encoder_config"].get("precompute_rgb_features", False)

    # --- Clean up state_dict key mismatches ---
    # Strip torch.compile `_orig_mod.` prefix from all keys.
    if any("_orig_mod." in k for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        print("  Stripped _orig_mod. prefix from state_dict keys (torch.compile artifact).")

    model.load_state_dict(sd, strict=True)

    # --- EMA: use ema_model.pt for EMA weights, skip EMA at deploy ---
    # train.py saves ema_model.pt with EMA weights already in model_state_dict.
    # Deploy with --ckpt ema_model.pt to use EMA weights directly.
    # For epoch_*.pt / latest.pt, trained weights are used as-is (non-EMA).
    model.to(device)
    if hasattr(model, "ema"):
        model.ema = _NoOpEMA()

    # --- Attach backbone for on-the-fly inference (precomputed checkpoints) ---
    if was_precomputed and obs_encoder.rgb_encoders is not None:
        for enc in obs_encoder.rgb_encoders:
            enc.attach_backbone()
        if len(obs_encoder.rgb_encoders) > 1:
            shared = obs_encoder.rgb_encoders[0].backbone
            for enc in obs_encoder.rgb_encoders[1:]:
                enc.backbone = shared
        for enc in obs_encoder.rgb_encoders:
            if enc.backbone is not None:
                enc.backbone.to(device)
        print(f"  Attached pretrained backbone ({obs_enc_cfg.rgb_encoder_type}, "
              f"shared across {len(obs_encoder.rgb_encoders)} cameras).")
    model.eval()

    # Attach norm_stats so predict_action can un-normalize actions.
    norm_stats = stats_from_json(ckpt["norm_stats"])
    model.norm_stats = norm_stats
    if "depth" in norm_stats:
        obs_encoder.set_depth_stats(norm_stats["depth"])

    # --- AMP dtype (match training precision at inference) ---
    _amp_str = ckpt.get("amp_dtype")
    if _amp_str == "bfloat16":
        amp_dtype = torch.bfloat16
    elif _amp_str == "float16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32  # no AMP or old checkpoint

    print(f"Loaded model '{model_type}' from {ckpt_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Modalities: {obs_enc_cfg.representation_type}")
    print(f"  Cameras: {obs_enc_cfg.camera_indices}")
    print(f"  ObsEncoder total_dim: {obs_encoder.total_dim}")
    print(f"  AMP dtype: {amp_dtype}")

    # --- Sanity check: norm_stats must be populated for safe deployment ---
    if not norm_stats:
        print("  *** WARNING: norm_stats is EMPTY — model will return normalized "
              "actions in [-1,1] which are NOT valid joint positions! ***",
              file=sys.stderr)
    elif "action" not in norm_stats:
        print("  *** WARNING: norm_stats has no 'action' key — action "
              "denormalization may fail! ***", file=sys.stderr)

    # --- Warn if touch is in representation but unsupported at deploy ---
    if "touch" in obs_enc_cfg.representation_type:
        print("  *** WARNING: Model was trained with 'touch' modality but "
              "deploy_policy does not provide touch data. ***", file=sys.stderr)

    # --- Instruction conditioning diagnostics ---
    _diagnose_instruction_conditioning(model, model_type)

    return model, obs_enc_cfg, norm_stats, amp_dtype


def _diagnose_instruction_conditioning(model, model_type: str):
    """Print diagnostic info about instruction conditioning at load time."""
    if not hasattr(model, "text_encoder"):
        return
    te = model.text_encoder
    raw = te.raw_tokens  # (N, L, D)
    mask = te.pad_mask   # (N, L) True=padding
    N, L, D = raw.shape
    norms = raw.float().norm(dim=-1).mean(dim=-1)  # (N,)

    print(f"\n  [Instruction Diagnostics] T5TextEncoder buffers:")
    print(f"    raw_tokens shape: ({N}, {L}, {D})")
    all_zero = True
    for i in range(N):
        tag = "ZERO" if norms[i] < 1e-6 else "ok"
        if norms[i] >= 1e-6:
            all_zero = False
        print(f"    instr {i:2d}: norm={norms[i]:.4f}  [{tag}]")

    if all_zero:
        print("    *** CRITICAL: All T5 embeddings are ZERO — "
              "instruction conditioning is DISABLED! ***")
    else:
        # Pairwise cosine similarity between mean-pooled embeddings
        flat = raw.float().mean(dim=1)  # (N, D)
        flat_norm = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        cos_sim = flat_norm @ flat_norm.T  # (N, N)
        # Show min/max off-diagonal similarity
        mask_diag = ~torch.eye(N, dtype=torch.bool, device=cos_sim.device)
        off_diag = cos_sim[mask_diag]
        print(f"    Pairwise cosine sim: min={off_diag.min():.4f}, "
              f"max={off_diag.max():.4f}, mean={off_diag.mean():.4f}")
        if off_diag.min() > 0.99:
            print("    *** WARNING: All instruction embeddings are nearly identical! ***")
        else:
            print("    Instruction embeddings appear distinguishable.")


def _get_config_class(model_cls):
    """
    Derive the config dataclass for a model class.

    Convention: if the model class has a ``Config`` inner class or
    the module exports ``<ClassName>Config``, use that.  Otherwise
    fall back to the base ModelConfig.
    """
    import inspect
    module = inspect.getmodule(model_cls)
    cfg_name = model_cls.__name__ + "Config"
    if module is not None and hasattr(module, cfg_name):
        return getattr(module, cfg_name)
    from learning.base import ModelConfig
    return ModelConfig


def _infer_baku_film_stage_channels(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int, int] | None:
    """Infer legacy BAKU FiLM stage widths from checkpoint parameter shapes."""
    for prefix in ("visual_film", "rgb_film", "depth_film"):
        channels: list[int] = []
        idx = 0
        while True:
            key = f"{prefix}.projs.{idx}.weight"
            if key not in state_dict:
                break
            out_features = int(state_dict[key].shape[0])
            if out_features % 2 != 0:
                return None
            channels.append(out_features // 2)
            idx += 1
        if channels:
            if len(channels) == 4:
                return tuple(channels)
            return None
    return None


# ---------------------------------------------------------------------------
# Observation conversion helpers
# ---------------------------------------------------------------------------

# Maps batch key to slice of joint_positions / joint_efforts / joint_velocities.
_EEF_SLICE = slice(0, 6)    # arm joints
_HAND_SLICE = slice(6, 30)  # hand joints


def _obs_list_to_batch(
    obs_list: list[dict],
    obs_enc_cfg: ObsEncoderConfig,
    norm_stats: dict,
    device: torch.device,
    model: object = None,
) -> dict[str, torch.Tensor]:
    """
    Convert a list of raw observation dicts (from robot_client.py) to the
    canonical batch format expected by ObsEncoder / PolicyModel.

    Args:
        obs_list:     List of T raw obs dicts (oldest first).
        obs_enc_cfg:  Encoder config (representation_type, camera_indices …).
        norm_stats:   Normalization statistics from the checkpoint.
        device:       Target device.
        model:        The PolicyModel (used to detect model-level text_encoder).

    Returns:
        Batch dict with all tensors on *device*, batch size B=1.
    """
    rep = obs_enc_cfg.representation_type
    cam_idxs = obs_enc_cfg.camera_indices
    T = len(obs_list)
    batch: dict[str, torch.Tensor] = {}

    # ---- RGB ----
    if "img" in rep:
        # shape: (T, num_cams, H, W, 3) → add B dim → (1, T, num_cams, H, W, 3)
        rgb_frames = []
        for obs in obs_list:
            cam_imgs = []
            for ci in cam_idxs:
                img = np.array(obs[f"images_cam{ci}"], dtype=np.float32)  # (H, W, 3) 0-255
                cam_imgs.append(img)
            rgb_frames.append(np.stack(cam_imgs, axis=0))   # (num_cams, H, W, 3)
        rgb_arr = np.stack(rgb_frames, axis=0)               # (T, num_cams, H, W, 3)
        # The encoder expects uint8 in [0,255]; client already provides float [0,255].
        rgb_tensor = torch.from_numpy(rgb_arr).unsqueeze(0).to(device)  # (1, T, C, H, W, 3)
        # Cast to uint8 — values are already in 0-255 float range.
        batch["rgb"] = rgb_tensor.to(torch.uint8)

    # ---- Depth ----
    if "depth" in rep:
        depth_frames = []
        for obs in obs_list:
            cam_depths = []
            for ci in cam_idxs:
                dep = np.array(obs[f"depth_cam{ci}"], dtype=np.float32)  # (H, W)
                cam_depths.append(dep)
            depth_frames.append(np.stack(cam_depths, axis=0))   # (num_cams, H, W)
        depth_arr = np.stack(depth_frames, axis=0)               # (T, num_cams, H, W)
        batch["depth"] = torch.from_numpy(depth_arr).unsqueeze(0).to(device)

    # ---- Proprioception ----
    # Gather (T, dim) arrays, normalize, then add batch dim.

    def _prop(key: str, raw: np.ndarray):
        """Normalize *raw* (T, dim) using norm_stats[key] and add to batch."""
        if key in norm_stats:
            raw = normalize_data(raw, norm_stats[key])
        else:
            print(f"  WARNING: '{key}' in representation_type but missing from "
                  f"norm_stats — passing raw (un-normalized) values to model.")
        t = torch.from_numpy(raw.astype(np.float32)).unsqueeze(0).to(device)  # (1, T, dim)
        batch[key] = t

    if "pos" in rep:
        pos_arr = np.stack(
            [np.array(o["joint_positions"], dtype=np.float32) for o in obs_list], axis=0
        )  # (T, 30)
        _prop("pos", pos_arr)

    if "eef" in rep:
        eef_arr = np.stack(
            [np.array(o["joint_positions"], dtype=np.float32)[_EEF_SLICE] for o in obs_list], axis=0
        )  # (T, 6)
        _prop("eef", eef_arr)

    if "hand_pos" in rep:
        hand_arr = np.stack(
            [np.array(o["joint_positions"], dtype=np.float32)[_HAND_SLICE] for o in obs_list], axis=0
        )  # (T, 24)
        _prop("hand_pos", hand_arr)

    if "efforts" in rep:
        eff_arr = np.stack(
            [np.array(o["joint_efforts"], dtype=np.float32)[_HAND_SLICE] for o in obs_list], axis=0
        )  # (T, 24)
        _prop("efforts", eff_arr)

    if "velocity" in rep:
        vel_arr = np.stack(
            [np.array(o["joint_velocities"], dtype=np.float32) for o in obs_list], axis=0
        )  # (T, 30)
        _prop("velocity", vel_arr)

    # touch is not provided by the current robot_client — skip silently.

    # ---- Instruction ----
    # Instruction support can come from the ObsEncoder (use_instruction /
    # use_text_instruction) OR from the model itself (RDT has
    # their own text_encoder that consumes batch["instruction"] directly).
    has_instruction = (
        obs_enc_cfg.use_instruction
        or obs_enc_cfg.use_text_instruction
        or (model is not None and hasattr(model, "text_encoder"))
    )
    if has_instruction and "instruction" in obs_list[-1]:
        # Client sends a one-hot vector; encoder expects an integer ID.
        instr_vec = np.array(obs_list[-1]["instruction"], dtype=np.float32)
        if instr_vec.sum() < 0.5:
            print("  WARNING: Instruction vector is all-zeros — defaulting to "
                  "instruction 0. Was --instruction omitted on the client?")
        instr_id = int(np.argmax(instr_vec))
        batch["instruction"] = torch.tensor([instr_id], dtype=torch.long, device=device)

    return batch


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class PolicyServer:
    def __init__(self, ckpt_path: str, port: int = 13579, device_str: str = "auto",
                 bind_ip: str = "192.168.1.200",
                 capture_request_dir: str = "captured_robot_requests",
                 capture_request_limit: int = 100):
        if capture_request_limit < 0:
            raise ValueError("capture_request_limit must be >= 0; use 0 for unlimited.")

        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        print(f"Using device: {self.device}")

        self.model, self.obs_enc_cfg, self.norm_stats, self.amp_dtype = load_checkpoint(
            ckpt_path, self.device
        )
        self.model_cfg = self.model.config
        self._capture_request_dir = Path(capture_request_dir).expanduser()
        self._capture_request_limit = capture_request_limit
        self._capture_request_count = 0
        self._capture_request_dir.mkdir(parents=True, exist_ok=True)
        limit_text = (
            "unlimited"
            if self._capture_request_limit == 0
            else str(self._capture_request_limit)
        )
        print(
            "Capturing raw robot_client observation JSON requests to "
            f"{self._capture_request_dir} (limit={limit_text})."
        )

        # ZeroMQ REP socket.
        self.context = zmq.Context()
        self._bind_ip = bind_ip
        self._port = port
        self._bind_socket()

        # Warm-up inference to trigger CUDA lazy init / JIT compilation.
        self._warmup()

    def _bind_socket(self):
        self.socket = self.context.socket(zmq.REP)
        bind_addr = f"tcp://{self._bind_ip}:{self._port}"
        self.socket.bind(bind_addr)
        print(f"Server listening on {bind_addr}")

    def _warmup(self):
        """Run a dummy inference to warm up CUDA / JIT / memory allocator."""
        print("Running warm-up inference ...")
        cam_idxs = self.obs_enc_cfg.camera_indices
        num_instr = getattr(self.model_cfg, "num_instructions",
                            self.obs_enc_cfg.num_instructions)
        fake_obs = {}
        for ci in cam_idxs:
            fake_obs[f"images_cam{ci}"] = np.random.randint(
                0, 256, (480, 640, 3), dtype=np.uint8).astype(np.float32).tolist()
            fake_obs[f"depth_cam{ci}"] = np.zeros((480, 640), dtype=np.float32).tolist()
        fake_obs["joint_positions"] = np.zeros(30, dtype=np.float32).tolist()
        fake_obs["joint_efforts"] = np.zeros(30, dtype=np.float32).tolist()
        fake_obs["joint_velocities"] = np.zeros(30, dtype=np.float32).tolist()
        instr = np.zeros(num_instr, dtype=np.float32)
        instr[0] = 1.0
        fake_obs["instruction"] = instr.tolist()
        try:
            t0 = time.perf_counter()
            self._predict([fake_obs] * self.model_cfg.obs_horizon)
            dt = time.perf_counter() - t0
            print(f"Warm-up done ({dt*1e3:.0f} ms).")
        except Exception as exc:
            print(f"Warm-up inference failed (non-fatal): {exc}")

    # ------------------------------------------------------------------
    # Config response (sent once when client starts)
    # ------------------------------------------------------------------

    def _build_config_response(self) -> dict:
        cfg = self.model_cfg
        enc = self.obs_enc_cfg
        # Instruction support can come from the ObsEncoder (use_instruction /
        # use_text_instruction) OR from the model itself (RDT has
        # their own text_encoder that consumes batch["instruction"]).
        has_instruction = (
            enc.use_instruction
            or enc.use_text_instruction
            or hasattr(self.model, "text_encoder")
        )
        num_instr = getattr(cfg, "num_instructions", enc.num_instructions)
        return {
            "config": {
                "obs_horizon":     cfg.obs_horizon,
                "action_horizon":  cfg.action_horizon,
                "use_instruction": has_instruction,
                "instruction_dim": num_instr if has_instruction else 1,
                "camera_indices":  enc.camera_indices,
                "predict_pos_delta": False,   # model predicts absolute actions
                "clip_depth_max":  False,
                "depth_max_value": 4000.0,
            }
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict(self, obs_list: list[dict]) -> np.ndarray:
        """
        Convert raw obs_list → canonical batch → call model.predict_action().

        Returns: (action_horizon, action_dim) numpy array, un-normalized.
        """
        batch = _obs_list_to_batch(
            obs_list, self.obs_enc_cfg, self.norm_stats, self.device,
            model=self.model,
        )
        if "instruction" in batch:
            instr_id = batch["instruction"].item()
            if not hasattr(self, "_instr_logged"):
                print(f"  [Instruction] Using instruction ID: {instr_id}")
                if hasattr(self.model, "text_encoder"):
                    tok = self.model.text_encoder.raw_tokens[instr_id]
                    per_token_norm = tok.float().norm(dim=-1).mean()
                    print(f"  [Instruction] T5 mean per-token norm: {per_token_norm:.4f}")
                self._instr_logged = True
        elif hasattr(self.model, "text_encoder"):
            if not hasattr(self, "_no_instr_warned"):
                print("  [Instruction] WARNING: No instruction in batch! "
                      "Text conditioning will be SKIPPED.")
                self._no_instr_warned = True
        use_amp = self.amp_dtype != torch.float32 and self.device.type == "cuda"
        with torch.no_grad(), torch.amp.autocast(
            self.device.type, dtype=self.amp_dtype, enabled=use_amp,
        ):
            actions = self.model.predict_action(batch)  # (1, action_horizon, action_dim)
        return actions[0].float().cpu().numpy()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _capture_request(self, message: dict) -> None:
        """Save the raw JSON request received from robot_client.py."""
        if (
            self._capture_request_limit > 0
            and self._capture_request_count >= self._capture_request_limit
        ):
            return

        capture_idx = self._capture_request_count + 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = (
            self._capture_request_dir
            / f"robot_request_{ts}_{capture_idx:06d}.json"
        )
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(message, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")
        tmp_path.replace(path)
        self._capture_request_count = capture_idx
        print(f"Captured raw robot_client request: {path}")
        if (
            self._capture_request_limit > 0
            and self._capture_request_count >= self._capture_request_limit
        ):
            print("Capture request limit reached; further requests will not be saved.")

    def run(self):
        print("Policy server ready. Waiting for requests...")
        while True:
            try:
                message = self.socket.recv_json()

                # --- Health check ---
                if message.get("type") == "health_check":
                    self.socket.send_json({"status": "ok"})
                    continue

                # --- Config handshake ---
                if message.get("type") == "config_request":
                    self.socket.send_json(self._build_config_response())
                    continue

                # --- Inference request ---
                obs_list = message.get("observation")
                if obs_list is None:
                    self.socket.send_json({"error": "Missing 'observation' key"})
                    continue
                self._capture_request(message)

                t0 = time.perf_counter()
                actions = self._predict(obs_list)
                dt = time.perf_counter() - t0
                print(f"Predicted actions shape={actions.shape}  inference={dt*1e3:.1f}ms")

                self.socket.send_json({"action": actions.tolist()})

            except KeyboardInterrupt:
                print("\nShutting down.")
                break
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {exc}"
                print(f"Error handling request: {err_msg}")
                traceback.print_exc()
                try:
                    self.socket.send_json({"error": err_msg})
                except Exception:
                    # REP socket is in an invalid state (failed to send reply).
                    # Recreate it to avoid permanently breaking the server.
                    print("  Failed to send error reply — resetting socket.",
                          file=sys.stderr)
                    try:
                        self.socket.close()
                    except Exception:
                        pass
                    self._bind_socket()

        self.socket.close()
        self.context.term()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Deploy a trained policy as a ZeroMQ server.")
    p.add_argument("--ckpt", required=True,
                   help="Path to checkpoint file (e.g. checkpoints/dp_exp1/latest.pt).")
    p.add_argument("--port", type=int, default=13579,
                   help="ZeroMQ port to bind.")
    p.add_argument("--device", default="auto",
                   help="Device: 'auto', 'cpu', 'cuda', 'cuda:1', …")
    p.add_argument("--bind", default="192.168.1.200",
                   help="IP address to bind the server (default: 192.168.1.200).")
    p.add_argument("--capture-request-dir", default="captured_robot_requests",
                   help=(
                       "Directory for raw robot_client observation JSON requests. "
                       "Relative paths are created under the current working directory. "
                       "Saved files can be played back directly because the root object "
                       "is the original deploy_policy request."
                   ))
    p.add_argument("--capture-request-limit", type=int, default=10,
                   help=(
                       "How many raw observation requests to save. "
                       "Use 0 for unlimited. Default: 100."
                   ))
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(1)

    server = PolicyServer(str(ckpt), port=args.port, device_str=args.device,
                          bind_ip=args.bind,
                          capture_request_dir=args.capture_request_dir,
                          capture_request_limit=args.capture_request_limit)
    server.run()


if __name__ == "__main__":
    main()
