"""
Offline visual feature extraction (DinoV2 or SigLIP).

Reads every episode (data*.npz or data*/ .npy directory) in a directory tree,
runs a frozen vision backbone on each camera's RGB frames, and writes the
resulting feature vectors to a separate feature directory:

    data_dir/0/pick_up_card_train_50/data0001/       ← untouched
    feature_dir/0/pick_up_card_train_50/data0001/    ← features only

Each feature directory contains per-camera .npy files:
    rgb_features_cam0.npy  (T, raw_dim)          ← DinoV2: (T, D) CLS token
    rgb_features_cam0.npy  (T, N_patches, D)     ← SigLIP:  (T, 729, 1152) patch tokens
    rgb_features_cam1.npy  ...
    ...

After running this script, pass --feature_dir to train.py (or the shell
script) so the encoder loads pre-extracted features instead of running the
heavy backbone every batch.

Encoder choices
---------------
  dinov2_vitl14        — DinoV2 ViT-L/14, 1024-d CLS token       (used by DiffusionPolicy)
  dinov2_vitl14_patch  — DinoV2 ViT-L/14, 256×1024 patch tokens
  siglip_so400m        — SigLIP-SO400M @ 384px, 729×1152 patches (used by RDT)
  dinov2_vits14        — DinoV2 ViT-S/14, 384-d
  dinov2_vitb14        — DinoV2 ViT-B/14, 768-d

SigLIP note
-----------
RDT-1B uses SigLIP patch tokens (the full spatial sequence) as cross-attention
memory in the ACI decoder — NOT the pooled CLS vector.  This script therefore
saves last_hidden_state[:, 1:] (excluding the CLS token) for siglip_so400m,
giving shape (T, 729, 1152) per camera at 384-px input (27×27 patches).

Usage
-----
# DinoV2 features for DiffusionPolicy
python workflow/precompute_features.py \
    --data_dir  data/easy_mode \
    --feature_dir data/vitl14_features

# SigLIP features for RDT
python workflow/precompute_features.py \
    --data_dir  data/easy_mode \
    --feature_dir data/siglip_features \
    --encoder siglip_so400m

# Dry run — print which files would be processed without writing
python workflow/precompute_features.py \
    --data_dir data/easy_mode \
    --feature_dir data/vitl14_features \
    --dry_run

Options
-------
--data_dir        Root directory to search for episodes (data*.npz or data*/ dirs, recursive).
--feature_dir     Root directory to write feature .npy files (mirrors data_dir tree).
--encoder         Backbone variant                                         [default: dinov2_vitl14]
--camera_indices  Which cameras to extract (space-separated ints)          [default: 0 1 2]
--batch_size      Frames per GPU batch                                     [default: 64]
--gpu             CUDA device index (-1 for CPU)                           [default: 0]
--overwrite       Re-extract even if rgb_features_camX already exists      [default: False]
--dry_run         List files without writing anything
--no-fp16         Disable fp16 inference (enabled by default on CUDA for ~2× speedup)

Performance
-----------
fp16 inference is enabled by default on CUDA, giving ~2× throughput on A800/A100.
Features are saved as float16 to halve disk usage (SigLIP: ~2TB→~500GB for full
dataset) while maintaining sufficient precision for frozen encoder outputs.
Use --shard for multi-GPU parallelism.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import natsort
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Backbone dims (output feature dimension per encoder variant)
# ---------------------------------------------------------------------------
_BACKBONE_DIMS: dict[str, int] = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
    "dinov2_vitl14_patch": 1024,   # DinoV2 ViT-L/14 patch tokens
    "siglip_so400m": 1152,         # google/siglip-so400m-patch14-384, patch token dim
}

# Number of patch tokens for encoders that output spatial sequences.
# dinov2 @ 224px with patch_size=14 → 16×16 = 256 patches (CLS excluded).
# siglip_so400m @ 384px → 27×27 = 729 patches (CLS excluded).
_PATCH_COUNTS: dict[str, int] = {
    "dinov2_vitl14_patch": 256,
    "siglip_so400m": 729,
}


def _load_backbone(encoder: str, device: torch.device) -> torch.nn.Module:
    """Download (or load cached) vision backbone and freeze it."""
    print(f"Loading backbone: {encoder} …")
    if encoder.startswith("dinov2"):
        # dinov2_vitl14_patch reuses the same backbone as dinov2_vitl14
        hub_name = encoder.replace("_patch", "")
        model = torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=True)
    elif encoder == "siglip_so400m":
        try:
            from transformers import SiglipVisionModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for SigLIP feature extraction. "
                "Install with: pip install transformers"
            ) from exc
        model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
    else:
        raise ValueError(f"Unknown encoder: {encoder!r}. Choices: {list(_BACKBONE_DIMS)}")
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def _extract(
    backbone: torch.nn.Module,
    images: np.ndarray,       # (T, H, W, 3) uint8
    batch_size: int,
    device: torch.device,
    encoder: str = "dinov2_vitl14",
    use_fp16: bool = True,
) -> np.ndarray:
    """Run backbone on all frames of one camera.

    Returns:
        DinoV2 encoders  →  (T, D)           float32   CLS token
        siglip_so400m    →  (T, N_patches, D) float32   patch tokens (no CLS)
    """
    T = images.shape[0]
    all_feats = []

    is_siglip = encoder == "siglip_so400m"
    is_dinov2_patch = encoder.endswith("_patch")
    amp_dtype = torch.float16 if use_fp16 else torch.float32

    for start in range(0, T, batch_size):
        chunk = images[start : start + batch_size]             # (B, H, W, 3) uint8
        img = torch.from_numpy(chunk).float().to(device)       # (B, H, W, 3)
        img = img.permute(0, 3, 1, 2) / 255.0                  # (B, 3, H, W) [0,1]

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda" and use_fp16)):
            if is_siglip:
                img = F.interpolate(img, size=(384, 384),
                                    mode="bilinear", align_corners=False)
                img = (img - 0.5) / 0.5                            # [-1, 1] (SigLIP norm)
                hidden = backbone(pixel_values=img).last_hidden_state  # (B, 1+N, D)
                feats = hidden[:, 1:]                              # (B, N_patches, 1152)
            elif is_dinov2_patch:
                img = F.interpolate(img, size=(224, 224),
                                    mode="bilinear", align_corners=False)
                out = backbone.forward_features(img)
                if isinstance(out, dict):
                    feats = out["x_norm_patchtokens"]              # (B, N_patches, D)
                elif out.dim() == 3:
                    feats = out[:, 1:]                             # drop CLS → (B, N_patches, D)
                else:
                    raise RuntimeError(f"Unexpected DinoV2 output shape: {out.shape}")
            else:
                img = F.interpolate(img, size=(224, 224),
                                    mode="bilinear", align_corners=False)
                feats = backbone.forward_features(img)
                if isinstance(feats, dict):
                    feats = feats["x_norm_clstoken"]               # (B, D)
                elif feats.dim() == 3:
                    feats = feats[:, 0]                            # CLS token → (B, D)

        all_feats.append(feats.cpu().float().numpy())

    # DinoV2 CLS → (T, D) ; DinoV2 patch → (T, N_patches, D) ; SigLIP → (T, N_patches, D)
    return np.concatenate(all_feats, axis=0)


def _feat_path(data_path: Path, data_dir: Path, feature_dir: Path) -> Path:
    """Return the feature directory path that mirrors data_path under feature_dir.

    Works for both .npz files and .npy directories:
        data0001.npz → feature_dir/.../data0001/
        data0001/    → feature_dir/.../data0001/
    """
    rel = data_path.relative_to(data_dir)
    # Strip .npz suffix so result is always a directory path
    if rel.suffix == ".npz":
        rel = rel.with_suffix("")
    return feature_dir / rel


def _find_episodes(data_dir: Path) -> list[Path]:
    """Find all episode paths (data*.npz or data*/ directories) recursively."""
    npz_files = list(data_dir.rglob("data*.npz"))
    npy_dirs = [
        d for d in data_dir.rglob("data*")
        if d.is_dir() and (d / "images_cam0.npy").exists()
    ]
    # Deduplicate: if both data0001.npz and data0001/ exist, prefer the directory
    seen = set()
    episodes = []
    for d in npy_dirs:
        seen.add(d.stem)
        episodes.append(d)
    for f in npz_files:
        if f.stem not in seen:
            episodes.append(f)
    return natsort.natsorted(episodes, key=lambda p: str(p))


def _load_episode_array(data_path: Path, key: str) -> np.ndarray | None:
    """Load a single array from an episode (NPZ file or .npy directory)."""
    if data_path.is_dir():
        npy_file = data_path / f"{key}.npy"
        if npy_file.exists():
            return np.load(npy_file)
        return None
    else:
        try:
            data = np.load(data_path, allow_pickle=True)
            if key in data.files:
                return data[key]
        except Exception:
            pass
        return None


def _needs_extraction(
    data_path: Path,
    data_dir: Path,
    feature_dir: Path,
    camera_indices: list[int],
    overwrite: bool,
) -> bool:
    """Return True if any requested camera is missing rgb_features."""
    if overwrite:
        return True
    fp = _feat_path(data_path, data_dir, feature_dir)
    if not fp.exists() or not fp.is_dir():
        return True
    for ci in camera_indices:
        if not (fp / f"rgb_features_cam{ci}.npy").exists():
            return True
    return False


def _process_file(
    data_path: Path,
    data_dir: Path,
    feature_dir: Path,
    backbone: torch.nn.Module,
    camera_indices: list[int],
    batch_size: int,
    device: torch.device,
    overwrite: bool,
    encoder: str = "dinov2_vitl14",
    use_fp16: bool = True,
) -> bool:
    """Extract features for one episode. Returns True if features were written."""
    out_dir = _feat_path(data_path, data_dir, feature_dir)

    # Check which cameras already have features
    existing_cams: set[int] = set()
    if out_dir.exists() and not overwrite:
        for ci in camera_indices:
            if (out_dir / f"rgb_features_cam{ci}.npy").exists():
                existing_cams.add(ci)

    wrote = False
    for ci in camera_indices:
        if not overwrite and ci in existing_cams:
            continue

        images = _load_episode_array(data_path, f"images_cam{ci}")
        if images is None:
            continue

        if images.dtype != np.uint8:
            images = images.astype(np.uint8)

        feats = _extract(backbone, images, batch_size, device, encoder, use_fp16=use_fp16)

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"rgb_features_cam{ci}.npy", feats.astype(np.float16))
        wrote = True

    return wrote


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", required=True, type=Path,
                        help="Root directory containing episodes (data*.npz or data*/ dirs, searched recursively).")
    parser.add_argument("--feature_dir", required=True, type=Path,
                        help="Root directory to write feature NPZ files (mirrors data_dir tree).")
    parser.add_argument("--encoder", default="dinov2_vitl14",
                        choices=list(_BACKBONE_DIMS.keys()),
                        help="Backbone to use. dinov2_vitl14 for DiffusionPolicy; "
                             "siglip_so400m for RDT.")
    parser.add_argument("--camera_indices", nargs="+", type=int, default=[0, 1, 2],
                        help="Camera indices to extract features for.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Number of frames per GPU batch.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device index. Use -1 for CPU.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if rgb_features_camX already exists.")
    parser.add_argument("--dry_run", action="store_true",
                        help="List files that would be processed without writing.")
    parser.add_argument("--shard", type=str, default=None,
                        help="Process only a subset of files: SHARD_IDX/NUM_SHARDS "
                             "(e.g. '0/4' = first quarter). Use with multiple GPUs "
                             "to parallelize extraction.")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", default=True,
                        help="Disable fp16 inference (default: fp16 enabled on CUDA).")
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: {args.data_dir} does not exist.")
        sys.exit(1)

    # ---- Collect episodes (NPZ files or .npy directories) ----
    all_episodes = _find_episodes(args.data_dir)
    if not all_episodes:
        print(f"No episodes (data*.npz or data*/) found under {args.data_dir}")
        sys.exit(1)

    # ---- Filter episodes that actually need extraction ----
    to_process = [
        p for p in all_episodes
        if _needs_extraction(p, args.data_dir, args.feature_dir,
                             args.camera_indices, args.overwrite)
    ]

    # ---- Optional sharding ----
    if args.shard is not None:
        shard_idx, num_shards = (int(x) for x in args.shard.split("/"))
        assert 0 <= shard_idx < num_shards, f"Invalid shard: {shard_idx}/{num_shards}"
        to_process = [f for i, f in enumerate(to_process) if i % num_shards == shard_idx]

    raw_dim = _BACKBONE_DIMS[args.encoder]
    print(f"Encoder     : {args.encoder}  (raw_dim={raw_dim})")
    print(f"Cameras     : {args.camera_indices}")
    print(f"Data dir    : {args.data_dir}")
    print(f"Feature dir : {args.feature_dir}")
    shard_str = f"  (shard {args.shard})" if args.shard else ""
    print(f"Episodes    : {len(to_process)} / {len(all_episodes)} need extraction{shard_str}")

    if args.dry_run:
        for p in to_process:
            feat_p = _feat_path(p, args.data_dir, args.feature_dir)
            print(f"  {p}  →  {feat_p}")
        return

    if not to_process:
        print("Nothing to do.")
        return

    # ---- Device ----
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    use_fp16 = args.fp16 and device.type == "cuda"
    print(f"Device      : {device}  (fp16={'on' if use_fp16 else 'off'})\n")

    # ---- Load backbone ----
    backbone = _load_backbone(args.encoder, device)

    # ---- Process files ----
    n_written = 0
    for path in tqdm(to_process, unit="file", desc="Extracting"):
        written = _process_file(
            path, args.data_dir, args.feature_dir,
            backbone, args.camera_indices,
            args.batch_size, device, args.overwrite,
            encoder=args.encoder,
            use_fp16=use_fp16,
        )
        if written:
            n_written += 1

    print(f"\nDone. Wrote {n_written} / {len(to_process)} feature files.")
    if args.encoder in _PATCH_COUNTS:
        n_patches = _PATCH_COUNTS[args.encoder]
        print(f"Feature key format: rgb_features_cam{{i}}  shape (T, {n_patches}, {raw_dim})")
        print(f"  ↑ patch tokens (spatial sequence), suitable for RDT ACI cross-attention")
    else:
        print(f"Feature key format: rgb_features_cam{{i}}  shape (T, {raw_dim})")
    print(f"\nNext step — train with precomputed features:")
    print(f"  Add --feature_dir {args.feature_dir} to your train.py command.")


if __name__ == "__main__":
    main()
