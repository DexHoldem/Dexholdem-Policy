"""
Generate a small synthetic dataset for testing the training pipeline.

Creates NPZ files that match the exact format expected by
data_processing/loading.py, placed in the easy_mode directory structure
so they can be used directly with train.py.

Usage
-----
# Single-task dataset (instruction 0 only, 3 train + 1 test episode)
python test_code/generate_test_dataset.py

# Multi-task dataset (all 14 instructions)
python test_code/generate_test_dataset.py --multitask

# Custom output dir, more episodes, longer trajectories
python test_code/generate_test_dataset.py \
    --output_dir /tmp/my_test_data \
    --num_train 5 \
    --num_test 2 \
    --timesteps 80

# Minimal (no images, pos only) for fast CPU testing
python test_code/generate_test_dataset.py --no_images

Options
-------
--output_dir    Where to write easy_mode/ tree  [default: data/test_easy_mode]
--num_train     Episodes per instruction for train split  [default: 3]
--num_test      Episodes per instruction for test split   [default: 1]
--timesteps     Trajectory length (T) per episode         [default: 60]
--num_cams      Number of camera streams (0–3)            [default: 3]
--img_h         Image height                              [default: 240]
--img_w         Image width                               [default: 320]
--multitask     Generate all 14 instruction IDs           [default: False]
--instruction   Single instruction ID to generate         [default: 0]
--no_images          Skip image fields (proprioception only)
--precompute_features    Write synthetic rgb_features_cam{i} arrays to a separate
                         feature directory (no backbone download needed).
--feature_output_dir     Where to write feature NPZs [default: <output_dir>/../features]
--encoder                Backbone variant for feature dim/shape  [default: dinov2_vitl14]
--n_patches              Override patch token count for patch encoders (e.g. 4 for fast
                         CPU tests with siglip_so400m). 0 = encoder default (729 for
                         siglip_so400m, pooled for dino).
--seed                   Random seed                       [default: 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Instruction table (mirrors workflow/organize_data.py)
# ---------------------------------------------------------------------------

INSTRUCTIONS = [
    {"id": 0,  "operation": "pick_up_left"},
    {"id": 1,  "operation": "pick_up_right"},
    {"id": 2,  "operation": "push_5"},
    {"id": 3,  "operation": "push_10"},
    {"id": 4,  "operation": "push_50"},
    {"id": 5,  "operation": "push_100"},
    {"id": 6,  "operation": "pull_5"},
    {"id": 7,  "operation": "pull_10"},
    {"id": 8,  "operation": "pull_50"},
    {"id": 9,  "operation": "pull_100"},
    {"id": 10, "operation": "put_down_left"},
    {"id": 11, "operation": "put_down_right"},
    {"id": 12, "operation": "show_left"},
    {"id": 13, "operation": "show_right"},
]

NUM_JOINTS   = 30   # 6 arm + 24 hand
NUM_TACTILE  = 60

_BACKBONE_DIMS: dict[str, int] = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitl14_patch": 1024,  # same dim, patch tokens
    "dinov2_vitg14": 1536,
    "siglip_so400m": 1152,
}
# Keep old name for backwards compat
_DINOV2_DIMS = _BACKBONE_DIMS

# Encoders that output patch token sequences instead of a single pooled vector.
# The value is the default N_patches (overridable via --n_patches).
_PATCH_ENCODERS: dict[str, int] = {
    "dinov2_vitl14_patch": 256,  # 16×16 at 224px; use --n_patches 4 for fast tests
    "siglip_so400m": 729,        # 27×27 at 384px; use --n_patches 4 for fast tests
}


def _make_episode(
    rng: np.random.Generator,
    T: int,
    num_cams: int,
    img_h: int,
    img_w: int,
    with_images: bool,
    precompute_features: bool = False,
    feature_dim: int = 1024,
) -> dict[str, np.ndarray]:
    """Generate one synthetic episode as a dict of numpy arrays."""
    ep: dict[str, np.ndarray] = {}

    # Joint state (always required)
    ep["joint_positions"]  = rng.uniform(-1.0, 1.0, (T, NUM_JOINTS)).astype(np.float32)
    ep["joint_efforts"]    = rng.uniform(-0.5, 0.5, (T, NUM_JOINTS)).astype(np.float32)
    ep["joint_velocities"] = rng.uniform(-0.3, 0.3, (T, NUM_JOINTS)).astype(np.float32)

    # Tactile
    ep["tactile_states"] = rng.uniform(0.0, 1.0, (T, NUM_TACTILE)).astype(np.float32)

    # Images (optional for fast CPU tests)
    if with_images:
        for cam in range(num_cams):
            ep[f"images_cam{cam}"] = rng.integers(
                0, 256, (T, img_h, img_w, 3), dtype=np.uint8
            )
            ep[f"depth_cam{cam}"] = rng.uniform(
                0.1, 5.0, (T, img_h, img_w)
            ).astype(np.float32)

    return ep


def _make_feature_episode(
    rng: np.random.Generator,
    T: int,
    num_cams: int,
    feature_dim: int,
    n_patches: int = 0,
) -> dict[str, np.ndarray]:
    """Generate synthetic precomputed feature arrays (no images needed).

    Args:
        n_patches: If > 0, generate patch token tensors (T, n_patches, feature_dim)
                   as used by SigLIP / RDT.  If 0, generate pooled vectors (T, feature_dim)
                   as used by DinoV2 / DiffusionPolicy.
    """
    if n_patches > 0:
        return {
            f"rgb_features_cam{cam}": rng.standard_normal(
                (T, n_patches, feature_dim)
            ).astype(np.float32)
            for cam in range(num_cams)
        }
    return {
        f"rgb_features_cam{cam}": rng.standard_normal((T, feature_dim)).astype(np.float32)
        for cam in range(num_cams)
    }


def _write_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"  wrote {path.name}  ({path.stat().st_size // 1024} KB)")


def generate(
    output_dir: Path,
    instruction_ids: list[int],
    num_train: int,
    num_test: int,
    T: int,
    num_cams: int,
    img_h: int,
    img_w: int,
    with_images: bool,
    precompute_features: bool,
    feature_output_dir: Path | None,
    feature_dim: int,
    n_patches: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    task_name = "pick_up_card"   # matches the real data naming convention

    for instr_id in instruction_ids:
        op = next(x["operation"] for x in INSTRUCTIONS if x["id"] == instr_id)
        print(f"\nInstruction {instr_id} ({op})")

        for split, n_eps in [("train", num_train), ("test", num_test)]:
            split_dir = output_dir / str(instr_id) / f"{task_name}_{split}_{n_eps}"
            feat_dir  = (feature_output_dir / str(instr_id) / f"{task_name}_{split}_{n_eps}"
                         if precompute_features and feature_output_dir else None)
            print(f"  {split_dir.relative_to(output_dir.parent)} — {n_eps} episodes")
            for i in range(1, n_eps + 1):
                ep = _make_episode(rng, T, num_cams, img_h, img_w, with_images)
                _write_npz(split_dir / f"data{i:04d}.npz", ep)
                if feat_dir is not None:
                    feat_ep = _make_feature_episode(rng, T, num_cams, feature_dim, n_patches)
                    _write_npz(feat_dir / f"data{i:04d}.npz", feat_ep)

    print(f"\nDataset written to: {output_dir}")
    print("\nExample train command:")
    first_id = instruction_ids[0]
    task_name_dir = f"{task_name}_train_{num_train}"
    test_name_dir = f"{task_name}_test_{num_test}"
    print(
        f"  python train.py --model diffusion_policy \\\n"
        f"    --train_path {output_dir}/{first_id}/{task_name_dir} \\\n"
        f"    --val_path   {output_dir}/{first_id}/{test_name_dir} \\\n"
        f"    --save_path  checkpoints/test_run \\\n"
        f"    --epochs 2 --batch_size 2"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=Path, default=Path("data/test_easy_mode"))
    parser.add_argument("--num_train",  type=int, default=3)
    parser.add_argument("--num_test",   type=int, default=1)
    parser.add_argument("--timesteps",  type=int, default=60)
    parser.add_argument("--num_cams",   type=int, default=3)
    parser.add_argument("--img_h",      type=int, default=240)
    parser.add_argument("--img_w",      type=int, default=320)
    parser.add_argument("--multitask",  action="store_true",
                        help="Generate all 14 instruction IDs.")
    parser.add_argument("--instruction", type=int, default=0,
                        help="Single instruction ID (ignored if --multitask).")
    parser.add_argument("--no_images",  action="store_true",
                        help="Skip image fields (proprioception only, much faster).")
    parser.add_argument("--precompute_features", action="store_true",
                        help="Write synthetic rgb_features_cam{i} to a separate feature directory.")
    parser.add_argument("--feature_output_dir", type=Path, default=None,
                        help="Where to write feature NPZs (default: <output_dir>/../features).")
    parser.add_argument("--encoder", default="dinov2_vitl14",
                        choices=list(_BACKBONE_DIMS.keys()),
                        help="Backbone variant; sets the feature dim (and shape) for synthetic features.")
    parser.add_argument("--n_patches", type=int, default=0,
                        help="Override number of patch tokens for patch-sequence encoders "
                             "(e.g. --n_patches 4 for fast CPU tests with siglip_so400m). "
                             "0 = use encoder default from _PATCH_ENCODERS, or pooled if not a patch encoder.")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    instruction_ids = (
        [x["id"] for x in INSTRUCTIONS]
        if args.multitask
        else [args.instruction]
    )

    feat_out = None
    if args.precompute_features:
        feat_out = args.feature_output_dir or (args.output_dir.parent / "features")

    # Determine n_patches: explicit override > encoder default > 0 (pooled)
    if args.n_patches > 0:
        n_patches = args.n_patches
    elif args.encoder in _PATCH_ENCODERS:
        n_patches = _PATCH_ENCODERS[args.encoder]
    else:
        n_patches = 0

    generate(
        output_dir=args.output_dir,
        instruction_ids=instruction_ids,
        num_train=args.num_train,
        num_test=args.num_test,
        T=args.timesteps,
        num_cams=args.num_cams,
        img_h=args.img_h,
        img_w=args.img_w,
        with_images=not args.no_images,
        precompute_features=args.precompute_features,
        feature_output_dir=feat_out,
        feature_dim=_BACKBONE_DIMS[args.encoder],
        n_patches=n_patches,
        seed=args.seed,
    )

    if feat_out:
        print(f"Feature files written to: {feat_out}")
        print("Pass to train.py with: --feature_dir <leaf_subdir> --val_feature_dir <leaf_subdir>")


if __name__ == "__main__":
    main()
