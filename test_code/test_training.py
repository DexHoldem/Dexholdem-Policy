"""
End-to-end training smoke test.

Generates a tiny synthetic dataset, then runs train.py for a few epochs
to verify the full pipeline works: data loading → encoder → model → loss →
checkpoint save. Does NOT test convergence — just checks nothing crashes.

Usage
-----
# Run all scenarios (takes ~1–2 min on CPU)
python test_code/test_training.py

# Run a specific scenario by name
python test_code/test_training.py --run dp_pos_only
python test_code/test_training.py --run dp_resnet_img
python test_code/test_training.py --run act_pos_only
python test_code/test_training.py --run dp_multitask
python test_code/test_training.py --run dp_dinov2_transformer
python test_code/test_training.py --run dp_dinov2_transformer_multitask
python test_code/test_training.py --run baku_pos_only
python test_code/test_training.py --run baku_resnet_multitask
python test_code/test_training.py --run rdt_pos_only
python test_code/test_training.py --run rdt_pos_multitask
python test_code/test_training.py --run rdt_precomputed_multitask

# List available scenarios
python test_code/test_training.py --list
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

# Each scenario is a dict:
#   name        — unique identifier
#   description — one-line summary
#   gen_args    — extra args for generate_test_dataset.py
#   train_args  — extra args for train.py
#   multitask   — whether to build multi-path args for train.py

SCENARIOS: list[dict] = [
    {
        "name": "dp_pos_only",
        "description": "DiffusionPolicy (UNet) + proprioception only, single-task",
        "gen_args":   ["--no_images", "--num_train", "4", "--num_test", "2", "--timesteps", "70"],
        "train_args": [
            "--model", "diffusion_policy",
            "--representation_type", "pos",
            "--diffusion_model_type", "unet",
            "--obs_horizon", "1",
            "--pred_horizon", "16",
            "--action_horizon", "8",
            "--epochs", "3",
            "--batch_size", "4",
            "--num_workers", "1",
            "--no-enable_crop",
        ],
        "multitask": False,
    },
    {
        "name": "dp_resnet_img",
        "description": "DiffusionPolicy (UNet) + ResNet18 RGB + pos, single-task",
        "gen_args":   ["--num_train", "3", "--num_test", "1", "--timesteps", "70",
                       "--num_cams", "1"],
        "train_args": [
            "--model", "diffusion_policy",
            "--representation_type", "img-pos",
            "--camera_indices", "0",
            "--rgb_encoder", "resnet18",
            "--rgb_per_cam_output", "32",
            "--pos_output_size", "32",
            "--diffusion_model_type", "unet",
            "--obs_horizon", "1",
            "--pred_horizon", "16",
            "--action_horizon", "8",
            "--epochs", "3",
            "--batch_size", "2",
            "--num_workers", "1",
        ],
        "multitask": False,
    },
    {
        "name": "act_pos_only",
        "description": "ACT + proprioception only, single-task",
        "gen_args":   ["--no_images", "--num_train", "4", "--num_test", "2", "--timesteps", "70"],
        "train_args": [
            "--model", "act",
            "--representation_type", "pos",
            "--obs_horizon", "1",
            "--pred_horizon", "16",
            "--action_horizon", "8",
            "--act_hidden_dim", "64",
            "--act_num_heads", "2",
            "--act_latent_dim", "8",
            "--epochs", "3",
            "--batch_size", "4",
            "--num_workers", "1",
            "--no-enable_crop",
        ],
        "multitask": False,
    },
    {
        "name": "baku_pos_only",
        "description": "BAKU + proprioception only, single-task",
        "gen_args":   ["--no_images", "--num_train", "4", "--num_test", "2", "--timesteps", "70"],
        "train_args": [
            "--model", "baku",
            "--representation_type", "pos",
            "--obs_horizon", "2",
            "--pred_horizon", "16",
            "--action_horizon", "8",
            "--baku_hidden_size", "64",
            "--baku_depth", "2",
            "--baku_num_heads", "2",
            "--baku_ff_dim", "128",
            "--baku_dropout", "0.0",
            "--epochs", "3",
            "--batch_size", "4",
            "--num_workers", "1",
            "--no-enable_crop",
        ],
        "multitask": False,
    },
    {
        "name": "baku_resnet_multitask",
        "description": "BAKU + fused ResNet18 RGBD + pos + instruction, multi-task (2 tasks, default FiLM)",
        "gen_args":   ["--num_train", "3", "--num_test", "1", "--timesteps", "70",
                       "--num_cams", "1", "--multitask"],
        "train_args": [
            "--model", "baku",
            "--representation_type", "img-depth-pos",
            "--camera_indices", "0",
            "--rgb_encoder", "resnet18",
            "--depth_encoder", "resnet18",
            "--fuse_rgbd",
            "--rgb_per_cam_output", "32",
            "--depth_per_cam_output", "16",
            "--pos_output_size", "32",
            "--use_instruction",
            "--num_instructions", "14",
            "--instruction_embed_dim", "16",
            "--obs_horizon", "2",
            "--pred_horizon", "16",
            "--action_horizon", "8",
            "--baku_hidden_size", "64",
            "--baku_depth", "2",
            "--baku_num_heads", "2",
            "--baku_ff_dim", "128",
            "--baku_dropout", "0.0",
            "--epochs", "3",
            "--batch_size", "2",
            "--num_workers", "1",
        ],
        "multitask": True,
        "multitask_ids": [0, 1],
    },
    {
        "name": "dp_multitask",
        "description": "DiffusionPolicy (UNet) + pos + instruction one-hot, multi-task (2 tasks)",
        "gen_args":   ["--no_images", "--num_train", "3", "--num_test", "1",
                       "--timesteps", "70", "--multitask"],
        "train_args": [
            "--model", "diffusion_policy",
            "--representation_type", "pos",
            "--diffusion_model_type", "unet",
            "--use_instruction",
            "--num_instructions", "14",
            "--obs_horizon", "1",
            "--pred_horizon", "16",
            "--action_horizon", "8",
            "--epochs", "3",
            "--batch_size", "4",
            "--num_workers", "1",
            "--no-enable_crop",
        ],
        "multitask": True,
        "multitask_ids": [0, 1],   # subset to keep the test fast
    },
    {
        "name": "dp_dinov2_transformer",
        "description": "DiffusionPolicy (Transformer) + DinoV2 precomputed features + pos, single-task",
        "gen_args":   [
            "--num_train", "3", "--num_test", "1", "--timesteps", "70",
            "--num_cams", "1",
            "--precompute_features", "--encoder", "dinov2_vitl14",
        ],
        "train_args": [
            "--model", "diffusion_policy",
            "--representation_type", "img-pos",
            "--camera_indices", "0",
            "--rgb_encoder", "dinov2_vitl14",
            "--freeze_rgb_encoder",
            "--precompute_rgb_features",
            "--rgb_per_cam_output", "32",
            "--pos_output_size", "32",
            "--diffusion_model_type", "transformer",
            "--obs_horizon", "1",
            "--pred_horizon", "16",
            "--action_horizon", "8",
            "--epochs", "3",
            "--batch_size", "2",
            "--num_workers", "1",
        ],
        "multitask": False,
        "precompute_encoder": "dinov2_vitl14",   # signals _build_train_cmd to add feature dirs
    },
    {
        "name": "dp_dinov2_transformer_multitask",
        "description": "DiffusionPolicy (Transformer) + DinoV2 precomputed + pos + instruction, multi-task (2 tasks)",
        "gen_args":   [
            "--num_train", "3", "--num_test", "1", "--timesteps", "70",
            "--num_cams", "1",
            "--precompute_features", "--encoder", "dinov2_vitl14",
            "--multitask",
        ],
        "train_args": [
            "--model", "diffusion_policy",
            "--representation_type", "img-pos",
            "--camera_indices", "0",
            "--rgb_encoder", "dinov2_vitl14",
            "--freeze_rgb_encoder",
            "--precompute_rgb_features",
            "--rgb_per_cam_output", "32",
            "--pos_output_size", "32",
            "--diffusion_model_type", "transformer",
            "--use_instruction",
            "--num_instructions", "14",
            "--obs_horizon", "1",
            "--pred_horizon", "16",
            "--action_horizon", "8",
            "--epochs", "3",
            "--batch_size", "2",
            "--num_workers", "1",
        ],
        "multitask": True,
        "multitask_ids": [0, 1],
        "precompute_encoder": "dinov2_vitl14",
    },
    # -----------------------------------------------------------------------
    # RDT scenarios
    # -----------------------------------------------------------------------
    {
        "name": "rdt_pos_only",
        "description": "RDT + proprioception only, single-task, tiny model (no T5 download)",
        "gen_args":   ["--no_images", "--num_train", "4", "--num_test", "2", "--timesteps", "70"],
        "train_args": [
            "--model", "rdt",
            "--representation_type", "pos",
            "--obs_horizon", "1",
            "--pred_horizon", "16",
            "--action_horizon", "8",
            "--rdt_hidden_size", "64",
            "--rdt_depth", "2",
            "--rdt_num_heads", "2",
            "--rdt_ff_dim", "128",
            "--rdt_prediction_type", "sample",
            "--rdt_token_max_len", "8",
            "--rdt_num_diffusion_iters", "5",
            "--rdt_num_inference_iters", "3",
            "--rdt_inference_scheduler", "ddpm",
            "--rdt_cond_mask_prob", "0.0",
            "--instructions_file", "",
            "--epochs", "3",
            "--batch_size", "4",
            "--num_workers", "1",
            "--no-enable_crop",
        ],
        "multitask": False,
    },
    {
        "name": "rdt_pos_multitask",
        "description": "RDT + pos + T5 instruction conditioning, multi-task (2 tasks, zero-init T5 buffers)",
        "gen_args":   ["--no_images", "--num_train", "3", "--num_test", "1",
                       "--timesteps", "70", "--multitask"],
        "train_args": [
            "--model", "rdt",
            "--representation_type", "pos",
            "--obs_horizon", "1",
            "--pred_horizon", "16",
            "--action_horizon", "8",
            "--num_instructions", "14",
            "--rdt_hidden_size", "64",
            "--rdt_depth", "2",
            "--rdt_num_heads", "2",
            "--rdt_ff_dim", "128",
            "--rdt_prediction_type", "sample",
            "--rdt_token_max_len", "8",
            "--rdt_num_diffusion_iters", "5",
            "--rdt_num_inference_iters", "3",
            "--rdt_inference_scheduler", "ddpm",
            "--rdt_cond_mask_prob", "0.0",
            "--epochs", "3",
            "--batch_size", "4",
            "--num_workers", "1",
            "--no-enable_crop",
        ],
        "multitask": True,
        "multitask_ids": [0, 1],
    },
    {
        "name": "rdt_precomputed_multitask",
        "description": "RDT + SigLIP-SO400M patch-token features + T5, multi-task (2 tasks)",
        # --n_patches 4: generate tiny (T, 4, 1152) patch arrays instead of full 729
        # so the test runs fast on CPU without a GPU or real backbone.
        "gen_args":   [
            "--num_train", "3", "--num_test", "1", "--timesteps", "70",
            "--num_cams", "1",
            "--precompute_features", "--encoder", "siglip_so400m", "--n_patches", "4",
            "--multitask",
        ],
        "train_args": [
            "--model", "rdt",
            "--representation_type", "img-pos",
            "--camera_indices", "0",
            "--rgb_encoder", "siglip_so400m",
            "--freeze_rgb_encoder",
            "--precompute_rgb_features",
            "--rgb_per_cam_output", "32",
            "--pos_output_size", "32",
            "--obs_horizon", "1",
            "--pred_horizon", "16",
            "--action_horizon", "8",
            "--num_instructions", "14",
            "--rdt_hidden_size", "64",
            "--rdt_depth", "2",
            "--rdt_num_heads", "2",
            "--rdt_ff_dim", "128",
            "--rdt_prediction_type", "sample",
            "--rdt_token_max_len", "8",
            "--rdt_num_diffusion_iters", "5",
            "--rdt_num_inference_iters", "3",
            "--rdt_inference_scheduler", "ddpm",
            "--rdt_cond_mask_prob", "0.0",
            "--rdt_prop_dim", "30",
            "--epochs", "3",
            "--batch_size", "2",
            "--num_workers", "1",
        ],
        "multitask": True,
        "multitask_ids": [0, 1],
        "precompute_encoder": "siglip_so400m",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATE_SCRIPT = REPO_ROOT / "test_code" / "generate_test_dataset.py"
TRAIN_SCRIPT    = REPO_ROOT / "train.py"


def _run(cmd: list[str], cwd: Path, label: str) -> tuple[bool, str]:
    """Run a subprocess, return (success, combined_output)."""
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    if result.returncode != 0:
        return False, combined
    return True, combined


def _build_train_cmd(
    scenario: dict,
    data_dir: Path,
    ckpt_dir: Path,
    feature_root: Path | None = None,
) -> list[str]:
    """Assemble the train.py command for a scenario."""
    cmd = [sys.executable, str(TRAIN_SCRIPT)]

    task = "pick_up_card"
    n_train = next(
        int(b) for a, b in zip(scenario["gen_args"], scenario["gen_args"][1:])
        if a == "--num_train"
    )
    n_test = next(
        int(b) for a, b in zip(scenario["gen_args"], scenario["gen_args"][1:])
        if a == "--num_test"
    )

    if scenario["multitask"]:
        ids = scenario.get("multitask_ids", [0, 1])
        train_paths = ";".join(
            str(data_dir / str(i) / f"{task}_train_{n_train}") for i in ids
        )
        val_paths = ";".join(
            str(data_dir / str(i) / f"{task}_test_{n_test}") for i in ids
        )
        cmd += ["--multitask", "--train_paths", train_paths, "--val_paths", val_paths]

        if feature_root is not None:
            feat_train = ";".join(
                str(feature_root / str(i) / f"{task}_train_{n_train}") for i in ids
            )
            feat_val = ";".join(
                str(feature_root / str(i) / f"{task}_test_{n_test}") for i in ids
            )
            cmd += ["--feature_dirs", feat_train, "--val_feature_dirs", feat_val]
    else:
        cmd += [
            "--train_path", str(data_dir / "0" / f"{task}_train_{n_train}"),
            "--val_path",   str(data_dir / "0" / f"{task}_test_{n_test}"),
        ]

        if feature_root is not None:
            cmd += [
                "--feature_dir",     str(feature_root / "0" / f"{task}_train_{n_train}"),
                "--val_feature_dir", str(feature_root / "0" / f"{task}_test_{n_test}"),
            ]

    cmd += ["--save_path", str(ckpt_dir)]
    cmd += scenario["train_args"]
    return cmd


def _run_scenario(scenario: dict, verbose: bool) -> tuple[bool, str, float]:
    """
    Run one scenario in a temp directory.
    Returns (passed, message, elapsed_seconds).
    """
    with tempfile.TemporaryDirectory(prefix="dp_test_") as tmp:
        tmp = Path(tmp)
        data_dir = tmp / "data"
        ckpt_dir = tmp / "ckpt"

        # For precompute scenarios, feature files go to a sibling directory.
        feature_root: Path | None = None
        extra_gen_args: list[str] = []
        if scenario.get("precompute_encoder"):
            feature_root = tmp / "features"
            extra_gen_args = ["--feature_output_dir", str(feature_root)]

        # --- generate dataset ---
        gen_cmd = [
            sys.executable, str(GENERATE_SCRIPT),
            "--output_dir", str(data_dir),
            "--instruction", "0",
            *scenario["gen_args"],
            *extra_gen_args,
        ]
        ok, out = _run(gen_cmd, cwd=REPO_ROOT, label="generate")
        if not ok:
            return False, f"Dataset generation failed:\n{out}", 0.0

        # --- train ---
        train_cmd = _build_train_cmd(scenario, data_dir, ckpt_dir, feature_root=feature_root)
        t0 = time.time()
        ok, out = _run(train_cmd, cwd=REPO_ROOT, label="train")
        elapsed = time.time() - t0

        if not ok:
            # Missing optional dependency → skip instead of fail.
            if "ModuleNotFoundError" in out or "No module named" in out:
                import re
                mod = re.search(r"No module named '([^']+)'", out)
                missing = mod.group(1) if mod else "unknown"
                return None, f"SKIP — missing dependency: {missing}", elapsed
            detail = out[-3000:] if len(out) > 3000 else out
            return False, f"Training failed:\n{detail}", elapsed

        # --- verify checkpoint was written ---
        ckpts = list(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
        if not ckpts:
            return False, f"Training exited 0 but no .pt checkpoint found in {ckpt_dir}", elapsed

        return True, f"checkpoint: {ckpts[0].name}", elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", metavar="NAME",
                        help="Run only the named scenario.")
    parser.add_argument("--list", action="store_true",
                        help="List scenario names and exit.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full subprocess output on failure.")
    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for s in SCENARIOS:
            print(f"  {s['name']:<22}  {s['description']}")
        return

    scenarios = SCENARIOS
    if args.run:
        scenarios = [s for s in SCENARIOS if s["name"] == args.run]
        if not scenarios:
            print(f"ERROR: unknown scenario '{args.run}'. Use --list to see options.")
            sys.exit(1)

    print(f"Running {len(scenarios)} scenario(s)...\n")
    results = []
    for s in scenarios:
        print(f"[{s['name']}] {s['description']}")
        passed, msg, elapsed = _run_scenario(s, verbose=args.verbose)
        if passed is None:
            status = "SKIP"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {status}  {elapsed:.1f}s  —  {msg}\n")
        results.append((s["name"], passed, msg))

    n_pass = sum(1 for _, p, _ in results if p is True)
    n_skip = sum(1 for _, p, _ in results if p is None)
    n_fail = sum(1 for _, p, _ in results if p is False)
    print(f"{'='*50}")
    print(f"Results: {n_pass} passed, {n_skip} skipped, {n_fail} failed")

    if n_fail:
        print("\nFailed scenarios:")
        for name, passed, msg in results:
            if passed is False:
                print(f"  {name}")
                print(textwrap.indent(msg, "    "))
        sys.exit(1)


if __name__ == "__main__":
    main()
