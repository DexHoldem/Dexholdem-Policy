#!/usr/bin/env python3
"""
Upload trained checkpoints to HuggingFace Hub.

Uploads checkpoint .pt files from a local directory to a HuggingFace model
repository, organized by model name.

Usage:
    # Upload latest checkpoint for one model
    python workflow/upload_to_hf.py --model dp --ckpt_dir checkpoints/dp_exp1

    # Upload all checkpoints (not just latest)
    python workflow/upload_to_hf.py --model dp --ckpt_dir checkpoints/dp_exp1 --all

    # Upload all models at once
    python workflow/upload_to_hf.py --all_models --ckpt_root checkpoints

    # Custom repo
    python workflow/upload_to_hf.py --model rdt --ckpt_dir checkpoints/rdt \
        --repo_id YourName/YourRepo

First-time setup:
    pip install huggingface_hub
    huggingface-cli login
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, login


# Default mapping from checkpoint directory names to HF subfolder names
DEFAULT_MODEL_MAP = {
    "dp": "dp",
    "dp_unet": "dp_unet",
    "dp_transformer_resnet": "dp_transformer_resnet",
    "dp_light": "dp_light",
    "act": "act",
    "baku": "baku",
    "rdt": "rdt",
    "rdt_small": "rdt_small",
    "rdt_ft": "rdt_ft",
}

DEFAULT_REPO_ID = "Winniechen2002/DexasPolicy"


def upload_checkpoint(
    api: HfApi,
    repo_id: str,
    ckpt_dir: Path,
    model_name: str,
    upload_all: bool = False,
):
    """Upload checkpoint files from ckpt_dir to repo_id/model_name/."""
    if not ckpt_dir.exists():
        print(f"  [SKIP] {ckpt_dir} does not exist")
        return

    if upload_all:
        pt_files = sorted(ckpt_dir.glob("*.pt"))
    else:
        latest = ckpt_dir / "latest.pt"
        if not latest.exists():
            # Fall back to the highest epoch file
            pt_files = sorted(ckpt_dir.glob("epoch_*.pt"))
            if pt_files:
                pt_files = [pt_files[-1]]
            else:
                print(f"  [SKIP] No .pt files found in {ckpt_dir}")
                return
        else:
            pt_files = [latest]

    for pt_file in pt_files:
        size_gb = pt_file.stat().st_size / (1024 ** 3)
        hf_path = f"{model_name}/{pt_file.name}"
        print(f"  Uploading {pt_file.name} ({size_gb:.1f} GB) → {hf_path}")
        api.upload_file(
            path_or_fileobj=str(pt_file),
            path_in_repo=hf_path,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  ✓ Done: {hf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload checkpoints to HuggingFace Hub"
    )
    parser.add_argument(
        "--model", type=str,
        help="Model name (used as subfolder on HF). E.g. dp, act, baku, rdt"
    )
    parser.add_argument(
        "--ckpt_dir", type=str,
        help="Path to checkpoint directory (e.g. checkpoints/dp_exp1)"
    )
    parser.add_argument(
        "--all_models", action="store_true",
        help="Upload all models from --ckpt_root"
    )
    parser.add_argument(
        "--ckpt_root", type=str, default="checkpoints",
        help="Root checkpoint directory (default: checkpoints)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Upload all .pt files, not just latest.pt"
    )
    parser.add_argument(
        "--repo_id", type=str, default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO_ID})"
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace token (or use huggingface-cli login)"
    )
    args = parser.parse_args()

    if not args.all_models and (not args.model or not args.ckpt_dir):
        parser.error("Either --all_models or both --model and --ckpt_dir are required")

    # Authenticate
    if args.token:
        login(token=args.token)

    api = HfApi()

    # Verify repo exists
    try:
        api.repo_info(repo_id=args.repo_id, repo_type="model")
    except Exception as e:
        print(f"Error accessing repo {args.repo_id}: {e}")
        print("Make sure the repo exists and you are logged in (huggingface-cli login)")
        sys.exit(1)

    print(f"Uploading to: https://huggingface.co/{args.repo_id}")
    print()

    if args.all_models:
        ckpt_root = Path(args.ckpt_root)
        if not ckpt_root.exists():
            print(f"Checkpoint root {ckpt_root} does not exist")
            sys.exit(1)

        for dir_name, model_name in DEFAULT_MODEL_MAP.items():
            ckpt_dir = ckpt_root / dir_name
            print(f"[{model_name}] {ckpt_dir}")
            upload_checkpoint(api, args.repo_id, ckpt_dir, model_name, args.all)
            print()
    else:
        ckpt_dir = Path(args.ckpt_dir)
        print(f"[{args.model}] {ckpt_dir}")
        upload_checkpoint(api, args.repo_id, ckpt_dir, args.model, args.all)

    print("All uploads complete!")
    print(f"View at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
