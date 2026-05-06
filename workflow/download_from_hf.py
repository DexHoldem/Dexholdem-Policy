#!/usr/bin/env python3
"""
Download trained checkpoints from HuggingFace Hub.

Usage:
    # Download one model
    python workflow/download_from_hf.py --model dp --save_dir checkpoints/dp

    # Download all models
    python workflow/download_from_hf.py --all_models --save_dir checkpoints

    # Download all checkpoints (not just latest)
    python workflow/download_from_hf.py --model rdt --save_dir checkpoints/rdt --all

First-time setup:
    pip install huggingface_hub
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


AVAILABLE_MODELS = [
    "dp",
    "dp_unet",
    "dp_transformer_resnet",
    "act",
    "baku",
    "rdt",
    "rdt_small",
    "rdt_ft",
]
DEFAULT_REPO_ID = "Winniechen2002/DexasPolicy"


def download_model(
    api: HfApi,
    repo_id: str,
    model_name: str,
    save_dir: Path,
    download_all: bool = False,
):
    """Download checkpoint files for a model."""
    # List files in the model subfolder
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    model_files = [f for f in all_files if f.startswith(f"{model_name}/") and f.endswith(".pt")]

    if not model_files:
        print(f"  [SKIP] No files found for {model_name}")
        return

    if not download_all:
        # Prefer latest.pt, fall back to highest epoch
        latest = [f for f in model_files if f.endswith("latest.pt")]
        if latest:
            model_files = latest
        else:
            model_files = [sorted(model_files)[-1]]

    save_dir.mkdir(parents=True, exist_ok=True)

    for hf_path in model_files:
        filename = Path(hf_path).name
        print(f"  Downloading {hf_path} → {save_dir / filename}")
        hf_hub_download(
            repo_id=repo_id,
            filename=hf_path,
            local_dir=save_dir.parent,
            repo_type="model",
        )
        print(f"  ✓ Done: {save_dir / filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Download checkpoints from HuggingFace Hub"
    )
    parser.add_argument(
        "--model", type=str, choices=AVAILABLE_MODELS,
        help="Model to download"
    )
    parser.add_argument(
        "--all_models", action="store_true",
        help="Download all available models"
    )
    parser.add_argument(
        "--save_dir", type=str, default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all checkpoints, not just latest.pt"
    )
    parser.add_argument(
        "--repo_id", type=str, default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO_ID})"
    )
    args = parser.parse_args()

    if not args.all_models and not args.model:
        parser.error("Either --all_models or --model is required")

    api = HfApi()

    print(f"Downloading from: https://huggingface.co/{args.repo_id}")
    print()

    if args.all_models:
        for model_name in AVAILABLE_MODELS:
            save_dir = Path(args.save_dir) / model_name
            print(f"[{model_name}]")
            download_model(api, args.repo_id, model_name, save_dir, args.all)
            print()
    else:
        save_dir = Path(args.save_dir)
        print(f"[{args.model}]")
        download_model(api, args.repo_id, args.model, save_dir, args.all)

    print("All downloads complete!")


if __name__ == "__main__":
    main()
