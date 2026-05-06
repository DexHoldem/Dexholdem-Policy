#!/usr/bin/env python3
"""Download the public TexasPokerRobot dataset from Hugging Face.

The dataset is large, so this script is intentionally a thin wrapper around
``huggingface_hub.snapshot_download`` with include/exclude pattern support.

Examples
--------
Download the full dataset:

    python workflow/download_data.py --local_dir data/TexasPokerRobot

Download a small subset for plumbing checks:

    python workflow/download_data.py \
        --local_dir data/TexasPokerRobot_subset \
        --include "pick_up_left/**" "pick_up_right/**" "README.md"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "Winniechen2002/TexasPokerRobot"


def _none_if_empty(values: list[str] | None) -> list[str] | None:
    return values if values else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TexasPokerRobot from Hugging Face.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo_id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Dataset revision, branch, or commit hash.",
    )
    parser.add_argument(
        "--local_dir",
        type=Path,
        default=Path("data/TexasPokerRobot"),
        help="Directory where dataset files will be written.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help='Optional allow patterns, e.g. "pick_up_left/**".',
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help='Optional ignore patterns, e.g. "*.mp4".',
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Public downloads do not need one.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Parallel download workers.",
    )
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset : https://huggingface.co/datasets/{args.repo_id}")
    print(f"Target  : {args.local_dir.resolve()}")
    if args.include:
        print(f"Include : {args.include}")
    if args.exclude:
        print(f"Exclude : {args.exclude}")
    print()

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.local_dir),
        allow_patterns=_none_if_empty(args.include),
        ignore_patterns=_none_if_empty(args.exclude),
        token=args.token,
        max_workers=args.max_workers,
    )

    print()
    print(f"Downloaded to: {Path(path).resolve()}")
    print("Next:")
    print("  python workflow/organize_data.py --source_dir data/TexasPokerRobot --target_dir data/easy_mode --eval_count 5")


if __name__ == "__main__":
    main()
