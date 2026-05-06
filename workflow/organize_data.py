"""
Organize raw per-operation data into the easy_mode training layout.

Source layout (data/TexasPokerRobot/)
--------------------------------------
    {operation_name}/
        data_0001.npz
        data_0002.npz
        ...

Target layout (data/easy_mode/)
---------------------------------
    instructions.json           ← task metadata: id, operation, text, one_hot
    {instruction_id}/
        {task_name}_train_{N}/  ← training episodes (.npy directories)
        {task_name}_test/       ← test episodes (.npy directories)

Usage
-----
    # Default paths, 5 test files per task
    python workflow/organize_data.py

    # Custom paths
    python workflow/organize_data.py \\
        --source_dir data/TexasPokerRobot \\
        --target_dir data/easy_mode \\
        --eval_count 5 \\
        --task_name pick_up_card

    # Use symlinks instead of copying (saves disk space)
    python workflow/organize_data.py --symlink
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Instruction definitions
# ---------------------------------------------------------------------------

# Full task list — 14 operations.  Edit text strings here to change the
# natural-language instruction seen by text-conditioned models.
INSTRUCTIONS: list[dict] = [
    {"id":  0, "operation": "pick_up_left",  "text": "Pick up the card on the left"},
    {"id":  1, "operation": "pick_up_right", "text": "Pick up the card on the right"},
    {"id":  2, "operation": "push_5",        "text": "Push chips worth 5 forward"},
    {"id":  3, "operation": "push_10",       "text": "Push chips worth 10 forward"},
    {"id":  4, "operation": "push_50",       "text": "Push chips worth 50 forward"},
    {"id":  5, "operation": "push_100",      "text": "Push chips worth 100 forward"},
    {"id":  6, "operation": "pull_5",        "text": "Pull chips worth 5 back"},
    {"id":  7, "operation": "pull_10",       "text": "Pull chips worth 10 back"},
    {"id":  8, "operation": "pull_50",       "text": "Pull chips worth 50 back"},
    {"id":  9, "operation": "pull_100",      "text": "Pull chips worth 100 back"},
    {"id": 10, "operation": "put_down_left", "text": "Put the card down on the left"},
    {"id": 11, "operation": "put_down_right","text": "Put the card down on the right"},
    {"id": 12, "operation": "show_left",     "text": "Show the card on the left"},
    {"id": 13, "operation": "show_right",    "text": "Show the card on the right"},
]

NUM_INSTRUCTIONS = len(INSTRUCTIONS)


def _build_instructions_json(instructions: list[dict]) -> dict:
    """Build the full instructions.json structure with one-hot vectors."""
    n = len(instructions)
    result = {"num_instructions": n, "instructions": {}}
    for entry in instructions:
        i = entry["id"]
        one_hot = [0] * n
        one_hot[i] = 1
        result["instructions"][str(i)] = {
            "id":        i,
            "operation": entry["operation"],
            "text":      entry["text"],
            "one_hot":   one_hot,
        }
    return result


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _sort_key(path: Path) -> int:
    """Extract the trailing number from data_XXXX.npz for stable sorting."""
    m = re.search(r"(\d+)\.npz$", path.name)
    return int(m.group(1)) if m else 0


def _explode_npz_to_dir(src: Path, dst_dir: Path) -> None:
    """Explode a .npz file into a directory of individual .npy files."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(src, allow_pickle=True)
    for key in data.files:
        arr = data[key]
        np.save(dst_dir / f"{key}.npy", arr)


def _transfer(src: Path, dst: Path, symlink: bool) -> None:
    """Copy or symlink src → dst.

    When not using symlinks, saves as a directory of .npy files
    (``dst`` without ``.npz`` suffix) for fast mmap loading.
    """
    dst_dir = dst.with_suffix("")  # data0001.npz → data0001/
    if dst_dir.exists() or dst.exists():
        return
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        _explode_npz_to_dir(src, dst_dir)


# ---------------------------------------------------------------------------
# Main organizer
# ---------------------------------------------------------------------------

def organize_data(
    source_dir: str = "data/TexasPokerRobot",
    target_dir: str = "data/easy_mode",
    eval_count: int = 5,
    task_name: str = "pick_up_card",
    symlink: bool = False,
    instructions: list[dict] = INSTRUCTIONS,
) -> None:
    """
    Reorganize per-operation NPZ files into the easy_mode training layout
    and write instructions.json with one-hot vectors and text descriptions.

    Images are kept at original resolution (640×480). Each model's encoder
    handles resizing internally (DinoV2 → 224×224, SigLIP → 384×384, etc.).

    Args:
        source_dir:   Root of raw per-operation folders.
        target_dir:   Output root (easy_mode layout).
        eval_count:   Number of files to hold out for testing per task.
        task_name:    Prefix used when naming train/test subdirectories.
        symlink:      Create symlinks instead of copying (saves disk space).
        instructions: List of instruction dicts (id, operation, text).
    """
    source = Path(source_dir)
    target = Path(target_dir)

    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    target.mkdir(parents=True, exist_ok=True)

    # ---- Write instructions.json ----
    instr_json = _build_instructions_json(instructions)
    instr_path = target / "instructions.json"
    with open(instr_path, "w", encoding="utf-8") as f:
        json.dump(instr_json, f, indent=2, ensure_ascii=False)
    print(f"Wrote {instr_path}")

    # ---- Process each operation ----
    processed = 0
    skipped = []

    for entry in instructions:
        instr_id   = entry["id"]
        operation  = entry["operation"]
        text       = entry["text"]

        op_dir = source / operation
        if not op_dir.exists():
            skipped.append(operation)
            continue

        npz_files = sorted(op_dir.glob("*.npz"), key=_sort_key)
        if not npz_files:
            skipped.append(operation)
            continue

        total = len(npz_files)
        if total <= eval_count:
            print(f"  WARNING {operation}: only {total} files (need >{eval_count}), skipping")
            skipped.append(operation)
            continue

        n_train = total - eval_count
        train_files = npz_files[:n_train]
        test_files  = npz_files[n_train:]

        # Create output directories.
        instr_dir  = target / str(instr_id)
        train_dir  = instr_dir / f"{task_name}_train_{n_train}"
        test_dir   = instr_dir / f"{task_name}_test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Transfer files, renaming to data0001.npz, data0002.npz, …
        for idx, src in enumerate(train_files, start=1):
            _transfer(src, train_dir / f"data{idx:04d}.npz", symlink)
        for idx, src in enumerate(test_files, start=1):
            _transfer(src, test_dir / f"data{idx:04d}.npz", symlink)

        print(f"  [{instr_id:2d}] {operation:<20s}  train={n_train}  test={eval_count}"
              f"  text='{text}'")
        processed += 1

    # ---- Summary ----
    print()
    print(f"Done. {processed} tasks organized under {target}/  [original resolution]")
    if skipped:
        print(f"Skipped (not found locally): {', '.join(skipped)}")
    print()
    print("instructions.json maps each task to:")
    print("  id, operation, text instruction, one-hot vector")
    print(f"  → {instr_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Organize raw per-operation NPZ files into easy_mode layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source_dir", default="data/TexasPokerRobot",
                   help="Root directory containing per-operation subfolders.")
    p.add_argument("--target_dir", default="data/easy_mode",
                   help="Output root directory (easy_mode layout).")
    p.add_argument("--eval_count", type=int, default=5,
                   help="Number of files held out for testing per task.")
    p.add_argument("--task_name", default="pick_up_card",
                   help="Prefix for train/test directory names.")
    p.add_argument("--symlink", action="store_true",
                   help="Symlink files instead of copying.")
    args = p.parse_args()

    organize_data(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        eval_count=args.eval_count,
        task_name=args.task_name,
        symlink=args.symlink,
    )


if __name__ == "__main__":
    main()
