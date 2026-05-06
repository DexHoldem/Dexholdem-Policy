"""
Split a flat folder of NPZ trajectory files into train and test sets.

Usage
-----
    # 80/20 split (default)
    python workflow/prepare_data.py --data_dir /path/to/raw --output_dir /path/to/split

    # Keep exactly 40 files for training, rest for test
    python workflow/prepare_data.py --data_dir /path/to/raw --output_dir /path/to/split --num_train 40

    # 90/10 split, custom names
    python workflow/prepare_data.py \
        --data_dir /path/to/raw \
        --output_dir /path/to/split \
        --train_ratio 0.9 \
        --name pick_up_card

Output layout
-------------
    output_dir/
        {name}_train_{N}/   ← training NPZ files
        {name}_test/        ← test NPZ files
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def prepare_data(
    data_dir: str,
    output_dir: str,
    name: str | None = None,
    num_train: int | None = None,
    train_ratio: float = 0.8,
    symlink: bool = False,
) -> tuple[Path, Path]:
    """
    Split NPZ files in *data_dir* into train and test sets.

    Priority: num_train > train_ratio.

    Args:
        data_dir:    Directory containing raw NPZ files.
        output_dir:  Where to write train/ and test/ directories.
        name:        Dataset name prefix (default: name of data_dir).
        num_train:   Exact number of files to use for training.
        train_ratio: Fraction for training when num_train is not given.
        symlink:     Create symlinks instead of copying (saves disk space).

    Returns:
        (train_dir, test_dir) as Path objects.
    """
    data_path = Path(data_dir).resolve()
    out_path = Path(output_dir).resolve()

    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    npz_files = sorted(data_path.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_path}")

    if name is None:
        name = data_path.name

    total = len(npz_files)
    if num_train is not None:
        if num_train >= total:
            raise ValueError(
                f"--num_train ({num_train}) must be less than total files ({total})"
            )
        n_train = num_train
    else:
        n_train = max(1, int(total * train_ratio))
        if n_train >= total:
            n_train = total - 1  # always keep at least 1 for test

    n_test = total - n_train

    train_dir = out_path / f"{name}_train_{n_train}"
    test_dir = out_path / f"{name}_test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_files = npz_files[:n_train]
    test_files = npz_files[n_train:]

    def _copy_or_link(src: Path, dst_dir: Path) -> None:
        dst = dst_dir / src.name
        if dst.exists():
            return
        if symlink:
            dst.symlink_to(src)
        else:
            shutil.copy2(src, dst)

    for f in train_files:
        _copy_or_link(f, train_dir)
    for f in test_files:
        _copy_or_link(f, test_dir)

    print(f"Total files : {total}")
    print(f"Train ({n_train}): {train_dir}")
    print(f"Test  ({n_test}): {test_dir}")
    return train_dir, test_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description="Split a flat folder of NPZ files into train and test sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", required=True,
                   help="Directory containing raw .npz trajectory files.")
    p.add_argument("--output_dir", required=True,
                   help="Where to write the train/ and test/ directories.")
    p.add_argument("--name", default=None,
                   help="Dataset name prefix (default: name of --data_dir).")
    p.add_argument("--num_train", type=int, default=None,
                   help="Exact number of files for training. Overrides --train_ratio.")
    p.add_argument("--train_ratio", type=float, default=0.8,
                   help="Fraction of files used for training (0–1).")
    p.add_argument("--symlink", action="store_true",
                   help="Create symlinks instead of copying files (saves disk space).")
    args = p.parse_args()

    prepare_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        name=args.name,
        num_train=args.num_train,
        train_ratio=args.train_ratio,
        symlink=args.symlink,
    )


if __name__ == "__main__":
    main()
