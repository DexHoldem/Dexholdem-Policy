#!/usr/bin/env python3
"""Convert NPZ episode files to .npy directory format for fast mmap loading.

Each data0001.npz becomes a data0001/ directory containing individual .npy
files. The original .npz is removed after successful conversion.

Optionally resize images to a target resolution during conversion, so
different models can use pre-resized data without on-the-fly interpolation:

    data/easy_mode_224/    ← DinoV2  (224×224)
    data/easy_mode_384/    ← SigLIP  (384×384)
    data/easy_mode_240x320/ ← ResNet18 (240×320)

Usage:
    # Basic conversion (keep original resolution)
    python workflow/npz_to_npy.py --data_dir data/easy_mode

    # Resize images to 224×224 for DinoV2
    python workflow/npz_to_npy.py --data_dir data/easy_mode --output_dir data/easy_mode_224 --image_size 224x224

    # Resize to 384×384 for SigLIP
    python workflow/npz_to_npy.py --data_dir data/easy_mode --output_dir data/easy_mode_384 --image_size 384x384

    # Resize to 240×320 for ResNet18
    python workflow/npz_to_npy.py --data_dir data/easy_mode --output_dir data/easy_mode_240x320 --image_size 240x320
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


def _resize_images(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a (T, H, W, C) or (T, H, W) image array to (T, target_h, target_w, ...)."""
    T = arr.shape[0]
    src_h, src_w = arr.shape[1], arr.shape[2]
    if src_h == target_h and src_w == target_w:
        return arr

    is_rgb = arr.ndim == 4  # (T, H, W, C)
    frames = []
    for t in range(T):
        if is_rgb:
            frame = cv2.resize(arr[t], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame = cv2.resize(arr[t].astype(np.float32), (target_w, target_h),
                               interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    return np.stack(frames)


def convert_file(
    npz_path: Path,
    output_dir: Path | None = None,
    image_size: tuple[int, int] | None = None,
    remove_npz: bool = True,
) -> bool:
    """Convert one .npz file to a directory of .npy files.

    Args:
        npz_path: Path to the .npz file.
        output_dir: If set, write .npy directory here instead of next to the .npz.
        image_size: (H, W) to resize image arrays to. None = keep original.
        remove_npz: Remove original .npz after conversion.

    Returns True if converted, False if already exists.
    """
    if output_dir is not None:
        # Mirror the relative path structure
        out_dir = output_dir / npz_path.with_suffix("").name
    else:
        out_dir = npz_path.with_suffix("")

    if out_dir.is_dir():
        return False  # already converted

    data = np.load(str(npz_path), allow_pickle=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for key in data.files:
            arr = data[key]
            # Resize image arrays if requested
            if image_size is not None and _is_image_key(key) and arr.ndim >= 3:
                target_h, target_w = image_size
                arr = _resize_images(arr, target_h, target_w)
            np.save(out_dir / f"{key}.npy", arr)
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise

    if remove_npz and output_dir is None:
        npz_path.unlink()
    return True


def _is_image_key(key: str) -> bool:
    """Check if a key corresponds to an image array."""
    return key.startswith("images_cam") or key.startswith("depth_cam")


def _convert_npy_dir(
    src_dir: Path,
    out_dir: Path,
    image_size: tuple[int, int],
) -> bool:
    """Re-save an existing .npy directory with resized images.

    Non-image arrays are copied as-is.
    Returns True if processed, False if output already exists.
    """
    if out_dir.is_dir():
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for npy_file in sorted(src_dir.glob("*.npy")):
            key = npy_file.stem
            try:
                arr = np.load(str(npy_file), mmap_mode="r")
                arr = np.array(arr)
            except ValueError:
                arr = np.load(str(npy_file), allow_pickle=True)
            if _is_image_key(key) and arr.ndim >= 3:
                target_h, target_w = image_size
                arr = _resize_images(arr, target_h, target_w)
            np.save(out_dir / npy_file.name, arr)
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return True


def _parse_image_size(s: str) -> tuple[int, int]:
    """Parse 'HxW' or 'H' (square) string to (H, W) tuple."""
    parts = s.lower().split("x")
    if len(parts) == 1:
        h = int(parts[0])
        return (h, h)
    elif len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    else:
        raise argparse.ArgumentTypeError(f"Invalid image size: {s!r}. Use HxW or H (square).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NPZ files to .npy directories, optionally resizing images."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory to scan recursively for data*.npz or data*/ dirs")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output root (default: convert in-place). Required when --image_size is set.")
    parser.add_argument("--image_size", type=str, default=None,
                        help="Target image size as HxW (e.g. 224x224, 384x384, 240x320). "
                             "Square shorthand: 224 means 224x224.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--keep_npz", action="store_true",
                        help="Keep original .npz files after conversion")
    args = parser.parse_args()

    image_size = _parse_image_size(args.image_size) if args.image_size else None

    if image_size is not None and args.output_dir is None:
        parser.error("--output_dir is required when --image_size is set "
                     "(to avoid overwriting original data)")

    root = Path(args.data_dir)
    output_root = Path(args.output_dir) if args.output_dir else None

    # Find both .npz files and existing .npy directories
    npz_files = sorted(root.rglob("data*.npz"))
    npy_dirs = sorted(
        p for p in root.rglob("data*")
        if p.is_dir() and not p.suffix and p.name.startswith("data")
    )

    if not npz_files and not npy_dirs:
        print(f"No data*.npz files or data*/ directories found in {root}")
        return

    import concurrent.futures
    from functools import partial

    from tqdm import tqdm

    converted = skipped = failed = 0

    # Process .npz files
    if npz_files:
        print(f"Found {len(npz_files)} NPZ files in {root}")

        def _convert_npz(f: Path) -> bool:
            if output_root is not None:
                rel = f.parent.relative_to(root)
                out = output_root / rel
                out.mkdir(parents=True, exist_ok=True)
                return convert_file(f, output_dir=out, image_size=image_size,
                                    remove_npz=not args.keep_npz and output_root is None)
            else:
                return convert_file(f, remove_npz=not args.keep_npz)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_convert_npz, f): f for f in npz_files}
            for fut in tqdm(concurrent.futures.as_completed(futures),
                            total=len(futures), unit="file", desc="Converting NPZ"):
                path = futures[fut]
                try:
                    if fut.result():
                        converted += 1
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"\n[ERROR] {path}: {e}")
                    failed += 1

    # Process existing .npy directories (only when resizing to a new output)
    if npy_dirs and output_root is not None and image_size is not None:
        # Filter out dirs that were already converted from .npz above
        npz_stems = {f.with_suffix("").name for f in npz_files}
        npy_only = [d for d in npy_dirs if d.name not in npz_stems]

        if npy_only:
            print(f"Found {len(npy_only)} .npy directories to resize in {root}")

            def _convert_npy(d: Path) -> bool:
                rel = d.parent.relative_to(root)
                out = output_root / rel / d.name
                return _convert_npy_dir(d, out, image_size)

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_convert_npy, d): d for d in npy_only}
                for fut in tqdm(concurrent.futures.as_completed(futures),
                                total=len(futures), unit="dir", desc="Resizing .npy dirs"):
                    path = futures[fut]
                    try:
                        if fut.result():
                            converted += 1
                        else:
                            skipped += 1
                    except Exception as e:
                        print(f"\n[ERROR] {path}: {e}")
                        failed += 1

    size_str = f" → {image_size[0]}×{image_size[1]}" if image_size else ""
    print(f"\nDone{size_str}: {converted} converted, {skipped} already exist, {failed} failed")


if __name__ == "__main__":
    main()
