#!/usr/bin/env python3
"""Save the first RGB frame from each camera in one trajectory NPZ."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    if frame.size and np.nanmax(frame) <= 1.0:
        return np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return np.clip(frame, 0, 255).astype(np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"failed to save image: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "npz_path",
        nargs="?",
        default="data/pick_up_left/data_0001.npz",
        help="trajectory NPZ to visualize",
    )
    parser.add_argument(
        "--save_dir",
        default="visualized_images",
        help="directory for output PNG files",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    save_dir = Path(args.save_dir)
    if not npz_path.exists():
        raise FileNotFoundError(f"trajectory file does not exist: {npz_path}")

    save_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    frames = []
    for cam_idx in range(3):
        cam_key = f"images_cam{cam_idx}"
        if cam_key not in data:
            print(f"missing {cam_key}")
            continue

        images = data[cam_key]
        print(f"{cam_key}: {images.shape} {images.dtype}")
        first_frame = to_uint8(images[0])
        save_path = save_dir / f"{npz_path.stem}_cam{cam_idx}_frame0.png"
        save_rgb(save_path, first_frame)
        print(f"saved {save_path}")
        frames.append(first_frame)

    if not frames:
        raise RuntimeError(f"no images_cam* arrays found in {npz_path}")

    h = min(frame.shape[0] for frame in frames)
    resized = [
        cv2.resize(frame, (round(frame.shape[1] * h / frame.shape[0]), h))
        for frame in frames
    ]
    contact_sheet = np.concatenate(resized, axis=1)
    contact_path = save_dir / f"{npz_path.stem}_frame0_all_cams.png"
    save_rgb(contact_path, contact_sheet)
    print(f"saved {contact_path}")


if __name__ == "__main__":
    main()
