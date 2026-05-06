"""
Benchmark: disk → GPU data loading throughput.

Measures how fast a DataLoader with batch_size=128 can deliver batches
to the GPU, including collation and .cuda() transfer time.

Tests both LazyRobotDataset (images on disk) and RobotDataset (images in RAM)
with various num_workers settings.

Usage:
    python test_code/bench_dataloader.py [--gpu 0] [--num_batches 20]
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from data_processing.dataset import (
    DatasetConfig,
    build_dataset_lazy,
    build_dataset,
    create_dataloader,
)


def bench_dataloader(loader, device, num_batches, warmup=3, label=""):
    """Time num_batches iterations of the dataloader, moving each batch to GPU."""
    it = iter(loader)

    # Warmup — let workers spin up, page caches fill
    for i in range(warmup):
        try:
            batch = next(it)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    v.to(device, non_blocking=True)
            torch.cuda.synchronize(device)
        except StopIteration:
            it = iter(loader)

    # Timed run
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    times = []
    for i in range(num_batches):
        t_batch_start = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        # Move to GPU
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v.to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        t_batch_end = time.perf_counter()
        times.append(t_batch_end - t_batch_start)

    total = time.perf_counter() - t0
    avg = total / num_batches
    median = sorted(times)[len(times) // 2]
    p95 = sorted(times)[int(len(times) * 0.95)]

    print(f"  [{label}]")
    print(f"    {num_batches} batches in {total:.2f}s")
    print(f"    avg {avg*1000:.1f}ms | median {median*1000:.1f}ms | p95 {p95*1000:.1f}ms per batch")

    # Print per-key shapes and sizes from last batch
    print(f"    Batch contents:")
    total_mb = 0
    for k, v in sorted(batch.items()):
        if isinstance(v, torch.Tensor):
            mb = v.nelement() * v.element_size() / 1e6
            total_mb += mb
            print(f"      {k}: {list(v.shape)} {v.dtype} ({mb:.1f} MB)")
    print(f"    Total batch size: {total_mb:.1f} MB")
    print()
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--data_dir", type=str,
                        default="data/easy_mode/0/pick_up_card_train_100")
    parser.add_argument("--feature_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Data: {args.data_dir}")
    print(f"Batch size: 128")
    print(f"Num batches to time: {args.num_batches}")
    print()

    batch_size = 128

    # =========================================================================
    # Test 1: Lazy dataset with precomputed features (typical training config)
    # =========================================================================
    print("=" * 70)
    print("TEST 1: LazyRobotDataset + precomputed features (no raw images)")
    print("=" * 70)

    feat_dir = args.feature_dir or "data/vitl14_features/0/pick_up_card_train_100"
    feat_path = Path(feat_dir)
    if not feat_path.exists():
        print(f"  [SKIP] feature dir not found: {feat_dir}\n")
    else:
        config_feat = DatasetConfig(
            representation_type=["pos"],
            camera_indices=[0, 1, 2],
            obs_horizon=1,
            pred_horizon=64,
            action_horizon=32,
            load_img=False,
            isolate_episodes=False,
            n_load_workers=4,
            feature_dir=feat_path,
        )
        ds_feat = build_dataset_lazy(args.data_dir, config_feat)
        print(f"  Dataset size: {len(ds_feat)} samples\n")

        for nw in [0, 2, 4, 8]:
            loader = create_dataloader(
                ds_feat, batch_size=batch_size, shuffle=True,
                num_workers=nw, pin_memory=True,
            )
            bench_dataloader(loader, device, args.num_batches,
                             label=f"num_workers={nw}")
            del loader

    # =========================================================================
    # Test 2: Lazy dataset with raw images
    # =========================================================================
    print("=" * 70)
    print("TEST 2: LazyRobotDataset + raw images (on-demand disk read)")
    print("=" * 70)

    config_img = DatasetConfig(
        representation_type=["img", "pos"],
        camera_indices=[0, 1, 2],
        obs_horizon=1,
        pred_horizon=64,
        action_horizon=32,
        load_img=True,
        isolate_episodes=False,
        n_load_workers=4,
    )
    ds_img = build_dataset_lazy(args.data_dir, config_img)
    print(f"  Dataset size: {len(ds_img)} samples\n")

    for nw in [0, 2, 4, 8]:
        loader = create_dataloader(
            ds_img, batch_size=batch_size, shuffle=True,
            num_workers=nw, pin_memory=True,
        )
        bench_dataloader(loader, device, args.num_batches,
                         label=f"num_workers={nw}")
        del loader

    # =========================================================================
    # Test 3: Eager dataset (all in RAM) with precomputed features
    # =========================================================================
    print("=" * 70)
    print("TEST 3: RobotDataset (eager, all in RAM) + precomputed features")
    print("=" * 70)

    if not feat_path.exists():
        print(f"  [SKIP] feature dir not found: {feat_dir}\n")
    else:
        config_eager = DatasetConfig(
            representation_type=["pos"],
            camera_indices=[0, 1, 2],
            obs_horizon=1,
            pred_horizon=64,
            action_horizon=32,
            load_img=False,
            isolate_episodes=False,
            n_load_workers=4,
            feature_dir=feat_path,
        )
        ds_eager = build_dataset(args.data_dir, config_eager)
        print(f"  Dataset size: {len(ds_eager)} samples\n")

        for nw in [0, 2, 4, 8]:
            loader = create_dataloader(
                ds_eager, batch_size=batch_size, shuffle=True,
                num_workers=nw, pin_memory=True,
            )
            bench_dataloader(loader, device, args.num_batches,
                             label=f"num_workers={nw}")
            del loader

    # =========================================================================
    # Test 4: Proprioception only (no images, no features)
    # =========================================================================
    print("=" * 70)
    print("TEST 4: LazyRobotDataset proprioception only (no images/features)")
    print("=" * 70)

    config_prop = DatasetConfig(
        representation_type=["pos"],
        camera_indices=[0, 1, 2],
        obs_horizon=1,
        pred_horizon=64,
        action_horizon=32,
        load_img=False,
        isolate_episodes=False,
        n_load_workers=4,
    )
    ds_prop = build_dataset_lazy(args.data_dir, config_prop)
    print(f"  Dataset size: {len(ds_prop)} samples\n")

    for nw in [0, 2, 4]:
        loader = create_dataloader(
            ds_prop, batch_size=batch_size, shuffle=True,
            num_workers=nw, pin_memory=True,
        )
        bench_dataloader(loader, device, args.num_batches,
                         label=f"num_workers={nw}")
        del loader


if __name__ == "__main__":
    main()
