"""
Verify observation consistency between robot_client → deploy_policy path
and the training dataset path.

Takes one timestep from a raw NPZ episode, processes it through both:
  1. Deploy path: simulate robot_client obs dict → _obs_list_to_batch
  2. Training path: dataset loading → normalize → batch

Reports per-field diffs so we can find any processing discrepancy.

Usage:
    python test_code/verify_obs_consistency.py --ckpt checkpoints/dp_light/latest.pt --gpu 0
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import learning  # noqa: F401
from deploy_policy import load_checkpoint, _obs_list_to_batch
from data_processing.loading import _JOINT_ORDER, load_episode
from data_processing.normalization import normalize_data


def simulate_robot_client_obs(npz_path: str, timestep: int, camera_indices: list[int]) -> dict:
    """Build an observation dict exactly like robot_client._get_observation()."""
    z = np.load(npz_path, allow_pickle=True)
    obs = {}

    # --- Images (same as robot_client lines 221-232) ---
    for ci in camera_indices:
        color_img = z[f"images_cam{ci}"][timestep].copy()  # (H,W,3) uint8
        # Robot client: color_img.astype(np.float32) (line 227)
        color_img = color_img.astype(np.float32)
        obs[f"images_cam{ci}"] = color_img

        depth_img = z[f"depth_cam{ci}"][timestep].copy()
        depth_img = depth_img.astype(np.float32)
        obs[f"depth_cam{ci}"] = depth_img

    # --- Joint positions (same as robot_client lines 247-277) ---
    raw_jp = z["joint_positions"][timestep]
    jp_list = []
    for name in _JOINT_ORDER:
        v = raw_jp.get(name, 0.0) if isinstance(raw_jp, dict) else 0.0
        if isinstance(v, dict):
            v = v.get("position", 0.0)
        jp_list.append(float(v))
    obs["joint_positions"] = np.array(jp_list, dtype=np.float64)

    raw_je = z["joint_efforts"][timestep]
    je_list = []
    for name in _JOINT_ORDER:
        v = raw_je.get(name, 0.0) if isinstance(raw_je, dict) else 0.0
        if isinstance(v, dict):
            v = v.get("effort", 0.0)
        je_list.append(float(v))
    obs["joint_efforts"] = np.array(je_list, dtype=np.float64)

    raw_jv = z["joint_velocities"][timestep]
    jv_list = []
    for name in _JOINT_ORDER:
        v = raw_jv.get(name, 0.0) if isinstance(raw_jv, dict) else 0.0
        if isinstance(v, dict):
            v = v.get("velocity", 0.0)
        jv_list.append(float(v))
    obs["joint_velocities"] = np.array(jv_list, dtype=np.float64)

    obs["instruction"] = np.zeros(14, dtype=np.float32)
    obs["instruction"][0] = 1.0

    return obs


def simulate_json_roundtrip(obs: dict) -> dict:
    """Simulate send_json + recv_json (numpy → list → numpy)."""
    import json
    serialized = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            serialized[k] = v.tolist()
        else:
            serialized[k] = v
    json_str = json.dumps(serialized)
    return json.loads(json_str)


def build_training_batch(npz_path: str, timestep: int, camera_indices: list[int],
                         norm_stats: dict, obs_enc_cfg=None, obs_horizon: int = 1) -> dict[str, torch.Tensor]:
    """Build a batch the way the training dataset would."""
    rep = obs_enc_cfg.representation_type if hasattr(obs_enc_cfg, 'representation_type') else ["img", "depth", "pos"]
    ep = load_episode(npz_path, representation_type=rep, camera_indices=camera_indices, load_img=True)

    batch = {}
    # RGB: load_episode stores as rgb_cam{i}, dataset returns uint8 tensor
    cam_imgs = []
    for ci in camera_indices:
        key = f"rgb_cam{ci}"
        if key in ep:
            img = ep[key][timestep]  # (H, W, 3) numpy uint8
            cam_imgs.append(torch.from_numpy(img))
    if cam_imgs:
        # Match deploy shape: (1, T=1, num_cams, H, W, 3)
        batch["rgb"] = torch.stack(cam_imgs, dim=0).unsqueeze(0).unsqueeze(0)

    # Depth: load_episode stores as depth_cam{i}
    cam_deps = []
    for ci in camera_indices:
        key = f"depth_cam{ci}"
        if key in ep:
            dep = ep[key][timestep]
            cam_deps.append(torch.from_numpy(dep.astype(np.float32)))
    if cam_deps:
        # Match deploy shape: (1, T=1, num_cams, H, W)
        batch["depth"] = torch.stack(cam_deps, dim=0).unsqueeze(0).unsqueeze(0)

    # Pos: normalized like dataset
    if "pos" in ep:
        pos = ep["pos"][timestep:timestep+1].copy()  # (1, 30)
        if "pos" in norm_stats:
            pos = normalize_data(pos, norm_stats["pos"])
        batch["pos"] = torch.from_numpy(pos.astype(np.float32)).unsqueeze(0)  # (1, 1, 30)

    # Efforts
    if "efforts" in ep:
        eff = ep["efforts"][timestep:timestep+1, 6:30].copy()
        if "efforts" in norm_stats:
            eff = normalize_data(eff, norm_stats["efforts"])
        batch["efforts"] = torch.from_numpy(eff.astype(np.float32)).unsqueeze(0)

    return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/dp_light/latest.pt")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data", default="data/pick_up_left/data_0001.npz")
    p.add_argument("--timestep", type=int, default=50)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model, obs_enc_cfg, norm_stats, _amp_dtype = load_checkpoint(args.ckpt, device)
    cam_idxs = obs_enc_cfg.camera_indices
    print(f"Cameras: {cam_idxs}, Rep: {obs_enc_cfg.representation_type}")
    print()

    # ===== PATH 1: Simulate robot_client → deploy_policy =====
    print("=" * 60)
    print("PATH 1: robot_client → deploy_policy (deploy path)")
    print("=" * 60)

    obs = simulate_robot_client_obs(args.data, args.timestep, cam_idxs)
    print(f"Robot obs keys: {list(obs.keys())}")
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape} dtype={v.dtype} range=[{v.min():.2f}, {v.max():.2f}]")

    # Simulate JSON serialization (this is what actually happens over ZMQ)
    obs_json = simulate_json_roundtrip(obs)
    print(f"\nAfter JSON roundtrip:")
    for k, v in obs_json.items():
        if isinstance(v, list) and isinstance(v[0], list):
            arr = np.array(v)
            print(f"  {k}: shape={arr.shape} first_val_type={type(v[0][0]).__name__}")
        elif isinstance(v, list):
            arr = np.array(v)
            print(f"  {k}: shape={arr.shape} dtype_equiv={type(v[0]).__name__}")

    # Process through deploy pipeline
    deploy_batch = _obs_list_to_batch([obs_json], obs_enc_cfg, norm_stats, device, model=model)
    print(f"\nDeploy batch:")
    for k, v in deploy_batch.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype} "
              f"range=[{v.float().min():.4f}, {v.float().max():.4f}]")

    # ===== PATH 2: Training dataset path =====
    print()
    print("=" * 60)
    print("PATH 2: NPZ → dataset loading (training path)")
    print("=" * 60)

    train_batch = build_training_batch(args.data, args.timestep, cam_idxs, norm_stats, obs_enc_cfg)
    print(f"Training batch:")
    for k, v in train_batch.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype} "
              f"range=[{v.float().min():.4f}, {v.float().max():.4f}]")

    # ===== COMPARISON =====
    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)

    all_keys = set(list(deploy_batch.keys()) + list(train_batch.keys()))
    issues = []

    for key in sorted(all_keys):
        d = deploy_batch.get(key)
        t = train_batch.get(key)

        if d is None:
            print(f"  {key}: MISSING in deploy batch")
            issues.append(f"{key} missing in deploy")
            continue
        if t is None:
            print(f"  {key}: MISSING in training batch (OK — may not be in raw data)")
            continue

        d_cpu = d.float().cpu()
        t_cpu = t.float().cpu()

        # Shape check
        if d_cpu.shape != t_cpu.shape:
            print(f"  {key}: SHAPE MISMATCH  deploy={d_cpu.shape}  train={t_cpu.shape}")
            issues.append(f"{key} shape mismatch")
            continue

        # Value check
        diff = (d_cpu - t_cpu).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        status = "OK" if max_diff < 1e-4 else ("WARN" if max_diff < 0.01 else "MISMATCH")
        if status != "OK":
            issues.append(f"{key}: max_diff={max_diff:.6f}")

        print(f"  {key}: {status}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")

        if status == "MISMATCH" and key in ("pos", "eef", "hand_pos", "efforts", "velocity"):
            # Print detailed comparison
            print(f"    Deploy first 5:  {d_cpu.flatten()[:5].numpy()}")
            print(f"    Train first 5:   {t_cpu.flatten()[:5].numpy()}")

    print()
    if issues:
        print(f"ISSUES FOUND ({len(issues)}):")
        for iss in issues:
            print(f"  - {iss}")
    else:
        print("ALL CHECKS PASSED — deploy and training paths produce identical batches.")


if __name__ == "__main__":
    main()
