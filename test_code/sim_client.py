"""
Simulated robot client for testing deploy_policy.py with real validation data.

Loads validation episodes from disk, sends observations to the policy server
via ZeroMQ (same protocol as robot_client.py), and compares predicted actions
with ground-truth actions. Reports per-model MSE metrics.

Usage:
    # First, start the server:
    python deploy_policy.py --ckpt checkpoints/dp/latest.pt --port 15001 --device cuda:1 --bind 127.0.0.1

    # Then run this test client:
    python test_code/sim_client.py --port 15001 --data_dir data/easy_mode \
        --num_episodes 3 --num_steps 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import zmq

# ---------------------------------------------------------------------------
# Joint ordering (must match robot_client.py and data_processing/loading.py)
# ---------------------------------------------------------------------------
JOINT_ORDER = [
    "ra_shoulder_pan_joint", "ra_shoulder_lift_joint", "ra_elbow_joint",
    "ra_wrist_1_joint", "ra_wrist_2_joint", "ra_wrist_3_joint",
    "rh_FFJ1", "rh_FFJ2", "rh_FFJ3", "rh_FFJ4",
    "rh_MFJ1", "rh_MFJ2", "rh_MFJ3", "rh_MFJ4",
    "rh_RFJ1", "rh_RFJ2", "rh_RFJ3", "rh_RFJ4",
    "rh_LFJ1", "rh_LFJ2", "rh_LFJ3", "rh_LFJ4", "rh_LFJ5",
    "rh_THJ1", "rh_THJ2", "rh_THJ3", "rh_THJ4", "rh_THJ5",
    "rh_WRJ1", "rh_WRJ2",
]


def dict_to_array(d: dict) -> np.ndarray:
    """Convert a joint dict to a 30-d array following JOINT_ORDER."""
    return np.array([d.get(j, 0.0) for j in JOINT_ORDER], dtype=np.float32)


def load_episode(episode_dir: Path):
    """Load a single episode from .npy directory."""
    imgs = {}
    depths = {}
    for ci in [0, 1, 2]:
        img_path = episode_dir / f"images_cam{ci}.npy"
        dep_path = episode_dir / f"depth_cam{ci}.npy"
        if img_path.exists():
            imgs[ci] = np.load(str(img_path), mmap_mode="r")
        if dep_path.exists():
            depths[ci] = np.load(str(dep_path), mmap_mode="r")

    jp_raw = np.load(str(episode_dir / "joint_positions.npy"), allow_pickle=True)
    je_raw = np.load(str(episode_dir / "joint_efforts.npy"), allow_pickle=True)
    jv_raw = np.load(str(episode_dir / "joint_velocities.npy"), allow_pickle=True)

    def to_array(raw):
        if raw.ndim == 1 and len(raw) > 0 and isinstance(raw[0], dict):
            return np.stack([dict_to_array(d) for d in raw])
        return raw.astype(np.float32)

    jp = to_array(jp_raw)
    je = to_array(je_raw)
    jv = to_array(jv_raw)

    return {
        "images": imgs,
        "depths": depths,
        "joint_positions": jp,
        "joint_efforts": je,
        "joint_velocities": jv,
        "T": jp.shape[0],
    }


def find_val_episodes(data_dir: Path, max_episodes: int = 5):
    """Find validation episodes across all instruction folders."""
    episodes = []
    for task_dir in sorted(data_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.isdigit():
            continue
        for split_dir in sorted(task_dir.iterdir()):
            if "_test" not in split_dir.name:
                continue
            for ep_dir in sorted(split_dir.iterdir()):
                if ep_dir.is_dir() and (ep_dir / "joint_positions.npy").exists():
                    instr_id = int(task_dir.name)
                    episodes.append((ep_dir, instr_id))
                    if len(episodes) >= max_episodes:
                        return episodes
    return episodes


def build_obs_dict(episode, t: int, cam_indices: list[int],
                   instruction: np.ndarray | None) -> dict:
    """Build a single observation dict as robot_client.py would send."""
    obs = {}
    for ci in cam_indices:
        if ci in episode["images"]:
            obs[f"images_cam{ci}"] = episode["images"][ci][t].tolist()
        if ci in episode["depths"]:
            obs[f"depth_cam{ci}"] = episode["depths"][ci][t].astype(np.float32).tolist()

    obs["joint_positions"] = episode["joint_positions"][t].tolist()
    obs["joint_efforts"] = episode["joint_efforts"][t].tolist()
    obs["joint_velocities"] = episode["joint_velocities"][t].tolist()

    if instruction is not None:
        obs["instruction"] = instruction.tolist()

    return obs


def run_test(port: int, data_dir: Path, num_episodes: int, num_steps: int):
    """Connect to server, send val data, compare predictions with ground truth."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 120000)  # 120s timeout
    sock.connect(f"tcp://127.0.0.1:{port}")

    # Get config
    sock.send_json({"type": "config_request", "timestamp": time.time()})
    config_resp = sock.recv_json()
    config = config_resp["config"]
    obs_horizon = config["obs_horizon"]
    action_horizon = config["action_horizon"]
    use_instruction = config["use_instruction"]
    instruction_dim = config.get("instruction_dim", 14)
    cam_indices = config.get("camera_indices", [0, 1, 2])
    print(f"Server config: obs_horizon={obs_horizon}, action_horizon={action_horizon}, "
          f"use_instruction={use_instruction}, instruction_dim={instruction_dim}, "
          f"cameras={cam_indices}")

    episodes = find_val_episodes(data_dir, max_episodes=num_episodes)
    if not episodes:
        print("No validation episodes found!")
        sock.close(); ctx.term()
        return

    print(f"Found {len(episodes)} validation episodes")

    all_mse, all_arm_mse, all_hand_mse, all_latencies = [], [], [], []
    errors = 0

    for ep_idx, (ep_dir, instr_id) in enumerate(episodes):
        print(f"\n--- Episode {ep_idx+1}/{len(episodes)}: {ep_dir.parent.name}/{ep_dir.name} (instr={instr_id}) ---")
        episode = load_episode(ep_dir)

        instruction = None
        if use_instruction:
            instruction = np.zeros(instruction_dim, dtype=np.float32)
            if instr_id < instruction_dim:
                instruction[instr_id] = 1.0

        T = episode["T"]
        max_t = T - action_horizon - 1
        if max_t <= 0:
            print(f"  Episode too short (T={T}), skipping")
            continue

        step_indices = np.linspace(0, max_t, min(num_steps, max_t + 1),
                                   dtype=int, endpoint=True)
        step_indices = np.unique(step_indices)

        ep_mses = []
        for t in step_indices:
            obs_list = []
            for h in range(obs_horizon):
                idx = max(0, t - obs_horizon + 1 + h)
                obs_list.append(build_obs_dict(episode, idx, cam_indices, instruction))

            msg = {"observation": obs_list, "timestamp": time.time()}
            t0 = time.perf_counter()
            sock.send_json(msg)
            resp = sock.recv_json()
            latency = (time.perf_counter() - t0) * 1000

            if "error" in resp:
                print(f"  t={t}: ERROR: {resp['error']}")
                errors += 1
                continue

            pred = np.array(resp["action"], dtype=np.float32)
            gt_end = min(t + 1 + pred.shape[0], T)
            gt = episode["joint_positions"][t + 1: gt_end]
            pred_trim = pred[:gt_end - (t + 1)]

            mse = float(np.mean((pred_trim - gt) ** 2))
            arm_mse = float(np.mean((pred_trim[:, :6] - gt[:, :6]) ** 2))
            hand_mse = float(np.mean((pred_trim[:, 6:] - gt[:, 6:]) ** 2))

            all_mse.append(mse)
            all_arm_mse.append(arm_mse)
            all_hand_mse.append(hand_mse)
            all_latencies.append(latency)
            ep_mses.append(mse)

        if ep_mses:
            print(f"  Steps: {len(ep_mses)}, Mean MSE: {np.mean(ep_mses):.6f}, "
                  f"Median latency: {np.median(all_latencies[-len(ep_mses):]):.0f}ms")

    sock.close()
    ctx.term()

    print("\n" + "=" * 60)
    print("DEPLOYMENT TEST RESULTS")
    print("=" * 60)
    if all_mse:
        print(f"  Episodes tested:    {len(episodes)}")
        print(f"  Total predictions:  {len(all_mse)}")
        print(f"  Errors:             {errors}")
        print(f"  Mean MSE:           {np.mean(all_mse):.6f}")
        print(f"  Mean Arm MSE:       {np.mean(all_arm_mse):.6f}")
        print(f"  Mean Hand MSE:      {np.mean(all_hand_mse):.6f}")
        print(f"  Median latency:     {np.median(all_latencies):.1f} ms")
        print(f"  P95 latency:        {np.percentile(all_latencies, 95):.1f} ms")
    else:
        print("  No successful predictions!")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--data_dir", type=str, default="data/easy_mode")
    p.add_argument("--num_episodes", type=int, default=5,
                   help="Max validation episodes to test")
    p.add_argument("--num_steps", type=int, default=10,
                   help="Timesteps to test per episode")
    args = p.parse_args()
    run_test(args.port, Path(args.data_dir), args.num_episodes, args.num_steps)
