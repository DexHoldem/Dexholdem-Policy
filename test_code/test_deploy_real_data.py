"""
End-to-end deployment test using real data from data/easy_mode.

Loads a real NPZ episode, feeds observations through the PolicyServer
(same protocol as robot_client.py), and verifies that the predicted
actions have MSE < 0.05 against the ground-truth joint positions.

This validates the full deploy pipeline: checkpoint loading,
observation normalization, model inference, and action un-normalization.

Usage:
    # Test with DP checkpoint (default)
    python test_code/test_deploy_real_data.py \
        --ckpt checkpoints/dp_exp1/latest.pt

    # Test with ACT checkpoint
    python test_code/test_deploy_real_data.py \
        --ckpt checkpoints/act_exp1/latest.pt

    # Specify GPU
    python test_code/test_deploy_real_data.py \
        --ckpt checkpoints/dp_exp1/latest.pt --device cuda:0

    # Custom data file
    python test_code/test_deploy_real_data.py \
        --ckpt checkpoints/dp_exp1/latest.pt \
        --npz data/easy_mode/0/pick_up_card_test/data0003.npz
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Joint ordering must match record_data.py / robot_client.py / loading.py
_JOINT_ORDER = [
    "ra_shoulder_pan_joint", "ra_shoulder_lift_joint", "ra_elbow_joint",
    "ra_wrist_1_joint",      "ra_wrist_2_joint",       "ra_wrist_3_joint",
    "rh_FFJ1", "rh_FFJ2", "rh_FFJ3", "rh_FFJ4",
    "rh_MFJ1", "rh_MFJ2", "rh_MFJ3", "rh_MFJ4",
    "rh_RFJ1", "rh_RFJ2", "rh_RFJ3", "rh_RFJ4",
    "rh_LFJ1", "rh_LFJ2", "rh_LFJ3", "rh_LFJ4", "rh_LFJ5",
    "rh_THJ1", "rh_THJ2", "rh_THJ3", "rh_THJ4", "rh_THJ5",
    "rh_WRJ1", "rh_WRJ2",
]
assert len(_JOINT_ORDER) == 30


# ---------------------------------------------------------------------------
# Load real NPZ episode into robot_client-style observation dicts
# ---------------------------------------------------------------------------

def load_episode_as_obs_list(
    npz_path: str,
    camera_indices: list[int],
    instruction_id: int,
    instruction_dim: int,
) -> tuple[list[dict], np.ndarray]:
    """
    Load one NPZ file and convert each timestep into the dict format
    that robot_client.py would send over ZeroMQ.

    Returns:
        obs_dicts:    List of T observation dicts (JSON-serializable).
        gt_positions: (T, 30) ground-truth joint positions.
    """
    data = np.load(npz_path, allow_pickle=True)
    T = data["images_cam0"].shape[0]

    # Extract ground-truth joint positions (T, 30)
    raw_jp = data["joint_positions"]  # (T,) object array of dicts
    gt_positions = np.zeros((T, 30), dtype=np.float32)
    for t in range(T):
        d = raw_jp[t] if raw_jp[t] is not None else {}
        for j, name in enumerate(_JOINT_ORDER):
            gt_positions[t, j] = float(d.get(name, 0.0))

    # Extract joint efforts (T, 30)
    raw_je = data["joint_efforts"]
    efforts = np.zeros((T, 30), dtype=np.float32)
    for t in range(T):
        d = raw_je[t] if raw_je[t] is not None else {}
        for j, name in enumerate(_JOINT_ORDER):
            efforts[t, j] = float(d.get(name, 0.0))

    # Extract joint velocities (T, 30)
    raw_jv = data["joint_velocities"]
    velocities = np.zeros((T, 30), dtype=np.float32)
    for t in range(T):
        d = raw_jv[t] if raw_jv[t] is not None else {}
        for j, name in enumerate(_JOINT_ORDER):
            velocities[t, j] = float(d.get(name, 0.0))

    # Build instruction one-hot
    instr_vec = np.zeros(instruction_dim, dtype=np.float32)
    instr_vec[min(instruction_id, instruction_dim - 1)] = 1.0

    # Build per-timestep observation dicts (same format as robot_client)
    obs_dicts = []
    for t in range(T):
        obs: dict = {}

        # Images — convert to float32 list (robot_client sends float32)
        for ci in camera_indices:
            key = f"images_cam{ci}"
            if key in data:
                obs[key] = data[key][t].astype(np.float32).tolist()
            key = f"depth_cam{ci}"
            if key in data:
                obs[key] = data[key][t].astype(np.float32).tolist()

        # Proprioception
        obs["joint_positions"] = gt_positions[t].tolist()
        obs["joint_efforts"] = efforts[t].tolist()
        obs["joint_velocities"] = velocities[t].tolist()

        # Instruction
        obs["instruction"] = instr_vec.tolist()

        obs_dicts.append(obs)

    return obs_dicts, gt_positions


# ---------------------------------------------------------------------------
# Test client
# ---------------------------------------------------------------------------

def run_test_client(
    port: int,
    npz_path: str,
    mse_threshold: float = 0.05,
    num_test_steps: int = 5,
    instruction_id: int = 0,
):
    """
    Connect to PolicyServer, send real observations, compare predicted
    actions to ground-truth joint positions.
    """
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{port}")

    # --- Config handshake ---
    print("\n[test] Sending config_request ...")
    sock.send_json({"type": "config_request", "timestamp": time.time()})
    config_resp = sock.recv_json()
    assert "config" in config_resp, f"Expected 'config', got: {config_resp}"

    cfg = config_resp["config"]
    print(f"[test] Server config: {cfg}")

    obs_horizon = cfg["obs_horizon"]
    action_horizon = cfg["action_horizon"]
    camera_indices = cfg["camera_indices"]
    use_instruction = cfg["use_instruction"]
    instruction_dim = cfg.get("instruction_dim", 1) if use_instruction else 1

    # --- Load real data ---
    print(f"\n[test] Loading episode from {npz_path} ...")
    obs_dicts, gt_positions = load_episode_as_obs_list(
        npz_path, camera_indices, instruction_id, instruction_dim,
    )
    T = len(obs_dicts)
    print(f"[test] Episode length: {T} timesteps")

    # Pick evenly spaced test timesteps (avoid very start/end)
    margin = obs_horizon
    end_margin = action_horizon
    usable = T - margin - end_margin
    if usable <= 0:
        print(f"[test] Episode too short ({T}) for obs_horizon={obs_horizon} "
              f"+ action_horizon={action_horizon}")
        sock.close()
        ctx.term()
        sys.exit(1)

    num_test_steps = min(num_test_steps, usable)
    test_indices = np.linspace(margin, T - end_margin - 1, num_test_steps, dtype=int)

    all_mses = []
    all_passed = True

    for step_i, t in enumerate(test_indices):
        print(f"\n[test] === Step {step_i + 1}/{num_test_steps}  (t={t}) ===")

        # Build obs_list of length obs_horizon ending at t
        start = max(0, t - obs_horizon + 1)
        obs_list = [obs_dicts[i] for i in range(start, t + 1)]
        # Pad from the left if needed
        while len(obs_list) < obs_horizon:
            obs_list.insert(0, obs_list[0])

        message = {"observation": obs_list, "timestamp": time.time()}
        t0 = time.perf_counter()
        sock.send_json(message)
        response = sock.recv_json()
        dt = time.perf_counter() - t0

        if "error" in response:
            print(f"[test] ERROR from server: {response['error']}")
            all_passed = False
            continue

        pred_actions = np.array(response["action"])  # (action_horizon, 30)
        print(f"[test] Predicted action shape={pred_actions.shape}  ({dt*1e3:.0f} ms)")

        # Validate shape
        assert pred_actions.ndim == 2 and pred_actions.shape[1] == 30, \
            f"Unexpected action shape: {pred_actions.shape}"
        assert np.all(np.isfinite(pred_actions)), "Action contains NaN/Inf!"

        # Compare against ground-truth future positions
        gt_start = t + 1
        gt_end = min(t + 1 + action_horizon, T)
        n_compare = gt_end - gt_start
        if n_compare <= 0:
            print(f"[test] No ground-truth future steps available at t={t}")
            continue

        gt_future = gt_positions[gt_start:gt_end]             # (n_compare, 30)
        pred_future = pred_actions[:n_compare]                  # (n_compare, 30)

        mse = np.mean((pred_future - gt_future) ** 2)
        per_joint_mse = np.mean((pred_future - gt_future) ** 2, axis=0)  # (30,)
        arm_mse = np.mean(per_joint_mse[:6])
        hand_mse = np.mean(per_joint_mse[6:])

        all_mses.append(mse)

        status = "PASS" if mse < mse_threshold else "FAIL"
        if mse >= mse_threshold:
            all_passed = False

        print(f"[test] MSE = {mse:.6f}  (arm={arm_mse:.6f}, hand={hand_mse:.6f})  "
              f"threshold={mse_threshold}  [{status}]")
        print(f"[test] Pred range: [{pred_actions.min():.4f}, {pred_actions.max():.4f}]")
        print(f"[test] GT   range: [{gt_future.min():.4f}, {gt_future.max():.4f}]")

    sock.close()
    ctx.term()

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_mses:
        avg_mse = np.mean(all_mses)
        max_mse = np.max(all_mses)
        print(f"[test] Average MSE: {avg_mse:.6f}")
        print(f"[test] Max MSE:     {max_mse:.6f}")
        print(f"[test] Threshold:   {mse_threshold}")
    if all_passed:
        print(f"[test] ALL {num_test_steps} STEPS PASSED (MSE < {mse_threshold})")
    else:
        print(f"[test] SOME STEPS FAILED (MSE >= {mse_threshold})")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Server launcher
# ---------------------------------------------------------------------------

def start_server_thread(ckpt_path: str, port: int, device: str) -> threading.Thread:
    from deploy_policy import PolicyServer

    server = PolicyServer(ckpt_path, port=port, device_str=device, bind_ip="127.0.0.1")
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(2)  # give server time to load model and bind
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Test deploy_policy with real data and verify action MSE."
    )
    p.add_argument("--ckpt", required=True, help="Path to checkpoint.")
    p.add_argument("--npz", default="data/easy_mode/0/pick_up_card_test/data0001.npz",
                   help="Path to test NPZ episode.")
    p.add_argument("--instruction_id", type=int, default=0,
                   help="Instruction ID for the test episode.")
    p.add_argument("--port", type=int, default=15679,
                   help="Port for test server (avoid conflicts).")
    p.add_argument("--device", default="cuda",
                   help="Device for model inference.")
    p.add_argument("--mse_threshold", type=float, default=0.05,
                   help="Maximum allowed MSE.")
    p.add_argument("--num_steps", type=int, default=5,
                   help="Number of timesteps to test.")
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(1)
    npz = Path(args.npz)
    if not npz.exists():
        print(f"NPZ file not found: {npz}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  test_deploy_real_data.py")
    print("=" * 60)
    print(f"  Checkpoint:     {ckpt}")
    print(f"  Data:           {npz}")
    print(f"  Instruction ID: {args.instruction_id}")
    print(f"  Device:         {args.device}")
    print(f"  MSE threshold:  {args.mse_threshold}")
    print(f"  Test steps:     {args.num_steps}")
    print("=" * 60)

    # Start server
    print("\n[server] Starting PolicyServer in background thread ...")
    start_server_thread(str(ckpt), args.port, args.device)

    # Run test
    run_test_client(
        port=args.port,
        npz_path=str(npz),
        mse_threshold=args.mse_threshold,
        num_test_steps=args.num_steps,
        instruction_id=args.instruction_id,
    )


if __name__ == "__main__":
    main()
