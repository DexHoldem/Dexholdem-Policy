"""
End-to-end test for deploy_policy.py + robot_client protocol.

Launches the PolicyServer in a background thread and simulates the
robot_client handshake + inference loop with synthetic observations.

Usage:
    python test_code/test_deploy.py --ckpt checkpoints/act_exp1/latest.pt
    python test_code/test_deploy.py --ckpt checkpoints/act_exp1/latest.pt --device cpu
    python test_code/test_deploy.py --ckpt checkpoints/act_exp1/latest.pt --num_rounds 5
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import zmq

# ---------------------------------------------------------------------------
# Synthetic observation builder
# ---------------------------------------------------------------------------

def make_fake_observation(
    camera_indices: list[int],
    instruction_dim: int,
    instruction_id: int = 0,
    img_h: int = 480,
    img_w: int = 640,
) -> dict:
    """Build one observation dict matching what robot_client sends."""
    obs = {}

    # RGB + depth per camera
    for ci in camera_indices:
        obs[f"images_cam{ci}"] = np.random.randint(
            0, 256, (img_h, img_w, 3), dtype=np.uint8
        ).astype(np.float32).tolist()
        obs[f"depth_cam{ci}"] = np.random.rand(img_h, img_w).astype(np.float32).tolist()

    # Proprioception (30-dim: 6 arm + 24 hand)
    obs["joint_positions"] = np.random.randn(30).astype(np.float32).tolist()
    obs["joint_efforts"] = np.random.randn(30).astype(np.float32).tolist()
    obs["joint_velocities"] = np.random.randn(30).astype(np.float32).tolist()

    # Instruction (one-hot)
    instr = np.zeros(instruction_dim, dtype=np.float32)
    instr[min(instruction_id, instruction_dim - 1)] = 1.0
    obs["instruction"] = instr.tolist()

    return obs


# ---------------------------------------------------------------------------
# Simulated client
# ---------------------------------------------------------------------------

def run_fake_client(port: int, num_rounds: int = 3):
    """Mimic robot_client.py: config handshake then inference loop."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://localhost:{port}")

    # --- Step 1: config handshake ---
    print("\n[client] Sending config_request ...")
    sock.send_json({"type": "config_request", "timestamp": time.time()})
    config_resp = sock.recv_json()

    assert "config" in config_resp, f"Expected 'config' key, got: {config_resp}"
    cfg = config_resp["config"]
    print(f"[client] Server config: {cfg}")

    obs_horizon = cfg["obs_horizon"]
    action_horizon = cfg["action_horizon"]
    camera_indices = cfg["camera_indices"]
    use_instruction = cfg["use_instruction"]
    instruction_dim = cfg.get("instruction_dim", 1) if use_instruction else 1

    # --- Step 2: inference rounds ---
    for rnd in range(1, num_rounds + 1):
        print(f"\n[client] === Round {rnd}/{num_rounds} ===")

        # Build obs_list of length obs_horizon
        obs_list = [
            make_fake_observation(camera_indices, instruction_dim)
            for _ in range(obs_horizon)
        ]

        message = {"observation": obs_list, "timestamp": time.time()}
        print(f"[client] Sending observation (obs_horizon={obs_horizon}) ...")
        t0 = time.perf_counter()
        sock.send_json(message)
        response = sock.recv_json()
        dt = time.perf_counter() - t0

        if "error" in response:
            print(f"[client] ERROR from server: {response['error']}")
            sock.close()
            ctx.term()
            sys.exit(1)

        action = np.array(response["action"])
        print(f"[client] Got action shape={action.shape}  ({dt*1e3:.0f} ms)")

        # Validate shape
        assert action.ndim == 2, f"Expected 2D action, got shape {action.shape}"
        assert action.shape[0] == action_horizon, (
            f"action_horizon mismatch: expected {action_horizon}, got {action.shape[0]}"
        )
        assert action.shape[1] == 30, (
            f"action_dim mismatch: expected 30, got {action.shape[1]}"
        )
        assert np.all(np.isfinite(action)), "Action contains NaN/Inf!"

        print(f"[client] Action range: [{action.min():.4f}, {action.max():.4f}]")
        print(f"[client] Round {rnd} PASSED")

    sock.close()
    ctx.term()
    print(f"\n[client] All {num_rounds} rounds passed!")


# ---------------------------------------------------------------------------
# Server launcher (in-process, background thread)
# ---------------------------------------------------------------------------

def start_server_thread(ckpt_path: str, port: int, device: str) -> threading.Thread:
    """Start PolicyServer.run() in a daemon thread."""
    # Import here so the test script itself doesn't fail on import errors.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from deploy_policy import PolicyServer

    server = PolicyServer(ckpt_path, port=port, device_str=device, bind_ip="127.0.0.1")

    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    # Give server a moment to bind
    time.sleep(1)
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Test deploy_policy end-to-end with fake observations.")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint.")
    p.add_argument("--port", type=int, default=15678, help="Port for test server (default: 15678 to avoid conflicts).")
    p.add_argument("--device", default="cpu", help="Device for model inference.")
    p.add_argument("--num_rounds", type=int, default=3, help="Number of inference rounds.")
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(1)

    print(f"=== test_deploy.py ===")
    print(f"Checkpoint: {ckpt}")
    print(f"Port:       {args.port}")
    print(f"Device:     {args.device}")
    print(f"Rounds:     {args.num_rounds}")

    # Start server
    print("\n[server] Starting PolicyServer in background thread ...")
    start_server_thread(str(ckpt), args.port, args.device)

    # Run fake client
    run_fake_client(args.port, args.num_rounds)
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
