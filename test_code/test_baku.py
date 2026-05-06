"""
Unit tests for the native BAKU model.

Run:
    python test_code/test_baku.py
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.normalization import stats_to_json
import learning  # noqa: F401
from learning.baku.model import BakuConfig
from learning.common.encoders import ObsEncoder, ObsEncoderConfig
from learning.registry import build_model, list_models


class TestBaku(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _build_model(
        self,
        representation_type: list[str],
        use_instruction: bool = False,
        num_cams: int = 1,
        fuse_rgbd: bool = False,
        use_film: bool = False,
        precompute_rgb_features: bool = False,
    ):
        obs_cfg = ObsEncoderConfig(
            representation_type=representation_type,
            camera_indices=list(range(num_cams)),
            rgb_encoder_type="resnet18",
            depth_encoder_type="resnet18",
            fuse_rgbd=fuse_rgbd,
            precompute_rgb_features=precompute_rgb_features,
            rgb_per_cam_output=16,
            depth_per_cam_output=8,
            pos_output_size=16,
            efforts_output_size=8,
            use_instruction=use_instruction,
            num_instructions=14,
            instruction_embed_dim=8,
            enable_crop=False,
            enable_downsample=False,
        )
        obs_encoder = ObsEncoder(obs_cfg)
        model_cfg = BakuConfig(
            action_dim=30,
            obs_horizon=2,
            action_horizon=4,
            pred_horizon=6,
            use_instruction=use_instruction,
            num_instructions=14,
            instruction_embed_dim=8,
            hidden_size=32,
            depth=2,
            num_heads=4,
            ff_dim=64,
            dropout=0.0,
            use_film=use_film,
        )
        model = build_model("baku", obs_encoder=obs_encoder, config=model_cfg)
        model.norm_stats = {
            "action": {
                "min": np.full(30, -1.0, dtype=np.float32),
                "max": np.full(30, 1.0, dtype=np.float32),
            }
        }
        return model

    def _make_batch(
        self,
        batch_size: int = 2,
        t_obs: int = 2,
        t_pred: int = 6,
        with_rgb: bool = False,
        with_depth: bool = False,
        with_instruction: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {
            "pos": torch.rand(batch_size, t_obs, 30) * 2 - 1,
            "action": torch.rand(batch_size, t_pred, 30) * 2 - 1,
        }
        if with_rgb:
            batch["rgb"] = torch.randint(
                0, 256, (batch_size, t_obs, 1, 32, 32, 3), dtype=torch.uint8
            )
        if with_depth:
            batch["depth"] = torch.rand(batch_size, t_obs, 1, 32, 32)
        if with_instruction:
            batch["instruction"] = torch.randint(0, 14, (batch_size,), dtype=torch.long)
        return batch

    def _make_obs_list(
        self,
        obs_horizon: int,
        camera_indices: list[int],
        instruction_dim: int = 14,
    ) -> list[dict]:
        obs_list: list[dict] = []
        for _ in range(obs_horizon):
            obs = {
                "joint_positions": (torch.rand(30) * 2 - 1).tolist(),
                "joint_efforts": (torch.rand(30) * 2 - 1).tolist(),
                "joint_velocities": (torch.rand(30) * 2 - 1).tolist(),
                "instruction": [1.0] + [0.0] * (instruction_dim - 1),
            }
            for ci in camera_indices:
                obs[f"images_cam{ci}"] = torch.randint(
                    0, 256, (32, 32, 3), dtype=torch.uint8
                ).float().tolist()
                obs[f"depth_cam{ci}"] = torch.rand(32, 32).tolist()
            obs_list.append(obs)
        return obs_list

    def test_registry_contains_baku(self):
        self.assertIn("baku", list_models())

    def test_compute_loss_pos_only(self):
        model = self._build_model(["pos"])
        batch = self._make_batch()
        loss_dict = model.compute_loss(batch)
        self.assertIn("loss", loss_dict)
        self.assertIn("arm_mse", loss_dict)
        self.assertEqual(loss_dict["loss"].ndim, 0)

    def test_predict_action_with_instruction(self):
        model = self._build_model(["pos"], use_instruction=True)
        model.eval()
        batch = self._make_batch(with_instruction=True)
        batch.pop("action")
        action = model.predict_action(batch)
        self.assertEqual(tuple(action.shape), (2, 4, 30))

    def test_compute_loss_with_rgbd(self):
        model = self._build_model(["img", "depth", "pos"], use_instruction=True, fuse_rgbd=True)
        batch = self._make_batch(with_rgb=True, with_depth=True, with_instruction=True)
        loss_dict = model.compute_loss(batch)
        self.assertIn("loss", loss_dict)
        self.assertTrue(torch.isfinite(loss_dict["loss"]))

    def test_compute_loss_with_rgbd_and_film(self):
        model = self._build_model(
            ["img", "depth", "pos"],
            use_instruction=True,
            fuse_rgbd=True,
            use_film=True,
        )
        batch = self._make_batch(with_rgb=True, with_depth=True, with_instruction=True)
        loss_dict = model.compute_loss(batch)
        self.assertIn("loss", loss_dict)
        self.assertTrue(torch.isfinite(loss_dict["loss"]))

    def test_deploy_rgb_channel_swap_only_affects_predict_action(self):
        model = self._build_model(["img", "pos"], use_instruction=False)
        model.eval()

        rgb = torch.zeros((1, 2, 1, 32, 32, 3), dtype=torch.uint8)
        rgb[..., 0] = 8
        rgb[..., 1] = 64
        rgb[..., 2] = 240
        rgb_before = rgb.clone()
        batch = {
            "pos": torch.zeros(1, 2, 30),
            "rgb": rgb,
        }

        action_plain = model.predict_action(batch)
        model.enable_deploy_rgb_channel_swap(True)
        action_swapped = model.predict_action(batch)

        self.assertTrue(torch.equal(batch["rgb"], rgb_before))
        self.assertFalse(torch.allclose(action_plain, action_swapped))

    def test_film_auto_disables_without_instruction(self):
        model = self._build_model(["img", "pos"], use_instruction=False, use_film=True)
        self.assertFalse(model.use_film)

    def test_film_rejects_precomputed_rgb_features(self):
        with self.assertRaises(ValueError):
            self._build_model(
                ["img", "pos"],
                use_instruction=True,
                use_film=True,
                precompute_rgb_features=True,
            )

    def test_state_dict_round_trip(self):
        model_a = self._build_model(["pos"], use_instruction=True)
        model_b = self._build_model(["pos"], use_instruction=True)
        state = model_a.state_dict()
        model_b.load_state_dict(state)
        for key, value in state.items():
            self.assertTrue(torch.equal(value, model_b.state_dict()[key]), msg=key)

    def test_deploy_round_trip_with_film(self):
        try:
            from deploy_policy import _obs_list_to_batch, load_checkpoint
        except ImportError as exc:
            self.skipTest(f"deploy dependencies unavailable: {exc}")

        model = self._build_model(
            ["img", "depth", "pos"],
            use_instruction=True,
            fuse_rgbd=True,
            use_film=True,
        )
        ckpt = {
            "model_type": "baku",
            "model_state_dict": model.state_dict(),
            "obs_encoder_config": dataclasses.asdict(model.obs_encoder.config),
            "model_config": dataclasses.asdict(model.config),
            "norm_stats": stats_to_json(model.norm_stats),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "baku_latest.pt"
            torch.save(ckpt, ckpt_path)

            loaded_model, obs_enc_cfg, norm_stats, _amp_dtype = load_checkpoint(
                str(ckpt_path),
                torch.device("cpu"),
            )

            self.assertTrue(loaded_model.use_film)
            obs_list = self._make_obs_list(
                obs_horizon=loaded_model.config.obs_horizon,
                camera_indices=obs_enc_cfg.camera_indices,
                instruction_dim=obs_enc_cfg.num_instructions,
            )
            batch = _obs_list_to_batch(
                obs_list,
                obs_enc_cfg,
                norm_stats,
                torch.device("cpu"),
                model=loaded_model,
            )
            action = loaded_model.predict_action(batch)
            self.assertEqual(tuple(action.shape), (1, loaded_model.config.action_horizon, 30))
            self.assertTrue(torch.isfinite(action).all())


if __name__ == "__main__":
    unittest.main()
