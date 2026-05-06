"""
Unit and integration tests for the RDT (Robotics Diffusion Transformer) model.

Tests
-----
  TestT5TextEncoder    — buffer zero-init, forward shapes, trainability
  TestACIDecoderLayer  — single cross-attn, output shapes
  TestACIDecoder       — alternating lang/img conditioning, graceful fallback
  TestRDTTransformer   — forward pass shapes with/without text/visual/prop tokens
  TestRDTModel         — build(), compute_loss(), predict_action(), checkpoint

Architecture being tested (aligned with official RDT repo):
  • Self-attn sequence: [timestep_tok | ctrl_freq_tok | state_tok | action_1 … action_N]
  • Even ACI layers: cross-attend to lang_cond (T5 text tokens)
  • Odd  ACI layers: cross-attend to img_cond  (SigLIP patch tokens)
  • Unified state_adaptor for both state and action tokens
  • Condition positional embeddings on lang/img
  • RMSNorm + QK-norm, single cross-attn per layer
  • prediction_type="sample", DPMSolver inference

Usage
-----
# Run all tests
python test_code/test_rdt.py

# Run a specific test class
python test_code/test_rdt.py TestRDTModel
"""

from __future__ import annotations

import dataclasses
import io
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    from learning.rdt.model import (
        RMSNorm,
        T5TextEncoder,
        _ACIDecoderLayer,
        _ACIDecoder,
        _RDTTransformer,
        RDT,
        RDTConfig,
    )
    from learning.common.encoders import ObsEncoder, ObsEncoderConfig
    _IMPORT_OK = True
    _IMPORT_ERR = ""
except Exception as exc:
    _IMPORT_OK = False
    _IMPORT_ERR = str(exc)


def _skip_if_missing(cls):
    """Class decorator: skip every test when required packages are absent."""
    if not _IMPORT_OK:
        return unittest.skip(f"Skipping — import failed: {_IMPORT_ERR}")(cls)
    return cls


# ---------------------------------------------------------------------------
# Shared tiny dimensions
# ---------------------------------------------------------------------------

B           = 2    # batch size
T_obs       = 1    # obs_horizon
T_pred      = 8    # pred_horizon
T_act       = 4    # action_horizon
A_DIM       = 6    # action_dim (tiny)
H           = 64   # hidden_size
DEPTH       = 2    # ACI layers
HEADS       = 2    # attention heads
FF          = 64   # FFN dim (= hidden_size, paper: 1x expansion)
SEQ_LEN     = 8    # T5 token sequence length
T5_RAW_DIM  = 768  # raw T5-base output dim (no projection in T5TextEncoder)
N_INSTR     = 4    # number of instructions
SIGLIP_DIM  = 1152 # raw SigLIP patch token dim (fixed by backbone)
N_PATCHES   = 4    # tiny patch count for fast CPU tests (real = 729)
PROP_DIM    = 30   # proprioceptive dim (6 arm + 24 hand)


def _tiny_obs_encoder():
    """Pos-only ObsEncoder with a small output dim."""
    cfg = ObsEncoderConfig(
        representation_type=["pos"],
        pos_output_size=32,
    )
    return ObsEncoder(cfg)


def _tiny_rdt_config(**overrides):
    """Minimal RDTConfig that avoids any T5 download and runs fast on CPU."""
    cfg = RDTConfig(
        action_dim=A_DIM,
        obs_horizon=T_obs,
        pred_horizon=T_pred,
        action_horizon=T_act,
        num_instructions=N_INSTR,
        text_encoder_type="t5_base",
        text_token_max_len=SEQ_LEN,
        instructions_file="",          # no file → texts=None → zero buffers
        hidden_size=H,
        depth=DEPTH,
        num_heads=HEADS,
        ff_dim=FF,
        dropout=0.0,
        num_diffusion_iters=5,
        num_inference_iters=3,
        inference_scheduler="ddpm",    # fastest for tests
        prediction_type="sample",
        cond_mask_prob=0.0,            # disable for deterministic tests
        siglip_raw_dim=SIGLIP_DIM,
        prop_dim=PROP_DIM,
        ctrl_freq=1.0,
        max_lang_cond_len=SEQ_LEN,
        max_img_cond_len=64,           # small for fast CPU tests
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _pos_batch() -> dict:
    """Minimal batch: prop state only, no visual features."""
    return {
        "pos":    torch.randn(B, T_obs, PROP_DIM),
        "action": torch.randn(B, T_pred, A_DIM),
    }


def _patch_batch(n_cams: int = 1) -> dict:
    """Batch with synthetic SigLIP patch tokens (B, T, C, N_patches, siglip_raw_dim)."""
    return {
        "rgb_features": torch.randn(B, T_obs, n_cams, N_PATCHES, SIGLIP_DIM),
        "pos":          torch.randn(B, T_obs, PROP_DIM),
        "action":       torch.randn(B, T_pred, A_DIM),
    }


# ---------------------------------------------------------------------------
# TestRMSNorm
# ---------------------------------------------------------------------------

@_skip_if_missing
class TestRMSNorm(unittest.TestCase):

    def test_output_shape(self):
        norm = RMSNorm(H)
        x = torch.randn(B, T_pred, H)
        out = norm(x)
        self.assertEqual(out.shape, (B, T_pred, H))

    def test_unit_rms(self):
        norm = RMSNorm(H)
        x = torch.randn(B, T_pred, H)
        out = norm(x)
        rms = out.float().pow(2).mean(-1).sqrt()
        # After norm, RMS should be approximately 1
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=0.1))


# ---------------------------------------------------------------------------
# TestT5TextEncoder
# ---------------------------------------------------------------------------

@_skip_if_missing
class TestT5TextEncoder(unittest.TestCase):

    def _make(self) -> T5TextEncoder:
        return T5TextEncoder(
            num_instructions=N_INSTR,
            encoder_type="t5_base",
            token_max_len=SEQ_LEN,
            texts=None,
        )

    def test_zero_init_when_texts_none(self):
        enc = self._make()
        self.assertTrue(torch.all(enc.raw_tokens == 0).item(),
                        "raw_tokens should be zero when texts=None")

    def test_forward_shape(self):
        enc = self._make()
        ids = torch.randint(0, N_INSTR, (B,))
        tokens, mask = enc(ids)
        # No projection — returns raw T5 dim
        self.assertEqual(tokens.shape, (B, SEQ_LEN, T5_RAW_DIM))
        self.assertEqual(mask.shape, (B, SEQ_LEN))

    def test_raw_dim_attribute(self):
        enc = self._make()
        self.assertEqual(enc.raw_dim, 768)

    def test_no_trainable_params(self):
        """T5TextEncoder has no trainable parameters (no proj layer)."""
        enc = self._make()
        trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
        self.assertEqual(len(trainable), 0,
                         "T5TextEncoder should have no trainable params")

    def test_buffers_not_in_parameters(self):
        enc = self._make()
        param_names = {n for n, _ in enc.named_parameters()}
        self.assertNotIn("raw_tokens", param_names)
        self.assertNotIn("pad_mask",   param_names)

    def test_state_dict_round_trip(self):
        """Buffer values survive save → load (simulates checkpoint restore)."""
        enc1 = self._make()
        enc1.raw_tokens.fill_(3.14)

        buf = io.BytesIO()
        torch.save(enc1.state_dict(), buf)
        buf.seek(0)

        enc2 = self._make()
        enc2.load_state_dict(torch.load(buf, weights_only=True))
        self.assertTrue(torch.allclose(enc2.raw_tokens, enc1.raw_tokens),
                        "raw_tokens must be restored via load_state_dict")

    def test_buffer_follows_device(self):
        enc = self._make()
        enc.to("cpu")
        self.assertEqual(enc.raw_tokens.device.type, "cpu")


# ---------------------------------------------------------------------------
# TestACIDecoderLayer
# ---------------------------------------------------------------------------

@_skip_if_missing
class TestACIDecoderLayer(unittest.TestCase):

    def _make(self) -> _ACIDecoderLayer:
        return _ACIDecoderLayer(H, HEADS, FF, dropout=0.0)

    def test_output_shape_with_cond(self):
        layer = self._make()
        x    = torch.randn(B, T_pred, H)
        cond = torch.randn(B, N_PATCHES, H)
        out  = layer(x, cond=cond)
        self.assertEqual(out.shape, (B, T_pred, H))

    def test_output_shape_no_cond(self):
        """When cond=None, cross-attention is skipped."""
        layer = self._make()
        x     = torch.randn(B, T_pred, H)
        out   = layer(x, cond=None)
        self.assertEqual(out.shape, (B, T_pred, H))

    def test_output_shape_with_mask(self):
        layer = self._make()
        x    = torch.randn(B, T_pred, H)
        cond = torch.randn(B, SEQ_LEN, H)
        mask = torch.zeros(B, SEQ_LEN, dtype=torch.bool)
        out  = layer(x, cond=cond, cond_mask=mask)
        self.assertEqual(out.shape, (B, T_pred, H))

    def test_uses_rmsnorm(self):
        layer = self._make()
        self.assertIsInstance(layer.norm1, RMSNorm)
        self.assertIsInstance(layer.norm2, RMSNorm)
        self.assertIsInstance(layer.norm3, RMSNorm)

    def test_single_cross_attn(self):
        """Paper: single cross-attention module per layer."""
        layer = self._make()
        self.assertTrue(hasattr(layer, "cross_attn"))
        self.assertFalse(hasattr(layer, "obs_cross_attn"))
        self.assertFalse(hasattr(layer, "text_cross_attn"))


# ---------------------------------------------------------------------------
# TestACIDecoder
# ---------------------------------------------------------------------------

@_skip_if_missing
class TestACIDecoder(unittest.TestCase):

    def _make(self, depth: int = DEPTH) -> _ACIDecoder:
        return _ACIDecoder(depth, H, HEADS, FF, dropout=0.0)

    def test_output_shape_img_only(self):
        dec = self._make()
        tgt = torch.randn(B, T_pred, H)
        img = torch.randn(B, N_PATCHES, H)
        out = dec(tgt, img_cond=img)
        self.assertEqual(out.shape, (B, T_pred, H))

    def test_output_shape_lang_only(self):
        dec  = self._make()
        tgt  = torch.randn(B, T_pred, H)
        lang = torch.randn(B, SEQ_LEN, H)
        mask = torch.zeros(B, SEQ_LEN, dtype=torch.bool)
        out  = dec(tgt, lang_cond=lang, lang_mask=mask)
        self.assertEqual(out.shape, (B, T_pred, H))

    def test_output_shape_both_conds(self):
        dec  = self._make()
        tgt  = torch.randn(B, T_pred, H)
        lang = torch.randn(B, SEQ_LEN, H)
        img  = torch.randn(B, N_PATCHES, H)
        mask = torch.zeros(B, SEQ_LEN, dtype=torch.bool)
        out  = dec(tgt, lang_cond=lang, img_cond=img, lang_mask=mask)
        self.assertEqual(out.shape, (B, T_pred, H))

    def test_output_shape_no_conds(self):
        """No conditions → all cross-attentions are skipped."""
        dec = self._make()
        tgt = torch.randn(B, T_pred, H)
        out = dec(tgt)
        self.assertEqual(out.shape, (B, T_pred, H))

    def test_layer_count(self):
        dec = self._make()
        self.assertEqual(len(dec.layers), DEPTH)

    def test_deeper_model_no_crash(self):
        dec  = self._make(depth=4)
        tgt  = torch.randn(B, T_pred, H)
        lang = torch.randn(B, SEQ_LEN, H)
        img  = torch.randn(B, N_PATCHES, H)
        out  = dec(tgt, lang_cond=lang, img_cond=img)
        self.assertEqual(out.shape, (B, T_pred, H))


# ---------------------------------------------------------------------------
# TestRDTTransformer
# ---------------------------------------------------------------------------

@_skip_if_missing
class TestRDTTransformer(unittest.TestCase):

    def _make(self) -> _RDTTransformer:
        return _RDTTransformer(
            action_dim=A_DIM,
            siglip_raw_dim=SIGLIP_DIM,
            prop_dim=PROP_DIM,
            text_raw_dim=T5_RAW_DIM,
            pred_horizon=T_pred,
            hidden_size=H,
            depth=DEPTH,
            num_heads=HEADS,
            ff_dim=FF,
            dropout=0.0,
            causal_attn=False,
            diffusion_step_embed_dim=32,
            max_lang_cond_len=SEQ_LEN,
            max_img_cond_len=64,
        )

    def _visual(self, n_vis: int = N_PATCHES) -> "torch.Tensor":
        """Synthetic visual tokens: (B, N_vis, siglip_raw_dim)."""
        return torch.randn(B, n_vis, SIGLIP_DIM)

    def _prop(self) -> "torch.Tensor":
        return torch.randn(B, T_obs, PROP_DIM)

    def _text(self) -> tuple:
        return (
            torch.randn(B, SEQ_LEN, T5_RAW_DIM),
            torch.zeros(B, SEQ_LEN, dtype=torch.bool),
        )

    # --- forward shapes ---

    def test_forward_prop_only(self):
        model = self._make()
        noisy = torch.randn(B, T_pred, A_DIM)
        ts    = torch.randint(0, 5, (B,))
        out   = model(noisy, ts, prop_state=self._prop())
        self.assertEqual(out.shape, (B, T_pred, A_DIM))

    def test_forward_visual_only(self):
        model = self._make()
        noisy = torch.randn(B, T_pred, A_DIM)
        ts    = torch.randint(0, 5, (B,))
        out   = model(noisy, ts, visual_tokens=self._visual())
        self.assertEqual(out.shape, (B, T_pred, A_DIM))

    def test_forward_visual_and_prop(self):
        model = self._make()
        noisy = torch.randn(B, T_pred, A_DIM)
        ts    = torch.randint(0, 5, (B,))
        out   = model(noisy, ts, visual_tokens=self._visual(), prop_state=self._prop())
        self.assertEqual(out.shape, (B, T_pred, A_DIM))

    def test_forward_with_text(self):
        model = self._make()
        noisy = torch.randn(B, T_pred, A_DIM)
        ts    = torch.randint(0, 5, (B,))
        text, mask = self._text()
        out   = model(
            noisy, ts,
            visual_tokens=self._visual(),
            prop_state=self._prop(),
            text_tokens=text,
            text_mask=mask,
        )
        self.assertEqual(out.shape, (B, T_pred, A_DIM))

    def test_forward_multi_cam_visual(self):
        n_cams = 3
        n_vis  = T_obs * n_cams * N_PATCHES
        model  = self._make()
        noisy  = torch.randn(B, T_pred, A_DIM)
        ts     = torch.randint(0, 5, (B,))
        vis    = torch.randn(B, n_vis, SIGLIP_DIM)
        out    = model(noisy, ts, visual_tokens=vis, prop_state=self._prop())
        self.assertEqual(out.shape, (B, T_pred, A_DIM))

    # --- causal mask ---

    def test_causal_mask_is_upper_triangular(self):
        model = self._make()
        model.causal_attn = True
        T    = T_pred + 3  # time_tok + freq_tok + state_tok + actions
        mask = model._causal_mask(T, torch.device("cpu"))
        self.assertIsNotNone(mask)
        for i in range(T):
            for j in range(i + 1, T):
                self.assertEqual(mask[i, j].item(), float("-inf"))
        for i in range(T):
            self.assertEqual(mask[i, i].item(), 0.0)

    def test_no_causal_mask_when_disabled(self):
        model = self._make()
        model.causal_attn = False
        self.assertIsNone(model._causal_mask(T_pred, torch.device("cpu")))

    # --- adaptor shapes ---

    def test_visual_proj_is_2layer_mlp(self):
        model = self._make()
        # 2-layer MLP: Linear + GELU + Linear
        self.assertIsInstance(model.visual_proj, nn.Sequential)
        self.assertEqual(len(model.visual_proj), 3)  # Linear, GELU, Linear

    def test_state_adaptor_is_3layer_mlp(self):
        model = self._make()
        self.assertIsInstance(model.state_adaptor, nn.Sequential)
        self.assertEqual(len(model.state_adaptor), 5)  # L, G, L, G, L
        # Input dim should be action_dim * 2 (value + mask)
        self.assertEqual(model.state_adaptor[0].in_features, A_DIM * 2)

    def test_text_proj_is_2layer_mlp(self):
        model = self._make()
        self.assertIsInstance(model.text_proj, nn.Sequential)
        self.assertEqual(len(model.text_proj), 3)

    def test_output_head_is_2layer_mlp(self):
        model = self._make()
        self.assertIsInstance(model.output_head, nn.Sequential)
        self.assertEqual(len(model.output_head), 3)  # L, G, L

    def test_output_head_zero_init(self):
        """Paper: final output layer weights are zero-initialized."""
        model = self._make()
        final_linear = model.output_head[-1]
        self.assertTrue(torch.all(final_linear.weight == 0),
                        "Output layer weight should be zero-initialized")
        self.assertTrue(torch.all(final_linear.bias == 0),
                        "Output layer bias should be zero-initialized")

    def test_forward_with_ctrl_freq(self):
        model = self._make()
        noisy = torch.randn(B, T_pred, A_DIM)
        ts    = torch.randint(0, 5, (B,))
        freq  = torch.ones(B)
        out   = model(noisy, ts, prop_state=self._prop(), ctrl_freq=freq)
        self.assertEqual(out.shape, (B, T_pred, A_DIM))

    def test_condition_pos_embeds_exist(self):
        model = self._make()
        self.assertTrue(hasattr(model, "x_pos_embed"))
        self.assertTrue(hasattr(model, "lang_cond_pos_embed"))
        self.assertTrue(hasattr(model, "img_cond_pos_embed"))
        # x_pos_embed: (1, 3+T_pred, H) — timestep + ctrl_freq + state + actions
        self.assertEqual(model.x_pos_embed.shape, (1, 3 + T_pred, H))
        self.assertEqual(model.lang_cond_pos_embed.shape, (1, SEQ_LEN, H))
        self.assertEqual(model.img_cond_pos_embed.shape, (1, 64, H))

    def test_freq_emb_exists(self):
        model = self._make()
        self.assertTrue(hasattr(model, "freq_emb"))
        self.assertIsInstance(model.freq_emb, nn.Sequential)


# ---------------------------------------------------------------------------
# TestRDTModel
# ---------------------------------------------------------------------------

@_skip_if_missing
class TestRDTModel(unittest.TestCase):

    def _build(self, **cfg_overrides) -> RDT:
        obs_encoder = _tiny_obs_encoder()
        config      = _tiny_rdt_config(**cfg_overrides)
        return RDT.build(obs_encoder, config)

    # --- build ---

    def test_build_without_t5_download(self):
        model = self._build()
        self.assertIsInstance(model, RDT)

    def test_build_creates_visual_proj(self):
        model = self._build()
        self.assertTrue(hasattr(model.transformer, "visual_proj"))

    def test_build_creates_state_adaptor(self):
        model = self._build()
        self.assertTrue(hasattr(model.transformer, "state_adaptor"))
        self.assertIsNotNone(model.transformer.state_adaptor)

    def test_build_uses_rmsnorm(self):
        model = self._build()
        self.assertIsInstance(model.transformer.ln_out, RMSNorm)

    # --- compute_loss: prop-only ---

    def test_compute_loss_pos_only(self):
        model  = self._build()
        losses = model.compute_loss(_pos_batch())
        self.assertIn("loss", losses)
        self.assertGreater(losses["loss"].item(), 0.0)

    def test_compute_loss_returns_scalar(self):
        model = self._build()
        loss  = model.compute_loss(_pos_batch())["loss"]
        self.assertEqual(loss.shape, torch.Size([]))

    def test_loss_is_differentiable(self):
        model = self._build()
        loss  = model.compute_loss(_pos_batch())["loss"]
        loss.backward()

    # --- compute_loss: with SigLIP patch tokens ---

    def test_compute_loss_with_patch_features(self):
        model  = self._build()
        losses = model.compute_loss(_patch_batch())
        self.assertIn("loss", losses)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_compute_loss_multicam_patches(self):
        model  = self._build()
        losses = model.compute_loss(_patch_batch(n_cams=3))
        self.assertIn("loss", losses)
        self.assertTrue(torch.isfinite(losses["loss"]))

    # --- compute_loss: with instruction ---

    def test_compute_loss_with_instruction(self):
        model = self._build()
        batch = _pos_batch()
        batch["instruction"] = torch.randint(0, N_INSTR, (B,))
        losses = model.compute_loss(batch)
        self.assertIn("loss", losses)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_compute_loss_patches_plus_instruction(self):
        model = self._build()
        batch = _patch_batch()
        batch["instruction"] = torch.randint(0, N_INSTR, (B,))
        losses = model.compute_loss(batch)
        self.assertTrue(torch.isfinite(losses["loss"]))

    # --- condition masking ---

    def test_condition_masking_no_crash(self):
        """Training with cond_mask_prob > 0 should not crash."""
        model = self._build(cond_mask_prob=0.5)
        model.train()
        batch = _patch_batch()
        batch["instruction"] = torch.randint(0, N_INSTR, (B,))
        losses = model.compute_loss(batch)
        self.assertTrue(torch.isfinite(losses["loss"]))

    # --- predict_action ---

    def test_predict_action_shape_pos_only(self):
        model = self._build()
        model.eval()
        with torch.no_grad():
            action = model.predict_action(_pos_batch())
        self.assertEqual(action.shape, (B, T_act, A_DIM))

    def test_predict_action_shape_with_patches(self):
        model = self._build()
        model.eval()
        with torch.no_grad():
            action = model.predict_action(_patch_batch())
        self.assertEqual(action.shape, (B, T_act, A_DIM))

    def test_predict_action_with_instruction(self):
        model = self._build()
        model.eval()
        batch = _pos_batch()
        batch["instruction"] = torch.randint(0, N_INSTR, (B,))
        with torch.no_grad():
            action = model.predict_action(batch)
        self.assertEqual(action.shape, (B, T_act, A_DIM))

    # --- optimizer ---

    def test_configure_optimizers_returns_one_adamw(self):
        model = self._build()
        opts  = model.configure_optimizers()
        self.assertEqual(len(opts), 1)
        self.assertIsInstance(opts[0], torch.optim.AdamW)

    def test_t5_buffers_excluded_from_optimizer(self):
        model    = self._build()
        opts     = model.configure_optimizers()
        all_ids  = {id(p) for opt in opts for pg in opt.param_groups for p in pg["params"]}
        self.assertNotIn(id(model.text_encoder.raw_tokens), all_ids,
                         "frozen T5 buffer should not be in optimizer")

    # --- EMA ---

    def test_ema_step_does_not_crash(self):
        model = self._build()
        loss  = model.compute_loss(_pos_batch())["loss"]
        loss.backward()
        for opt in model.configure_optimizers():
            opt.step()
        model.on_after_step()

    # --- encode: 5-D vs 4-D rgb_features ---

    def test_encode_5d_gives_patch_tokens(self):
        model = self._build()
        batch = _patch_batch(n_cams=2)
        vis, prop, text, mask = model._encode(batch)
        expected_n = T_obs * 2 * N_PATCHES
        self.assertIsNotNone(vis)
        self.assertEqual(vis.shape, (B, expected_n, SIGLIP_DIM))

    def test_encode_4d_gives_pooled_tokens(self):
        model = self._build()
        pooled_D = 1024
        batch = {
            "rgb_features": torch.randn(B, T_obs, 2, pooled_D),
            "pos":          torch.randn(B, T_obs, PROP_DIM),
            "action":       torch.randn(B, T_pred, A_DIM),
        }
        vis, prop, text, mask = model._encode(batch)
        self.assertIsNotNone(vis)
        self.assertEqual(vis.shape, (B, T_obs * 2, pooled_D))

    def test_encode_no_visual_returns_none(self):
        model = self._build()
        vis, _, _, _ = model._encode(_pos_batch())
        self.assertIsNone(vis)

    def test_encode_prop_extracted(self):
        model = self._build()
        _, prop, _, _ = model._encode(_pos_batch())
        self.assertIsNotNone(prop)
        self.assertEqual(prop.shape, (B, T_obs, PROP_DIM))

    # --- checkpoint round-trip ---

    def test_checkpoint_round_trip(self):
        from data_processing.normalization import stats_to_json

        model1 = self._build()
        batch  = _pos_batch()
        model1.eval()
        with torch.no_grad():
            loss1 = model1.compute_loss(batch)["loss"].item()

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "rdt_test.pt"
            torch.save({
                "model_state_dict":   model1.state_dict(),
                "obs_encoder_config": dataclasses.asdict(model1.obs_encoder.config),
                "model_config":       dataclasses.asdict(model1.config),
                "norm_stats":         stats_to_json({}),
            }, ckpt_path)

            ckpt = torch.load(ckpt_path, weights_only=False)

        obs_enc2 = ObsEncoder(ObsEncoderConfig(**ckpt["obs_encoder_config"]))
        model2   = RDT.build(obs_enc2, RDTConfig(**ckpt["model_config"]))
        model2.load_state_dict(ckpt["model_state_dict"])

        model2.eval()
        with torch.no_grad():
            loss2 = model2.compute_loss(batch)["loss"].item()

        self.assertAlmostEqual(loss1, loss2, places=5,
                               msg="Loss must match after checkpoint round-trip")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn as nn  # needed for test assertions
    unittest.main(verbosity=2)
