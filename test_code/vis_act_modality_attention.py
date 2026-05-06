"""Visualize per-modality attention in ACT's policy decoder.

ACT uses per-modality tokens (each projected to hidden_dim=512).
The policy decoder cross-attends action queries to the encoder memory:
  memory = policy_enc([z_tok, rgb_tok, pos_tok, instr_tok])

We extract cross-attention weights to see how much each action query
attends to each modality token.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from learning.common.encoders import ObsEncoder, ObsFeatures, InstructionEncoder

# ─── Load checkpoint ──────────────────────────────────────────────
print("Loading ACT checkpoint …")
ckpt = torch.load("checkpoints/act/latest.pt", map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]
model_cfg = ckpt["model_config"]
obs_cfg = ckpt["obs_encoder_config"]

D = model_cfg["hidden_dim"]       # 512
num_heads = model_cfg["num_heads"] # 8

# ─── Reconstruct the model parts we need ──────────────────────────
import torch.nn as nn
import math

# Modality projections
modality_projs = nn.ModuleDict()
for key in ["rgb", "pos", "instruction"]:
    prefix = f"modality_projs.{key}"
    w0 = sd[f"{prefix}.0.weight"]
    modality_projs[key] = nn.Sequential(
        nn.Linear(w0.shape[1], w0.shape[0]),
        nn.LayerNorm(w0.shape[0]),
    )

# Load weights
for key in ["rgb", "pos", "instruction"]:
    prefix = f"modality_projs.{key}"
    modality_projs[key][0].weight = nn.Parameter(sd[f"{prefix}.0.weight"])
    modality_projs[key][0].bias = nn.Parameter(sd[f"{prefix}.0.bias"])
    modality_projs[key][1].weight = nn.Parameter(sd[f"{prefix}.1.weight"])
    modality_projs[key][1].bias = nn.Parameter(sd[f"{prefix}.1.bias"])

# Latent projection
latent_proj = nn.Linear(model_cfg["latent_dim"], D)
latent_proj.weight = nn.Parameter(sd["latent_proj.weight"])
latent_proj.bias = nn.Parameter(sd["latent_proj.bias"])

# Sine positional encoding
class _SinePosEnc(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]

pos_enc = _SinePosEnc(D)

# Policy encoder
enc_layer = nn.TransformerEncoderLayer(
    d_model=D, nhead=num_heads,
    dim_feedforward=model_cfg["dim_feedforward"],
    dropout=0.0, batch_first=True,
)
policy_enc = nn.TransformerEncoder(enc_layer, num_layers=model_cfg["num_encoder_layers"])
policy_enc_sd = {k.replace("policy_enc.", ""): v for k, v in sd.items() if k.startswith("policy_enc.")}
policy_enc.load_state_dict(policy_enc_sd)

# Policy decoder
dec_layer = nn.TransformerDecoderLayer(
    d_model=D, nhead=num_heads,
    dim_feedforward=model_cfg["dim_feedforward"],
    dropout=0.0, batch_first=True,
)
policy_dec = nn.TransformerDecoder(dec_layer, num_layers=model_cfg["num_decoder_layers"])
policy_dec_sd = {k.replace("policy_dec.", ""): v for k, v in sd.items() if k.startswith("policy_dec.")}
policy_dec.load_state_dict(policy_dec_sd)

# Action queries
action_queries = nn.Embedding(model_cfg["pred_horizon"], D)
action_queries.weight = nn.Parameter(sd["action_queries.weight"])

# Instruction encoder
instr_enc = InstructionEncoder(num_instructions=14, embed_dim=64)
instr_sd = {k.replace("obs_encoder.instruction_encoder.", ""): v
            for k, v in sd.items() if "instruction_encoder" in k}
instr_enc.load_state_dict(instr_sd)

# Set all to eval
for m in [modality_projs, latent_proj, policy_enc, policy_dec, action_queries, instr_enc]:
    m.eval()

# ─── Build inputs ────────────────────────────────────────────────
print("Building inputs …")
B = 14  # one per instruction
T_obs = 1
pred_horizon = model_cfg["pred_horizon"]  # 64

# RGB features (fuse_rgbd=True, so 3 cams × (128+32) = 480 per cam → fused)
# Actually with fuse_rgbd, rgb dim = rgb_per_cam*3 + depth_per_cam*3 = 128*3+32*3 = 480
torch.manual_seed(42)
rgb_feat = torch.randn(1, T_obs, 480).expand(B, -1, -1).clone()

# Pos features
pos_feat = torch.randn(1, T_obs, 128).expand(B, -1, -1).clone()

# Instruction features (different per sample)
with torch.no_grad():
    instr_embs = instr_enc(torch.arange(14))  # (14, 64)
instr_feat = instr_embs.unsqueeze(1)  # (14, 1, 64)

# ─── Forward with attention extraction ────────────────────────────
print("Running forward with attention extraction …")

with torch.no_grad():
    # Project each modality to tokens
    rgb_tok = modality_projs["rgb"](rgb_feat)          # (B, 1, 512)
    pos_tok = modality_projs["pos"](pos_feat)          # (B, 1, 512)
    instr_tok = modality_projs["instruction"](instr_feat)  # (B, 1, 512)

    obs_tokens = torch.cat([rgb_tok, pos_tok, instr_tok], dim=1)  # (B, 3, 512)

    # Policy encoder: [z_tok, rgb_tok, pos_tok, instr_tok]
    z = torch.zeros(B, model_cfg["latent_dim"])  # inference z = 0
    z_tok = latent_proj(z).unsqueeze(1)  # (B, 1, 512)
    src = torch.cat([z_tok, obs_tokens], dim=1)  # (B, 4, 512)
    src = pos_enc(src)

    memory = policy_enc(src)  # (B, 4, 512)
    # memory tokens: [z, rgb, pos, instruction]

    # Policy decoder: action queries cross-attend to memory
    queries = action_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 64, 512)
    queries = pos_enc(queries)

    # Manual decoder forward to extract cross-attention
    x = queries
    cross_attn_per_layer = []
    cross_attn_per_head = []

    for layer in policy_dec.layers:
        # Self-attention
        x2 = layer.norm1(x)
        x = x + layer.dropout1(layer.self_attn(x2, x2, x2, need_weights=False)[0])

        # Cross-attention — extract weights
        x2 = layer.norm2(x)
        ao, aw_avg = layer.multihead_attn(
            x2, memory, memory, need_weights=True, average_attn_weights=True
        )
        x = x + layer.dropout2(ao)
        cross_attn_per_layer.append(aw_avg)  # (B, 64, 4)

        # Also per-head
        x3 = layer.norm2(x - layer.dropout2(ao))  # re-normalize (approx)
        _, aw_heads = layer.multihead_attn(
            x3, memory, memory, need_weights=True, average_attn_weights=False
        )
        cross_attn_per_head.append(aw_heads)  # (B, 8, 64, 4)

        # FFN
        x2 = layer.norm3(x)
        x = x + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(x2)))))


# ─── Analysis ─────────────────────────────────────────────────────
TOKEN_NAMES = ["Latent z", "RGB\n(fused)", "Proprio-\nception", "Instr-\nuction"]
TOKEN_COLORS = ["#9b59b6", "#e74c3c", "#2ecc71", "#f39c12"]
num_layers = len(cross_attn_per_layer)

# Per-layer avg attention to each token
layer_attn = np.zeros((num_layers, 4))
for li, aw in enumerate(cross_attn_per_layer):
    layer_attn[li] = aw.mean(dim=(0, 1)).numpy()  # avg over batch & action queries

# Per-instruction attention variation
instr_attn_per_layer = np.zeros((num_layers, 14))
for li, aw in enumerate(cross_attn_per_layer):
    instr_attn_per_layer[li] = aw[:, :, 3].mean(dim=1).numpy()  # (14,) instruction attn

# ─── Plot ─────────────────────────────────────────────────────────
print("Plotting …")
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30,
                       left=0.07, right=0.95, top=0.92, bottom=0.06)

# ===== (a) Cross-attn heatmap: layers × tokens =====
ax_a = fig.add_subplot(gs[0, 0])
im = ax_a.imshow(layer_attn.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.6)
ax_a.set_yticks(range(4))
ax_a.set_yticklabels(TOKEN_NAMES, fontsize=11)
ax_a.set_xticks(range(num_layers))
ax_a.set_xticklabels(range(num_layers))
ax_a.set_xlabel("Decoder Layer", fontsize=12)
ax_a.set_title("(a) Cross-Attention to Each Memory Token", fontsize=13, fontweight="bold")
for li in range(num_layers):
    for ti in range(4):
        v = layer_attn[li, ti]
        color = "white" if v > 0.35 else "black"
        ax_a.text(li, ti, f"{v:.2f}", ha="center", va="center", fontsize=10, color=color)
cb = plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
cb.set_label("Attention weight", fontsize=10)

# ===== (b) Stacked bar chart per layer =====
ax_b = fig.add_subplot(gs[0, 1])
x = np.arange(num_layers)
bottom = np.zeros(num_layers)
for ti in range(4):
    ax_b.bar(x, layer_attn[:, ti], bottom=bottom, color=TOKEN_COLORS[ti],
             label=TOKEN_NAMES[ti].replace("\n", " "), alpha=0.85, edgecolor="white", linewidth=0.5)
    bottom += layer_attn[:, ti]
ax_b.set_xlabel("Decoder Layer", fontsize=12)
ax_b.set_ylabel("Attention Weight", fontsize=12)
ax_b.set_title("(b) Attention Distribution per Layer (stacked)", fontsize=13, fontweight="bold")
ax_b.set_xticks(x)
ax_b.legend(fontsize=10, loc="upper right")
ax_b.set_ylim(0, 1.05)
ax_b.grid(True, alpha=0.3, axis="y")

# ===== (c) Per-instruction attention to instruction token =====
ax_c = fig.add_subplot(gs[1, 0])
INSTR_NAMES = [str(i) for i in range(14)]
colors14 = plt.cm.tab20(np.linspace(0, 1, 14))
for i in range(14):
    ax_c.plot(range(num_layers), instr_attn_per_layer[:, i],
              marker="o", markersize=4, color=colors14[i],
              label=f"{i}", linewidth=1.5, alpha=0.8)
ax_c.set_xlabel("Decoder Layer", fontsize=12)
ax_c.set_ylabel("Attention to Instruction Token", fontsize=12)
ax_c.set_title("(c) Instruction-Token Attention per Task ID\n(spread = model differentiates tasks)",
               fontsize=13, fontweight="bold")
ax_c.set_xticks(range(num_layers))
ax_c.legend(fontsize=8, ncol=7, loc="upper right", title="Instruction ID")
ax_c.grid(True, alpha=0.3)

# ===== (d) Comparison: DP vs ACT instruction fraction =====
ax_d = fig.add_subplot(gs[1, 1])

# DP data (from previous analysis)
dp_obs_attn = [0.3951, 0.4586, 0.4793, 0.3429, 0.3867, 0.3681,
               0.2537, 0.3079, 0.3179, 0.2092, 0.1295, 0.1809]
dp_instr_frac = [v * (64/576) for v in dp_obs_attn]  # instruction is 11.1% of obs token

# ACT instruction attention (direct)
act_instr_attn = layer_attn[:, 3]

# Plot
dp_x = np.arange(12)
act_x = np.arange(num_layers)
ax_d.bar(dp_x - 0.2, dp_instr_frac, 0.35, color="#e74c3c", alpha=0.85,
         label=f"DP (instr share of obs × obs_attn)")
ax_d.bar(act_x + 0.2, act_instr_attn, 0.35, color="#3498db", alpha=0.85,
         label=f"ACT (direct instr token attn)")
ax_d.set_xlabel("Decoder Layer", fontsize=12)
ax_d.set_ylabel("Effective Instruction Attention", fontsize=12)
ax_d.set_title("(d) DP vs ACT: Instruction Influence",
               fontsize=13, fontweight="bold")
max_layers = max(12, num_layers)
ax_d.set_xticks(range(max_layers))
ax_d.legend(fontsize=10)
ax_d.grid(True, alpha=0.3, axis="y")

# Add average annotations
dp_avg = np.mean(dp_instr_frac)
act_avg = np.mean(act_instr_attn)
ax_d.axhline(dp_avg, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=1)
ax_d.axhline(act_avg, color="#3498db", linestyle="--", alpha=0.5, linewidth=1)
ax_d.text(max_layers - 0.5, dp_avg + 0.005, f"DP avg: {dp_avg:.3f}", fontsize=9,
          color="#e74c3c", ha="right")
ax_d.text(max_layers - 0.5, act_avg + 0.005, f"ACT avg: {act_avg:.3f}", fontsize=9,
          color="#3498db", ha="right")

fig.suptitle("ACT Decoder: Per-Modality Cross-Attention Analysis",
             fontsize=16, fontweight="bold", y=0.97)

out_path = "test_code/act_modality_attention.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")

# ─── Print summary ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("ACT Cross-Attention Summary (avg over action queries & batch)")
print("=" * 60)
print(f"{'Layer':>6s}  {'Latent z':>10s}  {'RGB':>10s}  {'Pos':>10s}  {'Instr':>10s}")
print("-" * 52)
for li in range(num_layers):
    print(f"  {li:>3d}    {layer_attn[li,0]:>8.4f}  {layer_attn[li,1]:>8.4f}  {layer_attn[li,2]:>8.4f}  {layer_attn[li,3]:>8.4f}")
print("-" * 52)
avg = layer_attn.mean(axis=0)
print(f"  avg    {avg[0]:>8.4f}  {avg[1]:>8.4f}  {avg[2]:>8.4f}  {avg[3]:>8.4f}")

print(f"\nDP effective instruction attention (avg): {dp_avg:.4f}")
print(f"ACT direct instruction attention (avg):  {act_avg:.4f}")
print(f"ACT / DP ratio: {act_avg/dp_avg:.1f}×")
