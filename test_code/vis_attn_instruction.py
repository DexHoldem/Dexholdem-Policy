"""Visualize that DP's denoiser cross-attention barely responds to instruction changes.

Generates a multi-panel figure:
  (a) Cross-attn heatmap: layers × [timestep, obs] — obs fades in deeper layers
  (b) Per-instruction obs attention across layers — all 14 lines overlap
  (c) Obs attention sensitivity: instruction-change vs visual-change
  (d) Per-head breakdown at early/mid/late layers
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

from learning.dp.models import DiffusionTransformer
from learning.common.encoders import InstructionEncoder

NAMES = [
    "0:pick_L", "1:pick_R",
    "2:push5", "3:push10", "4:push50", "5:push100",
    "6:pull5", "7:pull10", "8:pull50", "9:pull100",
    "10:put_L", "11:put_R",
    "12:show_L", "13:show_R",
]
SHORT = [str(i) for i in range(14)]

# ─── Load model ───────────────────────────────────────────────────
print("Loading checkpoint …")
ckpt = torch.load("checkpoints/dp/latest.pt", map_location="cpu", weights_only=False)

net = DiffusionTransformer(
    input_dim=30, output_dim=30, horizon=64,
    obs_cond_dim=576, obs_horizon=1,
    hidden_size=768, depth=12, num_heads=12,
    ff_dim=2048, causal_attn=True, use_encoder_decoder=True,
)
sd = {k.replace("noise_pred_net.", ""): v
      for k, v in ckpt["model_state_dict"].items() if k.startswith("noise_pred_net.")}
net.load_state_dict(sd)
net.eval()

instr_enc = InstructionEncoder(num_instructions=14, embed_dim=64)
instr_sd = {k.replace("obs_encoder.instruction_encoder.", ""): v
            for k, v in ckpt["model_state_dict"].items() if "instruction_encoder" in k}
instr_enc.load_state_dict(instr_sd)
instr_enc.eval()


# ─── Helper: run forward, return per-layer cross-attn ─────────────
@torch.no_grad()
def get_cross_attn(obs_features, noisy_action, timestep, per_head=False):
    """Returns list of 12 tensors, each (B, T_act, 2) or (B, heads, T_act, 2)."""
    B, T = noisy_action.shape[:2]
    time_emb = net.timestep_embedder(timestep).unsqueeze(1)
    cond_emb = net.obs_embedder(obs_features.reshape(B, 1, 576))
    cond_tokens = torch.cat([time_emb, cond_emb], dim=1) + net.cond_pos_embed[:, :2, :]
    memory = net.condition_encoder(cond_tokens)

    x = net.input_embedder(noisy_action) + net.pos_embed[:, :T, :]
    layers_attn = []
    for layer in net.main_decoder.layers:
        x2 = layer.norm1(x)
        mask = net._create_causal_mask(x2.shape[1], x2.device)
        x = x + layer.dropout1(layer.self_attn(x2, x2, x2, attn_mask=mask, need_weights=False)[0])
        x2 = layer.norm2(x)
        ao, aw = layer.multihead_attn(
            x2, memory, memory,
            need_weights=True, average_attn_weights=not per_head,
        )
        x = x + layer.dropout2(ao)
        layers_attn.append(aw)
        x2 = layer.norm3(x)
        x = x + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(x2)))))
    return layers_attn


# ─── Build inputs ─────────────────────────────────────────────────
B, T = 14, 64
torch.manual_seed(42)
base_obs = torch.randn(1, 512).expand(B, -1)
with torch.no_grad():
    instr_embs = instr_enc(torch.arange(14))
obs_vary_instr = torch.cat([base_obs, instr_embs], dim=-1)  # same visual, diff instruction

torch.manual_seed(123)
noisy_action = torch.randn(1, T, 30).expand(B, -1, -1).clone()
timestep = torch.tensor([50] * B)

# Also build inputs where we vary VISUAL features instead
torch.manual_seed(99)
varied_visual = torch.randn(B, 512)  # different visual per sample
fixed_instr = instr_embs[0:1].expand(B, -1)  # same instruction
obs_vary_visual = torch.cat([varied_visual, fixed_instr], dim=-1)

print("Running forward passes …")
attn_instr = get_cross_attn(obs_vary_instr, noisy_action, timestep)
attn_visual = get_cross_attn(obs_vary_visual, noisy_action, timestep)
attn_heads = get_cross_attn(obs_vary_instr, noisy_action, timestep, per_head=True)

# ─── Figure ───────────────────────────────────────────────────────
print("Plotting …")
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                       left=0.07, right=0.95, top=0.93, bottom=0.06)

# ---------- (a) Cross-attn heatmap: avg over batch & action tokens ----------
ax_a = fig.add_subplot(gs[0, 0])
# shape per layer: (14, 64, 2) -> avg over batch & action -> (2,) per layer
heatmap = np.array([aw.mean(dim=(0, 1)).numpy() for aw in attn_instr])  # (12, 2)
im = ax_a.imshow(heatmap.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax_a.set_yticks([0, 1])
ax_a.set_yticklabels(["Timestep\ntoken", "Obs token\n(has instr)"], fontsize=11)
ax_a.set_xticks(range(12))
ax_a.set_xticklabels(range(12))
ax_a.set_xlabel("Decoder Layer", fontsize=12)
ax_a.set_title("(a) Cross-Attention Weight Distribution", fontsize=13, fontweight="bold")
# Annotate values
for li in range(12):
    for ti in range(2):
        v = heatmap[li, ti]
        color = "white" if v > 0.55 else "black"
        ax_a.text(li, ti, f"{v:.2f}", ha="center", va="center", fontsize=9, color=color)
cb = plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
cb.set_label("Attention weight", fontsize=10)

# ---------- (b) Per-instruction obs attention across layers ----------
ax_b = fig.add_subplot(gs[0, 1])
colors = plt.cm.tab20(np.linspace(0, 1, 14))
for i in range(14):
    per_layer = [aw[i, :, 1].mean().item() for aw in attn_instr]  # obs attn per layer
    ax_b.plot(range(12), per_layer, marker="o", markersize=4,
              color=colors[i], label=NAMES[i], linewidth=1.5, alpha=0.8)
ax_b.set_xlabel("Decoder Layer", fontsize=12)
ax_b.set_ylabel("Attention to Obs Token", fontsize=12)
ax_b.set_title("(b) Obs Attention per Instruction\n(lines overlap → instruction doesn't matter)",
               fontsize=13, fontweight="bold")
ax_b.set_xticks(range(12))
ax_b.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.8)
ax_b.set_ylim(0, 0.7)
ax_b.grid(True, alpha=0.3)

# ---------- (c) Sensitivity comparison: vary instruction vs vary visual ----------
ax_c = fig.add_subplot(gs[1, 0])
std_instr = [aw[:, :, 1].mean(dim=1).std().item() for aw in attn_instr]
std_visual = [aw[:, :, 1].mean(dim=1).std().item() for aw in attn_visual]

x = np.arange(12)
width = 0.35
bars1 = ax_c.bar(x - width/2, std_instr, width, label="Vary instruction (fix visual)",
                  color="#e74c3c", alpha=0.85)
bars2 = ax_c.bar(x + width/2, std_visual, width, label="Vary visual (fix instruction)",
                  color="#3498db", alpha=0.85)
ax_c.set_xlabel("Decoder Layer", fontsize=12)
ax_c.set_ylabel("Std of Obs Attention\nacross 14 samples", fontsize=12)
ax_c.set_title("(c) Attention Sensitivity: Instruction vs Visual Changes",
               fontsize=13, fontweight="bold")
ax_c.set_xticks(x)
ax_c.legend(fontsize=10, loc="upper left")
ax_c.grid(True, alpha=0.3, axis="y")

# Add ratio text
for li in range(12):
    if std_instr[li] > 0:
        ratio = std_visual[li] / std_instr[li]
        ax_c.text(li, max(std_instr[li], std_visual[li]) + 0.002,
                  f"{ratio:.1f}×", ha="center", va="bottom", fontsize=7, color="#2c3e50")

# ---------- (d) Per-head heatmap at layers 0, 5, 11 ----------
ax_d = fig.add_subplot(gs[1, 1])
target_layers = [0, 5, 11]
head_data = []
for li in target_layers:
    aw = attn_heads[li]  # (B, heads, T_act, 2)
    obs_attn = aw[:, :, :, 1].mean(dim=(0, 2))  # (heads,) avg over batch & action
    head_data.append(obs_attn.numpy())
head_matrix = np.array(head_data)  # (3, 12)

im2 = ax_d.imshow(head_matrix, aspect="auto", cmap="Blues", vmin=0, vmax=0.7)
ax_d.set_yticks(range(3))
ax_d.set_yticklabels([f"Layer {li}" for li in target_layers], fontsize=11)
ax_d.set_xticks(range(12))
ax_d.set_xticklabels([f"H{h}" for h in range(12)], fontsize=9)
ax_d.set_xlabel("Attention Head", fontsize=12)
ax_d.set_title("(d) Per-Head Obs Attention at Key Layers",
               fontsize=13, fontweight="bold")
for li in range(3):
    for h in range(12):
        v = head_matrix[li, h]
        color = "white" if v > 0.45 else "black"
        ax_d.text(h, li, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)
cb2 = plt.colorbar(im2, ax=ax_d, fraction=0.046, pad=0.04)
cb2.set_label("Attention to obs token", fontsize=10)

fig.suptitle("DP Denoiser: Instruction Signal is Invisible in Cross-Attention",
             fontsize=15, fontweight="bold", y=0.97)

out_path = "test_code/dp_instruction_attention.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.close()
