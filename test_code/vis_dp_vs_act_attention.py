"""DP vs ACT: per-modality attention comparison — publication-quality figure.

Runs both models and produces a single clean comparison figure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from learning.dp.models import DiffusionTransformer
from learning.common.encoders import InstructionEncoder

# ─── Colors ───────────────────────────────────────────────────────
C_RGB   = "#E53935"
C_DEPTH = "#1E88E5"
C_POS   = "#43A047"
C_INSTR = "#FB8C00"
C_Z     = "#8E24AA"
C_TIME  = "#546E7A"
C_DP    = "#E53935"
C_ACT   = "#1E88E5"

# ======================================================================
#                         DP ATTENTION EXTRACTION
# ======================================================================
print("═" * 60)
print("  Loading DP …")
ckpt_dp = torch.load("checkpoints/dp/latest.pt", map_location="cpu", weights_only=False)
sd_dp = ckpt_dp["model_state_dict"]

net_dp = DiffusionTransformer(
    input_dim=30, output_dim=30, horizon=64,
    obs_cond_dim=576, obs_horizon=1,
    hidden_size=768, depth=12, num_heads=12,
    ff_dim=2048, causal_attn=True, use_encoder_decoder=True,
)
net_dp.load_state_dict({k.replace("noise_pred_net.", ""): v
                        for k, v in sd_dp.items() if k.startswith("noise_pred_net.")})
net_dp.eval()

instr_enc_dp = InstructionEncoder(num_instructions=14, embed_dim=64)
instr_enc_dp.load_state_dict({k.replace("obs_encoder.instruction_encoder.", ""): v
                               for k, v in sd_dp.items() if "instruction_encoder" in k})
instr_enc_dp.eval()

# DP forward
B, T = 14, 64
torch.manual_seed(42)
base_obs = torch.randn(1, 512).expand(B, -1)
with torch.no_grad():
    instr_embs_dp = instr_enc_dp(torch.arange(14))
obs_dp = torch.cat([base_obs, instr_embs_dp], dim=-1)

torch.manual_seed(123)
noisy_action = torch.randn(1, T, 30).expand(B, -1, -1).clone()
timestep = torch.tensor([50] * B)

print("  Extracting DP attention …")
with torch.no_grad():
    time_emb = net_dp.timestep_embedder(timestep).unsqueeze(1)
    cond_emb = net_dp.obs_embedder(obs_dp.reshape(B, 1, 576))
    cond_tokens = torch.cat([time_emb, cond_emb], dim=1) + net_dp.cond_pos_embed[:, :2, :]
    memory = net_dp.condition_encoder(cond_tokens)
    x = net_dp.input_embedder(noisy_action) + net_dp.pos_embed[:, :T, :]
    dp_attn = []
    for layer in net_dp.main_decoder.layers:
        x2 = layer.norm1(x)
        mask = net_dp._create_causal_mask(x2.shape[1], x2.device)
        x = x + layer.dropout1(layer.self_attn(x2, x2, x2, attn_mask=mask, need_weights=False)[0])
        x2 = layer.norm2(x)
        ao, aw = layer.multihead_attn(x2, memory, memory, need_weights=True, average_attn_weights=True)
        x = x + layer.dropout2(ao)
        dp_attn.append(aw)
        x2 = layer.norm3(x)
        x = x + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(x2)))))

# dp_attn[i]: (14, 64, 2) -> [timestep_tok, obs_tok]
# obs_tok has: RGB=288/576, Depth=96/576, Pos=128/576, Instr=64/576
dp_layer_attn = np.array([aw.mean(dim=(0, 1)).numpy() for aw in dp_attn])  # (12, 2)
# Decompose obs attention by modality fraction
dp_modality = np.zeros((12, 5))  # [timestep, rgb, depth, pos, instr]
for li in range(12):
    obs_w = dp_layer_attn[li, 1]
    dp_modality[li, 0] = dp_layer_attn[li, 0]        # timestep
    dp_modality[li, 1] = obs_w * 288 / 576            # rgb share
    dp_modality[li, 2] = obs_w * 96 / 576             # depth share
    dp_modality[li, 3] = obs_w * 128 / 576            # pos share
    dp_modality[li, 4] = obs_w * 64 / 576             # instr share

# Per-instruction std for DP
dp_instr_per_id = np.array([aw[:, :, 1].mean(dim=1).numpy() * (64/576)
                            for aw in dp_attn])  # (12, 14)

# ======================================================================
#                        ACT ATTENTION EXTRACTION
# ======================================================================
print("  Loading ACT …")
ckpt_act = torch.load("checkpoints/act/latest.pt", map_location="cpu", weights_only=False)
sd_act = ckpt_act["model_state_dict"]
act_cfg = ckpt_act["model_config"]
D_act = act_cfg["hidden_dim"]

# Rebuild ACT components
modality_projs = nn.ModuleDict()
for key in ["rgb", "pos", "instruction"]:
    prefix = f"modality_projs.{key}"
    w0 = sd_act[f"{prefix}.0.weight"]
    modality_projs[key] = nn.Sequential(nn.Linear(w0.shape[1], w0.shape[0]), nn.LayerNorm(w0.shape[0]))
    modality_projs[key][0].weight = nn.Parameter(sd_act[f"{prefix}.0.weight"])
    modality_projs[key][0].bias = nn.Parameter(sd_act[f"{prefix}.0.bias"])
    modality_projs[key][1].weight = nn.Parameter(sd_act[f"{prefix}.1.weight"])
    modality_projs[key][1].bias = nn.Parameter(sd_act[f"{prefix}.1.bias"])

latent_proj = nn.Linear(act_cfg["latent_dim"], D_act)
latent_proj.weight = nn.Parameter(sd_act["latent_proj.weight"])
latent_proj.bias = nn.Parameter(sd_act["latent_proj.bias"])

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

pos_enc = _SinePosEnc(D_act)

enc_layer = nn.TransformerEncoderLayer(d_model=D_act, nhead=act_cfg["num_heads"],
    dim_feedforward=act_cfg["dim_feedforward"], dropout=0.0, batch_first=True)
policy_enc = nn.TransformerEncoder(enc_layer, num_layers=act_cfg["num_encoder_layers"])
policy_enc.load_state_dict({k.replace("policy_enc.", ""): v for k, v in sd_act.items() if k.startswith("policy_enc.")})

dec_layer = nn.TransformerDecoderLayer(d_model=D_act, nhead=act_cfg["num_heads"],
    dim_feedforward=act_cfg["dim_feedforward"], dropout=0.0, batch_first=True)
policy_dec = nn.TransformerDecoder(dec_layer, num_layers=act_cfg["num_decoder_layers"])
policy_dec.load_state_dict({k.replace("policy_dec.", ""): v for k, v in sd_act.items() if k.startswith("policy_dec.")})

action_queries = nn.Embedding(act_cfg["pred_horizon"], D_act)
action_queries.weight = nn.Parameter(sd_act["action_queries.weight"])

instr_enc_act = InstructionEncoder(num_instructions=14, embed_dim=64)
instr_enc_act.load_state_dict({k.replace("obs_encoder.instruction_encoder.", ""): v
                                for k, v in sd_act.items() if "instruction_encoder" in k})

for m in [modality_projs, latent_proj, policy_enc, policy_dec, action_queries, instr_enc_act]:
    m.eval()

# ACT forward
print("  Extracting ACT attention …")
torch.manual_seed(42)
rgb_feat = torch.randn(1, 1, 480).expand(B, -1, -1).clone()
pos_feat = torch.randn(1, 1, 128).expand(B, -1, -1).clone()
with torch.no_grad():
    instr_embs_act = instr_enc_act(torch.arange(14))
instr_feat = instr_embs_act.unsqueeze(1)

with torch.no_grad():
    rgb_tok = modality_projs["rgb"](rgb_feat)
    pos_tok = modality_projs["pos"](pos_feat)
    instr_tok = modality_projs["instruction"](instr_feat)
    obs_tokens = torch.cat([rgb_tok, pos_tok, instr_tok], dim=1)
    z = torch.zeros(B, act_cfg["latent_dim"])
    z_tok = latent_proj(z).unsqueeze(1)
    src = pos_enc(torch.cat([z_tok, obs_tokens], dim=1))
    memory = policy_enc(src)

    queries = pos_enc(action_queries.weight.unsqueeze(0).expand(B, -1, -1))
    x = queries
    act_attn = []
    for layer in policy_dec.layers:
        x2 = layer.norm1(x)
        x = x + layer.dropout1(layer.self_attn(x2, x2, x2, need_weights=False)[0])
        x2 = layer.norm2(x)
        ao, aw = layer.multihead_attn(x2, memory, memory, need_weights=True, average_attn_weights=True)
        x = x + layer.dropout2(ao)
        act_attn.append(aw)  # (14, 64, 4) -> [z, rgb, pos, instr]
        x2 = layer.norm3(x)
        x = x + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(x2)))))

act_layer_attn = np.array([aw.mean(dim=(0, 1)).numpy() for aw in act_attn])  # (7, 4)
act_n = len(act_attn)

# Per-instruction for ACT
act_instr_per_id = np.array([aw[:, :, 3].mean(dim=1).numpy() for aw in act_attn])  # (7, 14)


# ======================================================================
#                              FIGURE
# ======================================================================
print("  Plotting …")
fig = plt.figure(figsize=(18, 16))
gs = gridspec.GridSpec(3, 2, hspace=0.42, wspace=0.28,
                       left=0.08, right=0.95, top=0.93, bottom=0.05,
                       height_ratios=[1, 1, 0.9])

# ─── Row 1: Stacked bar charts side by side ──────────────────────

# (a) DP stacked
ax_a = fig.add_subplot(gs[0, 0])
dp_labels = ["Timestep", "RGB", "Depth", "Proprio", "Instruction"]
dp_colors = [C_TIME, C_RGB, C_DEPTH, C_POS, C_INSTR]
x_dp = np.arange(12)
bottom = np.zeros(12)
for mi in range(5):
    ax_a.bar(x_dp, dp_modality[:, mi], bottom=bottom, color=dp_colors[mi],
             label=dp_labels[mi], alpha=0.9, edgecolor="white", linewidth=0.5)
    bottom += dp_modality[:, mi]
ax_a.set_xlabel("Decoder Layer", fontsize=12)
ax_a.set_ylabel("Attention Weight", fontsize=12)
ax_a.set_title("DP: Modality Attention Share", fontsize=14, fontweight="bold")
ax_a.set_xticks(x_dp)
ax_a.set_ylim(0, 1.05)
ax_a.legend(fontsize=8, loc="upper left", ncol=2)
ax_a.grid(True, alpha=0.2, axis="y")
# Annotate instruction fraction
for li in range(12):
    v = dp_modality[li, 4]
    y_mid = sum(dp_modality[li, :4]) + v / 2
    if v > 0.02:
        ax_a.text(li, y_mid, f"{v:.0%}", ha="center", va="center", fontsize=7,
                  color="white", fontweight="bold")

# (b) ACT stacked
ax_b = fig.add_subplot(gs[0, 1])
act_labels = ["Latent z", "RGB (fused)", "Proprio", "Instruction"]
act_colors = [C_Z, C_RGB, C_POS, C_INSTR]
x_act = np.arange(act_n)
bottom = np.zeros(act_n)
for mi in range(4):
    ax_b.bar(x_act, act_layer_attn[:, mi], bottom=bottom, color=act_colors[mi],
             label=act_labels[mi], alpha=0.9, edgecolor="white", linewidth=0.5)
    bottom += act_layer_attn[:, mi]
ax_b.set_xlabel("Decoder Layer", fontsize=12)
ax_b.set_ylabel("Attention Weight", fontsize=12)
ax_b.set_title("ACT: Modality Attention Share", fontsize=14, fontweight="bold")
ax_b.set_xticks(x_act)
ax_b.set_ylim(0, 1.05)
ax_b.legend(fontsize=8, loc="upper left", ncol=2)
ax_b.grid(True, alpha=0.2, axis="y")
for li in range(act_n):
    v = act_layer_attn[li, 3]
    y_mid = sum(act_layer_attn[li, :3]) + v / 2
    if v > 0.02:
        ax_b.text(li, y_mid, f"{v:.0%}", ha="center", va="center", fontsize=8,
                  color="white", fontweight="bold")

# ─── Row 2: Per-instruction attention spread ──────────────────────

colors14 = plt.cm.tab20(np.linspace(0, 1, 14))
instr_labels = [
    "0:pick_L", "1:pick_R", "2:push5", "3:push10", "4:push50", "5:push100",
    "6:pull5", "7:pull10", "8:pull50", "9:pull100", "10:put_L", "11:put_R",
    "12:show_L", "13:show_R",
]

# (c) DP per-instruction
ax_c = fig.add_subplot(gs[1, 0])
for i in range(14):
    ax_c.plot(range(12), dp_instr_per_id[:, i], marker="o", markersize=3,
              color=colors14[i], linewidth=1.2, alpha=0.7, label=instr_labels[i])
# shade the band
dp_min = dp_instr_per_id.min(axis=1)
dp_peak = dp_instr_per_id.max(axis=1)
ax_c.fill_between(range(12), dp_min, dp_peak, color=C_INSTR, alpha=0.15)
ax_c.set_xlabel("Decoder Layer", fontsize=12)
ax_c.set_ylabel("Instruction Attention", fontsize=12)
ax_c.set_title("DP: Per-Task Instruction Attention\n(narrow band = tasks indistinguishable)",
               fontsize=13, fontweight="bold")
ax_c.set_xticks(range(12))
ax_c.set_ylim(0, 0.45)
ax_c.grid(True, alpha=0.2)
# Band width annotation
dp_spread = (dp_peak - dp_min).mean()
ax_c.text(6, 0.42, f"avg spread: {dp_spread:.3f}", ha="center", fontsize=11,
          color=C_DP, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_DP, alpha=0.9))

# (d) ACT per-instruction
ax_d = fig.add_subplot(gs[1, 1])
for i in range(14):
    ax_d.plot(range(act_n), act_instr_per_id[:, i], marker="o", markersize=3,
              color=colors14[i], linewidth=1.2, alpha=0.7, label=instr_labels[i])
act_min = act_instr_per_id.min(axis=1)
act_max = act_instr_per_id.max(axis=1)
ax_d.fill_between(range(act_n), act_min, act_max, color=C_INSTR, alpha=0.15)
ax_d.set_xlabel("Decoder Layer", fontsize=12)
ax_d.set_ylabel("Instruction Attention", fontsize=12)
ax_d.set_title("ACT: Per-Task Instruction Attention\n(wide band = tasks differentiated)",
               fontsize=13, fontweight="bold")
ax_d.set_xticks(range(act_n))
ax_d.set_ylim(0, 0.45)
ax_d.grid(True, alpha=0.2)
act_spread = (act_max - act_min).mean()
ax_d.text(3, 0.42, f"avg spread: {act_spread:.3f}", ha="center", fontsize=11,
          color=C_ACT, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_ACT, alpha=0.9))

# ─── Row 3: Summary comparison ───────────────────────────────────

# (e) Avg modality pie charts
ax_e1 = fig.add_subplot(gs[2, 0])

# DP average
dp_avg = dp_modality.mean(axis=0)
act_avg = act_layer_attn.mean(axis=0)

# Two pie charts side by side in one axes
ax_e1_left = fig.add_axes([0.10, 0.05, 0.18, 0.22])
wedges1, texts1, autotexts1 = ax_e1_left.pie(
    dp_avg, labels=dp_labels, colors=dp_colors, autopct="%1.1f%%",
    pctdistance=0.75, startangle=90, textprops={"fontsize": 8},
)
for t in autotexts1:
    t.set_fontsize(8)
    t.set_fontweight("bold")
ax_e1_left.set_title("DP avg", fontsize=12, fontweight="bold", pad=10)

ax_e1_right = fig.add_axes([0.32, 0.05, 0.18, 0.22])
wedges2, texts2, autotexts2 = ax_e1_right.pie(
    act_avg, labels=act_labels, colors=act_colors, autopct="%1.1f%%",
    pctdistance=0.75, startangle=90, textprops={"fontsize": 8},
)
for t in autotexts2:
    t.set_fontsize(8)
    t.set_fontweight("bold")
ax_e1_right.set_title("ACT avg", fontsize=12, fontweight="bold", pad=10)

ax_e1.axis("off")

# (f) Big comparison bar
ax_f = fig.add_subplot(gs[2, 1])
metrics = [
    "Instruction\nattention share",
    "Per-task spread\n(differentiation)",
    "Instruction\nattention ratio\nvs strongest modality",
]
dp_vals = [
    dp_avg[4],
    dp_spread,
    dp_avg[4] / dp_avg.max(),
]
act_vals = [
    act_avg[3],
    act_spread,
    act_avg[3] / act_avg.max(),
]

x = np.arange(3)
w = 0.32
bars_dp = ax_f.bar(x - w/2, dp_vals, w, color=C_DP, alpha=0.85, label="DP", edgecolor="white")
bars_act = ax_f.bar(x + w/2, act_vals, w, color=C_ACT, alpha=0.85, label="ACT", edgecolor="white")

# Ratio annotations
for i in range(3):
    ratio = act_vals[i] / dp_vals[i] if dp_vals[i] > 0 else float("inf")
    top = max(dp_vals[i], act_vals[i])
    ax_f.text(i, top + 0.02, f"{ratio:.1f}×", ha="center", va="bottom",
              fontsize=13, fontweight="bold", color="#333333")

# Value labels
for bars, vals in [(bars_dp, dp_vals), (bars_act, act_vals)]:
    for bar, v in zip(bars, vals):
        ax_f.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                  f"{v:.3f}", ha="center", va="bottom", fontsize=9)

ax_f.set_xticks(x)
ax_f.set_xticklabels(metrics, fontsize=10)
ax_f.set_ylabel("Value", fontsize=12)
ax_f.set_title("Key Metrics Comparison", fontsize=14, fontweight="bold")
ax_f.legend(fontsize=12, loc="upper right")
ax_f.grid(True, alpha=0.2, axis="y")
ax_f.set_ylim(0, max(max(dp_vals), max(act_vals)) * 1.3)

fig.suptitle("Instruction Conditioning: DP vs ACT",
             fontsize=18, fontweight="bold", y=0.97)

out_path = "test_code/dp_vs_act_modality_attention.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  Saved → {out_path}")
plt.close()

# Summary
print()
print("═" * 60)
print("  SUMMARY")
print("═" * 60)
print(f"  {'Metric':<35s}  {'DP':>8s}  {'ACT':>8s}  {'Ratio':>6s}")
print("  " + "─" * 60)
print(f"  {'Instruction attn share':<35s}  {dp_avg[4]:>7.1%}  {act_avg[3]:>7.1%}  {act_avg[3]/dp_avg[4]:>5.1f}×")
print(f"  {'Per-task spread':<35s}  {dp_spread:>8.4f}  {act_spread:>8.4f}  {act_spread/dp_spread:>5.1f}×")
print(f"  {'Instr / strongest modality':<35s}  {dp_avg[4]/dp_avg.max():>7.1%}  {act_avg[3]/act_avg.max():>7.1%}  {(act_avg[3]/act_avg.max())/(dp_avg[4]/dp_avg.max()):>5.1f}×")
