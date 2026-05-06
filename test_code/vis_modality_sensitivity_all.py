"""Per-modality output sensitivity for DP, DP-Light, and ACT.

Zeroes out each modality one at a time and measures output change.
This works for ALL architectures (attention, FiLM, etc).
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
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from learning.dp.models import DiffusionTransformer, ConditionalUnet1D
from learning.common.encoders import InstructionEncoder

C_RGB   = "#E53935"
C_DEPTH = "#1E88E5"
C_POS   = "#43A047"
C_INSTR = "#FB8C00"
C_DP    = "#E53935"
C_DPL   = "#7B1FA2"
C_ACT   = "#1E88E5"
MOD_COLORS = [C_RGB, C_POS, C_INSTR]  # ACT has fused rgbd
MOD_COLORS_DP = [C_RGB, C_DEPTH, C_POS, C_INSTR]

# ======================================================================
#  Load all 3 models
# ======================================================================
print("Loading DP …")
ckpt_dp = torch.load("checkpoints/dp/latest.pt", map_location="cpu", weights_only=False)
sd_dp = ckpt_dp["model_state_dict"]
net_dp = DiffusionTransformer(input_dim=30, output_dim=30, horizon=64,
    obs_cond_dim=576, obs_horizon=1, hidden_size=768, depth=12, num_heads=12,
    ff_dim=2048, causal_attn=True, use_encoder_decoder=True)
net_dp.load_state_dict({k.replace("noise_pred_net.", ""): v for k, v in sd_dp.items() if k.startswith("noise_pred_net.")})
net_dp.eval()
ie_dp = InstructionEncoder(14, 64)
ie_dp.load_state_dict({k.replace("obs_encoder.instruction_encoder.", ""): v for k, v in sd_dp.items() if "instruction_encoder" in k})
ie_dp.eval()

print("Loading DP-Light …")
ckpt_dpl = torch.load("checkpoints/dp_light/latest.pt", map_location="cpu", weights_only=False)
sd_dpl = ckpt_dpl["model_state_dict"]
net_dpl = ConditionalUnet1D(input_dim=30, global_cond_dim=576, diffusion_step_embed_dim=256,
    down_dims=[256, 512, 1024], kernel_size=5, n_groups=8)
net_dpl.load_state_dict({k.replace("noise_pred_net.", ""): v for k, v in sd_dpl.items() if k.startswith("noise_pred_net.")})
net_dpl.eval()
ie_dpl = InstructionEncoder(14, 64)
ie_dpl.load_state_dict({k.replace("obs_encoder.instruction_encoder.", ""): v for k, v in sd_dpl.items() if "instruction_encoder" in k})
ie_dpl.eval()

print("Loading ACT …")
ckpt_act = torch.load("checkpoints/act/latest.pt", map_location="cpu", weights_only=False)
sd_act = ckpt_act["model_state_dict"]
act_cfg = ckpt_act["model_config"]
D_act = act_cfg["hidden_dim"]

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
    def __init__(self, d, ml=512):
        super().__init__()
        pe = torch.zeros(ml, d); p = torch.arange(ml, dtype=torch.float32).unsqueeze(1)
        dv = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(p * dv); pe[:, 1::2] = torch.cos(p * dv)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.shape[1]]

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
action_head = nn.Linear(D_act, act_cfg["action_dim"])
action_head.weight = nn.Parameter(sd_act["action_head.weight"])
action_head.bias = nn.Parameter(sd_act["action_head.bias"])
ie_act = InstructionEncoder(14, 64)
ie_act.load_state_dict({k.replace("obs_encoder.instruction_encoder.", ""): v for k, v in sd_act.items() if "instruction_encoder" in k})
for m in [modality_projs, latent_proj, policy_enc, policy_dec, action_queries, action_head, ie_act]:
    m.eval()


# ======================================================================
#  Perturbation ablation for each modality
# ======================================================================
T_pred = 64
N_SEEDS = 5

# DP obs layout: [rgb(288) | depth(96) | pos(128) | instr(64)] = 576
DP_MODS = {"RGB": (0, 288), "Depth": (288, 384), "Proprio": (384, 512), "Instruction": (512, 576)}
# dp_light same layout: [rgbd(384) | pos(128) | instr(64)] = 576
DPL_MODS = {"RGBD": (0, 384), "Proprio": (384, 512), "Instruction": (512, 576)}

def ablation_study(name, forward_fn, modalities, n_seeds=N_SEEDS):
    """Zero out each modality, measure L2 change in output."""
    results = {mod: [] for mod in modalities}
    for seed in range(n_seeds):
        torch.manual_seed(seed * 100 + 42)
        obs_full = forward_fn("full", seed)
        for mod, (s, e) in modalities.items():
            out_ablated = forward_fn("ablate", seed, s, e)
            delta = (obs_full - out_ablated).norm().item()
            results[mod].append(delta)
    return {mod: np.mean(vals) for mod, vals in results.items()}

# DP
torch.manual_seed(123)
dp_noisy = torch.randn(1, T_pred, 30)
dp_ts = torch.tensor([50])

def dp_fwd(mode, seed, s=0, e=0):
    torch.manual_seed(seed * 100 + 42)
    obs = torch.randn(1, 576)
    instr_emb = ie_dp(torch.tensor([seed % 14]))
    obs[:, 512:576] = instr_emb
    if mode == "ablate":
        obs[:, s:e] = 0.0
    with torch.no_grad():
        return net_dp(dp_noisy, dp_ts, global_cond=obs)

# DP-Light
torch.manual_seed(123)
dpl_noisy = torch.randn(1, T_pred, 30)
dpl_ts = torch.tensor([50])

def dpl_fwd(mode, seed, s=0, e=0):
    torch.manual_seed(seed * 100 + 42)
    obs = torch.randn(1, 576)
    instr_emb = ie_dpl(torch.tensor([seed % 14]))
    obs[:, 512:576] = instr_emb
    if mode == "ablate":
        obs[:, s:e] = 0.0
    with torch.no_grad():
        return net_dpl(dpl_noisy, dpl_ts, global_cond=obs)

# ACT — needs special handling since modalities are separate tokens
def act_ablation(n_seeds=N_SEEDS):
    results = {"RGB": [], "Proprio": [], "Instruction": []}
    for seed in range(n_seeds):
        torch.manual_seed(seed * 100 + 42)
        rgb = torch.randn(1, 1, 480)
        pos = torch.randn(1, 1, 128)
        instr_emb = ie_act(torch.tensor([seed % 14])).unsqueeze(1)  # (1,1,64)

        def act_run(rgb_in, pos_in, instr_in):
            with torch.no_grad():
                rt = modality_projs["rgb"](rgb_in)
                pt = modality_projs["pos"](pos_in)
                it = modality_projs["instruction"](instr_in)
                ot = torch.cat([rt, pt, it], dim=1)
                zt = latent_proj(torch.zeros(1, act_cfg["latent_dim"])).unsqueeze(1)
                src = pos_enc(torch.cat([zt, ot], dim=1))
                mem = policy_enc(src)
                q = pos_enc(action_queries.weight.unsqueeze(0))
                out = policy_dec(q, mem)
                return action_head(out)

        out_full = act_run(rgb, pos, instr_emb)
        out_no_rgb = act_run(torch.zeros_like(rgb), pos, instr_emb)
        out_no_pos = act_run(rgb, torch.zeros_like(pos), instr_emb)
        out_no_ins = act_run(rgb, pos, torch.zeros_like(instr_emb))

        results["RGB"].append((out_full - out_no_rgb).norm().item())
        results["Proprio"].append((out_full - out_no_pos).norm().item())
        results["Instruction"].append((out_full - out_no_ins).norm().item())

    return {mod: np.mean(vals) for mod, vals in results.items()}


print("\nRunning DP ablation …")
dp_results = ablation_study("DP", dp_fwd, DP_MODS)
print("Running DP-Light ablation …")
dpl_results = ablation_study("DP-Light", dpl_fwd, DPL_MODS)
print("Running ACT ablation …")
act_results = act_ablation()

# Normalize to fractions
def to_frac(d):
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}

dp_frac = to_frac(dp_results)
dpl_frac = to_frac(dpl_results)
act_frac = to_frac(act_results)

# Print
print("\n" + "=" * 65)
print("Per-Modality Influence (perturbation ablation)")
print("=" * 65)
for name, frac, raw in [("DP", dp_frac, dp_results),
                         ("DP-Light", dpl_frac, dpl_results),
                         ("ACT", act_frac, act_results)]:
    print(f"\n  {name}:")
    for mod in frac:
        print(f"    {mod:<15s}: {frac[mod]:>6.1%}  (raw Δ = {raw[mod]:.4f})")

# Also measure: output distance when varying ONLY instruction
print("\nMeasuring per-instruction output variation …")
all_model_instr_outputs = {}
for model_name, fwd_fn, ie, make_obs_fn in [
    ("DP", lambda obs: net_dp(dp_noisy, dp_ts, global_cond=obs), ie_dp,
     lambda iid: None),
    ("DP-Light", lambda obs: net_dpl(dpl_noisy, dpl_ts, global_cond=obs), ie_dpl,
     lambda iid: None),
]:
    outputs = []
    torch.manual_seed(42)
    base = torch.randn(1, 512)
    for iid in range(14):
        with torch.no_grad():
            emb = ie(torch.tensor([iid]))
            obs = torch.cat([base, emb], dim=-1)
            out = fwd_fn(obs)
            outputs.append(out.squeeze(0).reshape(-1))
    stacked = torch.stack(outputs)  # (14, T*30)
    all_model_instr_outputs[model_name] = stacked

# ACT
outputs_act = []
torch.manual_seed(42)
rgb_base = torch.randn(1, 1, 480)
pos_base = torch.randn(1, 1, 128)
for iid in range(14):
    with torch.no_grad():
        emb = ie_act(torch.tensor([iid])).unsqueeze(1)
        rt = modality_projs["rgb"](rgb_base)
        pt = modality_projs["pos"](pos_base)
        it = modality_projs["instruction"](emb)
        ot = torch.cat([rt, pt, it], dim=1)
        zt = latent_proj(torch.zeros(1, act_cfg["latent_dim"])).unsqueeze(1)
        src = pos_enc(torch.cat([zt, ot], dim=1))
        mem = policy_enc(src)
        q = pos_enc(action_queries.weight.unsqueeze(0))
        out = policy_dec(q, mem)
        out = action_head(out)
        outputs_act.append(out.squeeze(0).reshape(-1))
all_model_instr_outputs["ACT"] = torch.stack(outputs_act)


# ======================================================================
#  Plot
# ======================================================================
print("Plotting …")
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.32,
                       left=0.06, right=0.96, top=0.92, bottom=0.06)

# ─── Row 1: Modality influence pie/bar for each model ─────────────

for mi, (name, frac, color) in enumerate([
    ("DP (Transformer)", dp_frac, C_DP),
    ("DP-Light (UNet/FiLM)", dpl_frac, C_DPL),
    ("ACT (Per-Token)", act_frac, C_ACT),
]):
    ax = fig.add_subplot(gs[0, mi])
    mods = list(frac.keys())
    vals = list(frac.values())
    mod_colors = []
    for m in mods:
        if "RGB" in m: mod_colors.append(C_RGB)
        elif "Depth" in m: mod_colors.append(C_DEPTH)
        elif "Proprio" in m: mod_colors.append(C_POS)
        else: mod_colors.append(C_INSTR)

    bars = ax.bar(range(len(mods)), vals, color=mod_colors, alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", va="bottom",
                fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(mods)))
    ax.set_xticklabels(mods, fontsize=11)
    ax.set_ylabel("Influence Fraction", fontsize=11)
    ax.set_title(f"{name}", fontsize=13, fontweight="bold", color=color)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(True, alpha=0.2, axis="y")

    # Highlight instruction
    instr_idx = mods.index("Instruction")
    bars[instr_idx].set_edgecolor(C_INSTR)
    bars[instr_idx].set_linewidth(3)

# ─── Row 2 left: Grouped bar comparison ──────────────────────────
ax_comp = fig.add_subplot(gs[1, 0])

# Common modalities: RGB/RGBD, Proprio, Instruction
common = ["Vision", "Proprio", "Instruction"]
dp_common = [dp_frac.get("RGB", 0) + dp_frac.get("Depth", 0),
             dp_frac["Proprio"], dp_frac["Instruction"]]
dpl_common = [dpl_frac["RGBD"], dpl_frac["Proprio"], dpl_frac["Instruction"]]
act_common = [act_frac["RGB"], act_frac["Proprio"], act_frac["Instruction"]]

x = np.arange(3)
w = 0.25
ax_comp.bar(x - w, dp_common, w, color=C_DP, alpha=0.85, label="DP")
ax_comp.bar(x, dpl_common, w, color=C_DPL, alpha=0.85, label="DP-Light")
ax_comp.bar(x + w, act_common, w, color=C_ACT, alpha=0.85, label="ACT")

for i in range(3):
    for j, (vals, off) in enumerate([(dp_common, -w), (dpl_common, 0), (act_common, w)]):
        ax_comp.text(i + off, vals[i] + 0.01, f"{vals[i]:.0%}",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")

ax_comp.set_xticks(x)
ax_comp.set_xticklabels(common, fontsize=12)
ax_comp.set_ylabel("Influence Fraction", fontsize=11)
ax_comp.set_title("Modality Influence Comparison", fontsize=13, fontweight="bold")
ax_comp.legend(fontsize=10)
ax_comp.grid(True, alpha=0.2, axis="y")

# ─── Row 2 middle: Per-instruction output distance heatmaps ──────
ax_heat = fig.add_subplot(gs[1, 1])
# Compute cosine similarity of outputs across instructions
sims = {}
for name, stacked in all_model_instr_outputs.items():
    normed = F.normalize(stacked, dim=-1)
    sim = (normed @ normed.T).numpy()
    sims[name] = sim

# Show all 3 as overlaid info
# Average across-instruction output distance (L2)
model_names = ["DP", "DP-Light", "ACT"]
model_colors_list = [C_DP, C_DPL, C_ACT]
avg_dists = []
for name in model_names:
    stacked = all_model_instr_outputs[name]
    dist = torch.cdist(stacked.unsqueeze(0), stacked.unsqueeze(0)).squeeze(0)
    mask = torch.triu(torch.ones(14, 14), diagonal=1).bool()
    avg_dists.append(dist[mask].mean().item())

bars = ax_heat.bar(range(3), avg_dists, color=model_colors_list, alpha=0.85,
                    edgecolor="black", linewidth=0.5)
for i, v in enumerate(avg_dists):
    ax_heat.text(i, v + 0.05, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=13, fontweight="bold")
ax_heat.set_xticks(range(3))
ax_heat.set_xticklabels(model_names, fontsize=12)
ax_heat.set_ylabel("Mean Output Distance", fontsize=11)
ax_heat.set_title("Output Variation Across\n14 Instructions (same visual input)",
                  fontsize=13, fontweight="bold")
ax_heat.grid(True, alpha=0.2, axis="y")

# ─── Row 2 right: Instruction influence summary ──────────────────
ax_sum = fig.add_subplot(gs[1, 2])

# 3 metrics per model
metrics = ["Modality\nInfluence %", "Output Δ\n(instr change)", "Instr/Vision\nRatio"]
dp_metrics = [dp_frac["Instruction"], avg_dists[0],
              dp_frac["Instruction"] / (dp_frac.get("RGB", 0) + dp_frac.get("Depth", 0))]
dpl_metrics = [dpl_frac["Instruction"], avg_dists[1],
               dpl_frac["Instruction"] / dpl_frac["RGBD"]]
act_metrics = [act_frac["Instruction"], avg_dists[2],
               act_frac["Instruction"] / act_frac["RGB"]]

data = np.array([dp_metrics, dpl_metrics, act_metrics])  # (3, 3)
# Normalize each column for heatmap
data_norm = data / data.max(axis=0, keepdims=True)

im = ax_sum.imshow(data_norm.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax_sum.set_xticks(range(3))
ax_sum.set_xticklabels(model_names, fontsize=12)
ax_sum.set_yticks(range(3))
ax_sum.set_yticklabels(metrics, fontsize=10)
for mi in range(3):
    for gi in range(3):
        v = data[mi, gi]
        vn = data_norm.T[gi, mi]
        color = "white" if vn > 0.6 else "black"
        if gi == 0:
            txt = f"{v:.1%}"
        else:
            txt = f"{v:.3f}"
        ax_sum.text(mi, gi, txt, ha="center", va="center", fontsize=13,
                    fontweight="bold", color=color)
ax_sum.set_title("Instruction Effectiveness\n(green = better)", fontsize=13, fontweight="bold")

fig.suptitle("Per-Modality Influence: DP vs DP-Light vs ACT",
             fontsize=17, fontweight="bold", y=0.97)

out_path = "test_code/modality_sensitivity_all_models.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.close()
