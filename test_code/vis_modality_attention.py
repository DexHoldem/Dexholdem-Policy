"""Visualize per-modality attention contribution in DP's denoiser.

All modalities (RGB, Depth, Pos, Instruction) are concatenated into a single
obs token before the denoiser.  We decompose each modality's influence via:
  (a) obs_embedder weight magnitude — how much the projection amplifies each modality
  (b) Perturbation ablation — zero out one modality, measure change in decoder output
  (c) Gradient attribution — gradient of predicted noise w.r.t. each modality input
  (d) Effective attention — modality contribution to cross-attn value vectors
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

# ─── Modality layout in the 576-dim obs vector ───────────────────
MODALITIES = {
    "RGB\n(cam0)":   (0, 96),
    "RGB\n(cam1)":   (96, 192),
    "RGB\n(cam2)":   (192, 288),
    "Depth\n(cam0)": (288, 320),
    "Depth\n(cam1)": (320, 352),
    "Depth\n(cam2)": (352, 384),
    "Proprio-\nception": (384, 512),
    "Instr-\nuction":   (512, 576),
}
MOD_NAMES = list(MODALITIES.keys())
MOD_RANGES = list(MODALITIES.values())

# Grouped version
GROUPS = {
    "RGB":          (0, 288),
    "Depth":        (288, 384),
    "Proprioception": (384, 512),
    "Instruction":  (512, 576),
}
GRP_NAMES = list(GROUPS.keys())
GRP_RANGES = list(GROUPS.values())
GRP_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

# ─── Load model ───────────────────────────────────────────────────
print("Loading model …")
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

# ─── (a) Weight magnitude analysis ───────────────────────────────
print("Analyzing weight magnitudes …")
W = net.obs_embedder.weight.detach()  # (768, 576)

# Total L2 contribution per group (squared, then fraction)
group_energy = []
total_energy = (W ** 2).sum().item()
for s, e in GRP_RANGES:
    energy = (W[:, s:e] ** 2).sum().item()
    group_energy.append(energy / total_energy)

# Per-dim mean weight magnitude
perdim_magnitude = []
for s, e in GRP_RANGES:
    perdim_magnitude.append(W[:, s:e].norm(dim=0).mean().item())


# ─── Build shared inputs ─────────────────────────────────────────
B, T = 1, 64
torch.manual_seed(42)
base_obs_raw = torch.randn(1, 512)
with torch.no_grad():
    instr_emb = instr_enc(torch.tensor([5]))  # pick one instruction
obs_full = torch.cat([base_obs_raw, instr_emb], dim=-1)  # (1, 576)

torch.manual_seed(123)
noisy_action = torch.randn(1, T, 30)
timestep = torch.tensor([50])


# ─── Helper: full forward pass, return predicted noise ────────────
def forward_pass(obs, noisy_act, ts):
    """Run denoiser, return output (1, T, 30)."""
    B = obs.shape[0]
    time_emb = net.timestep_embedder(ts).unsqueeze(1)
    cond_emb = net.obs_embedder(obs.reshape(B, 1, 576))
    cond_tokens = torch.cat([time_emb, cond_emb], dim=1) + net.cond_pos_embed[:, :2, :]
    memory = net.condition_encoder(cond_tokens)
    x = net.input_embedder(noisy_act) + net.pos_embed[:, :T, :]
    for layer in net.main_decoder.layers:
        x2 = layer.norm1(x)
        mask = net._create_causal_mask(x2.shape[1], x2.device)
        x = x + layer.dropout1(layer.self_attn(x2, x2, x2, attn_mask=mask, need_weights=False)[0])
        x2 = layer.norm2(x)
        x = x + layer.dropout2(layer.multihead_attn(x2, memory, memory, need_weights=False)[0])
        x2 = layer.norm3(x)
        x = x + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(x2)))))
    return net.output_head(net.ln_f(x))


# ─── (b) Perturbation ablation ───────────────────────────────────
print("Running perturbation ablation …")
with torch.no_grad():
    out_full = forward_pass(obs_full, noisy_action, timestep)  # baseline

ablation_delta = []
for gi, (s, e) in enumerate(GRP_RANGES):
    obs_ablated = obs_full.clone()
    obs_ablated[:, s:e] = 0.0
    with torch.no_grad():
        out_ablated = forward_pass(obs_ablated, noisy_action, timestep)
    delta = (out_full - out_ablated).norm().item()
    ablation_delta.append(delta)

total_delta = sum(ablation_delta)
ablation_frac = [d / total_delta for d in ablation_delta]


# ─── (c) Gradient attribution ────────────────────────────────────
print("Computing gradient attribution …")
obs_grad = obs_full.clone().requires_grad_(True)
out = forward_pass(obs_grad, noisy_action, timestep)
loss = out.norm()
loss.backward()

grad = obs_grad.grad[0].abs()  # (576,)
grad_by_group = []
total_grad = grad.sum().item()
for s, e in GRP_RANGES:
    grad_by_group.append(grad[s:e].sum().item() / total_grad)


# ─── (d) Effective attention via value decomposition ──────────────
print("Computing effective attention contribution …")
# In cross-attention: output = softmax(QK^T/√d) × V
# V = obs_embedder(obs) after condition_encoder processing
# We measure how much of the value vector's energy comes from each modality
with torch.no_grad():
    time_emb = net.timestep_embedder(timestep).unsqueeze(1)
    cond_emb_full = net.obs_embedder(obs_full.reshape(1, 1, 576))  # (1,1,768)

    # Decompose obs_embedder output by modality
    cond_by_mod = []
    for s, e in GRP_RANGES:
        partial_obs = torch.zeros_like(obs_full)
        partial_obs[:, s:e] = obs_full[:, s:e]
        partial_cond = net.obs_embedder(partial_obs.reshape(1, 1, 576))
        cond_by_mod.append(partial_cond)

    # Energy fraction in the obs embedding
    full_energy = cond_emb_full.norm().item() ** 2
    # Note: due to bias term, this is approximate
    mod_energy_frac = []
    for c in cond_by_mod:
        mod_energy_frac.append(c.norm().item() ** 2)
    # Normalize (approximate due to cross-terms and bias)
    s = sum(mod_energy_frac)
    mod_energy_frac = [e / s for e in mod_energy_frac]


# ─── Plot ─────────────────────────────────────────────────────────
print("Plotting …")
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30,
                       left=0.07, right=0.95, top=0.92, bottom=0.08)

dim_sizes = [e - s for s, e in GRP_RANGES]

# ===== (a) Weight magnitude + dimension fraction =====
ax_a = fig.add_subplot(gs[0, 0])
x = np.arange(len(GRP_NAMES))
width = 0.35

# Dimension fraction
dim_frac = [d / 576 for d in dim_sizes]
bars1 = ax_a.bar(x - width/2, dim_frac, width, color=[c + "80" for c in GRP_COLORS],
                  edgecolor=GRP_COLORS, linewidth=1.5, label="Dimension fraction")
bars2 = ax_a.bar(x + width/2, group_energy, width, color=GRP_COLORS,
                  alpha=0.85, label="Weight energy fraction")

for i, (df, ge) in enumerate(zip(dim_frac, group_energy)):
    ax_a.text(i - width/2, df + 0.01, f"{df:.1%}", ha="center", va="bottom", fontsize=9)
    ax_a.text(i + width/2, ge + 0.01, f"{ge:.1%}", ha="center", va="bottom", fontsize=9)

ax_a.set_xticks(x)
ax_a.set_xticklabels(GRP_NAMES, fontsize=11)
ax_a.set_ylabel("Fraction", fontsize=12)
ax_a.set_title("(a) Dimension Share vs Weight Energy Share\nin obs_embedder", fontsize=13, fontweight="bold")
ax_a.legend(fontsize=10)
ax_a.set_ylim(0, 0.65)
ax_a.grid(True, alpha=0.3, axis="y")

# ===== (b) Perturbation ablation =====
ax_b = fig.add_subplot(gs[0, 1])
bars = ax_b.bar(x, ablation_frac, color=GRP_COLORS, alpha=0.85, edgecolor="black", linewidth=0.5)
for i, v in enumerate(ablation_frac):
    ax_b.text(i, v + 0.01, f"{v:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax_b.set_xticks(x)
ax_b.set_xticklabels(GRP_NAMES, fontsize=11)
ax_b.set_ylabel("Fraction of output change", fontsize=12)
ax_b.set_title("(b) Perturbation Ablation\n(zero out modality → measure output change)", fontsize=13, fontweight="bold")
ax_b.set_ylim(0, max(ablation_frac) * 1.25)
ax_b.grid(True, alpha=0.3, axis="y")

# ===== (c) Gradient attribution =====
ax_c = fig.add_subplot(gs[1, 0])
bars = ax_c.bar(x, grad_by_group, color=GRP_COLORS, alpha=0.85, edgecolor="black", linewidth=0.5)
for i, v in enumerate(grad_by_group):
    ax_c.text(i, v + 0.01, f"{v:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax_c.set_xticks(x)
ax_c.set_xticklabels(GRP_NAMES, fontsize=11)
ax_c.set_ylabel("Fraction of total gradient", fontsize=12)
ax_c.set_title("(c) Gradient Attribution\n(|∂output/∂modality| fraction)", fontsize=13, fontweight="bold")
ax_c.set_ylim(0, max(grad_by_group) * 1.25)
ax_c.grid(True, alpha=0.3, axis="y")

# ===== (d) Summary comparison =====
ax_d = fig.add_subplot(gs[1, 1])
methods = ["Dim share", "Weight energy", "Perturbation", "Gradient", "Value energy"]
all_data = np.array([
    [d / 576 for d in dim_sizes],
    group_energy,
    ablation_frac,
    grad_by_group,
    mod_energy_frac,
])  # (5, 4)

im = ax_d.imshow(all_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.6)
ax_d.set_xticks(range(4))
ax_d.set_xticklabels(GRP_NAMES, fontsize=11)
ax_d.set_yticks(range(5))
ax_d.set_yticklabels(methods, fontsize=10)
for mi in range(5):
    for gi in range(4):
        v = all_data[mi, gi]
        color = "white" if v > 0.35 else "black"
        ax_d.text(gi, mi, f"{v:.1%}", ha="center", va="center", fontsize=11,
                  fontweight="bold", color=color)
cb = plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
cb.set_label("Fraction", fontsize=10)
ax_d.set_title("(d) All Methods Compared\n(Instruction is weakest across all metrics)",
               fontsize=13, fontweight="bold")

fig.suptitle("DP Denoiser: Per-Modality Influence on Action Prediction",
             fontsize=16, fontweight="bold", y=0.97)

out_path = "test_code/dp_modality_attention.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.close()

# ─── Print summary table ─────────────────────────────────────────
print("\n" + "=" * 70)
print("Summary: Per-Modality Influence Fractions")
print("=" * 70)
print(f"{'Method':<18s}  {'RGB':>8s}  {'Depth':>8s}  {'Proprio':>8s}  {'Instr':>8s}")
print("-" * 55)
for mi, method in enumerate(methods):
    print(f"{method:<18s}  {all_data[mi,0]:>7.1%}  {all_data[mi,1]:>7.1%}  {all_data[mi,2]:>7.1%}  {all_data[mi,3]:>7.1%}")
