"""Check cross-attention maps in DP's trained denoiser.
Measures how much action tokens attend to timestep vs obs token,
and whether changing instruction ID changes the attention pattern."""

import torch
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from learning.dp.models import DiffusionTransformer
from learning.common.encoders import InstructionEncoder

# --- Load model ---
ckpt = torch.load("checkpoints/dp/latest.pt", map_location="cpu", weights_only=False)
sd = {k.replace("noise_pred_net.", ""): v
      for k, v in ckpt["model_state_dict"].items() if k.startswith("noise_pred_net.")}

net = DiffusionTransformer(
    input_dim=30, output_dim=30, horizon=64,
    obs_cond_dim=576, obs_horizon=1,
    hidden_size=768, depth=12, num_heads=12,
    ff_dim=2048, causal_attn=True, use_encoder_decoder=True,
)
net.load_state_dict(sd)
net.eval()

instr_enc = InstructionEncoder(num_instructions=14, embed_dim=64)
instr_sd = {k.replace("obs_encoder.instruction_encoder.", ""): v
            for k, v in ckpt["model_state_dict"].items() if "instruction_encoder" in k}
instr_enc.load_state_dict(instr_sd)
instr_enc.eval()

# --- Build inputs: same obs, different instructions ---
B, T = 14, 64
torch.manual_seed(42)
base_obs = torch.randn(1, 512).expand(B, -1)   # rgb+depth+pos = 512d

with torch.no_grad():
    instr_embs = instr_enc(torch.arange(14))     # (14, 64)
obs_features = torch.cat([base_obs, instr_embs], dim=-1)  # (14, 576)

torch.manual_seed(123)
noisy_action = torch.randn(1, T, 30).expand(B, -1, -1).clone()
timestep = torch.tensor([50] * B)

# --- Forward with attention extraction ---
with torch.no_grad():
    time_emb = net.timestep_embedder(timestep).unsqueeze(1)          # (B,1,768)
    cond_emb = net.obs_embedder(obs_features.reshape(B, 1, 576))     # (B,1,768)
    cond_tokens = torch.cat([time_emb, cond_emb], dim=1)             # (B,2,768)
    cond_tokens = cond_tokens + net.cond_pos_embed[:, :2, :]
    memory = net.condition_encoder(cond_tokens)                       # (B,2,768)

    input_tokens = net.input_embedder(noisy_action) + net.pos_embed[:, :T, :]
    x = input_tokens
    layers_attn = []

    for layer in net.main_decoder.layers:
        # self-attention
        x2 = layer.norm1(x)
        mask = net._create_causal_mask(x2.shape[1], x2.device)
        sa_out = layer.self_attn(x2, x2, x2, attn_mask=mask, need_weights=False)[0]
        x = x + layer.dropout1(sa_out)
        # cross-attention (extract weights)
        x2 = layer.norm2(x)
        ao, aw = layer.multihead_attn(x2, memory, memory,
                                       need_weights=True, average_attn_weights=True)
        x = x + layer.dropout2(ao)
        layers_attn.append(aw)  # (B, 64, 2)
        # FFN
        x2 = layer.norm3(x)
        x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x2))))
        x = x + layer.dropout3(x2)

# --- Print results ---
print("=" * 65)
print("Cross-Attention: action tokens -> [timestep_tok, obs_tok]")
print("=" * 65)
print(f"{'Layer':>6s}  {'timestep':>10s}  {'obs':>10s}")
print("-" * 35)
for li, aw in enumerate(layers_attn):
    m = aw.mean(dim=(0, 1))  # avg over batch & action tokens
    print(f"  {li:2d}      {m[0]:.4f}      {m[1]:.4f}")

print()
print("=" * 65)
print("Obs-token attention: does instruction change the pattern?")
print("=" * 65)
print(f"{'Layer':>6s}  {'mean':>8s}  {'std':>10s}  {'min':>8s}  {'max':>8s}")
print("-" * 50)
for li, aw in enumerate(layers_attn):
    per_instr = aw[:, :, 1].mean(dim=1)  # (14,) avg obs-attn per instruction
    print(f"  {li:2d}    {per_instr.mean():.4f}  {per_instr.std():.8f}  {per_instr.min():.4f}  {per_instr.max():.4f}")

# Also check per-head cross-attention
print()
print("=" * 65)
print("Per-head cross-attention (Layer 0, avg over action tokens)")
print("=" * 65)
# Redo layer 0 cross-attn with per-head weights
with torch.no_grad():
    time_emb = net.timestep_embedder(timestep).unsqueeze(1)
    cond_emb = net.obs_embedder(obs_features.reshape(B, 1, 576))
    cond_tokens = torch.cat([time_emb, cond_emb], dim=1) + net.cond_pos_embed[:, :2, :]
    memory = net.condition_encoder(cond_tokens)
    input_tokens = net.input_embedder(noisy_action) + net.pos_embed[:, :T, :]
    x = input_tokens

    # Just do layer 0
    layer = net.main_decoder.layers[0]
    x2 = layer.norm1(x)
    mask = net._create_causal_mask(x2.shape[1], x2.device)
    x = x + layer.dropout1(layer.self_attn(x2, x2, x2, attn_mask=mask, need_weights=False)[0])
    x2 = layer.norm2(x)
    _, aw_per_head = layer.multihead_attn(x2, memory, memory,
                                           need_weights=True, average_attn_weights=False)
    # aw_per_head: (B, num_heads, 64, 2)
    avg_per_head = aw_per_head.mean(dim=(0, 2))  # (num_heads, 2)
    print(f"{'Head':>6s}  {'timestep':>10s}  {'obs':>10s}")
    print("-" * 30)
    for h in range(avg_per_head.shape[0]):
        print(f"  {h:2d}      {avg_per_head[h,0]:.4f}      {avg_per_head[h,1]:.4f}")
