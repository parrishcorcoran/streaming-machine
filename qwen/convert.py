"""QuantumQwen3: convert Qwen3-0.6B to the all-base-2 machine. The only place a float exists.
  - each gauge's gain (RMS, nothing taken out) folded into the slab after it
  - weights: 16-bit whole numbers, one power of two per column; slabs laid [in][out]
  - the per-head gauges' gains (with the 1/sqrt(128) of the scores folded in): 16-bit, one power of two
  - tables: the turns (cos, sin) for every position; the share sigma(x) for the gated bend; the arrival e^-d; log2, 2^x
    python3 convert.py
"""
import math, numpy as np, torch
from transformers import AutoModelForCausalLM
sd = {k: v.double().numpy() for k, v in AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).state_dict().items()}
ES, L, D, HD, NQ, NK = 16, 28, 1024, 128, 16, 8
def cols16(W):                                   # [in][out] -> int16 and one power of two per column
    mx = np.maximum(np.abs(W).max(0), 1e-30); e = np.ceil(np.log2(mx / 32767)).astype(np.int64)
    q = np.round(W / 2.0 ** e).astype(np.int64); assert np.abs(q).max() <= 32767; return q.astype(np.int16), e
def vec16(v):                                    # one vector, one power of two
    e = int(np.ceil(np.log2(np.abs(v).max() / 32767))); return np.round(v / 2.0 ** e).astype(np.int64), e
out = {}
for l in range(L):
    p = f"model.layers.{l}."
    g1, g2 = sd[p + "input_layernorm.weight"], sd[p + "post_attention_layernorm.weight"]
    qkv = np.concatenate([sd[p + "self_attn.q_proj.weight"].T, sd[p + "self_attn.k_proj.weight"].T, sd[p + "self_attn.v_proj.weight"].T], 1)
    out[f"qkv.{l}.w"], out[f"qkv.{l}.e"] = cols16(g1[:, None] * qkv)
    out[f"o.{l}.w"], out[f"o.{l}.e"] = cols16(sd[p + "self_attn.o_proj.weight"].T)
    gu = np.concatenate([sd[p + "mlp.gate_proj.weight"].T, sd[p + "mlp.up_proj.weight"].T], 1)
    out[f"gu.{l}.w"], out[f"gu.{l}.e"] = cols16(g2[:, None] * gu)
    out[f"down.{l}.w"], out[f"down.{l}.e"] = cols16(sd[p + "mlp.down_proj.weight"].T)
    out[f"qg.{l}"], out[f"qge.{l}"] = vec16(sd[p + "self_attn.q_norm.weight"] / math.sqrt(HD))     # the score's 1/sqrt(128) lives here
    out[f"kg.{l}"], out[f"kge.{l}"] = vec16(sd[p + "self_attn.k_norm.weight"])
E = sd["model.embed_tokens.weight"]
out["head.w"], out["head.e"] = cols16(sd["model.norm.weight"][:, None] * E.T)
out["embed"] = np.round(E * 2.0 ** ES).astype(np.int64)
# the turns: pair (i, i+64) of a head turns by pos * theta^(-2i/128); cos and sin at 2^-30
inv = 1e6 ** (-np.arange(0, HD, 2) / HD); ang = np.arange(256)[:, None] * inv[None, :]
out["cos"], out["sin"] = np.round(np.cos(ang) * 2 ** 30).astype(np.int64), np.round(np.sin(ang) * 2 ** 30).astype(np.int64)
# the gated bend's share sigma(x), x in [-16, 16) in steps of 2^-12, share at 2^-16
xs = (np.arange(131072) - 65536) / 4096.0; out["sig"] = np.round(1 / (1 + np.exp(-xs)) * 65536).astype(np.int64)
SC = 10; arr = []; d = 0
while True:
    a = round(math.exp(-d / 2 ** SC) * 2 ** 24)
    if a == 0: break
    arr.append(a); d += 1
out["arrival"] = np.array(arr, dtype=np.int64)
out["log2"] = np.array([round(math.log2(1 + m / 65536) * 65536) for m in range(65536)], dtype=np.int64)
out["exp2"] = np.array([round(2 ** (f / 65536) * 65536) for f in range(65536)], dtype=np.int64)
out["eps40"] = np.array(round(1e-6 * 2 ** 40))
np.savez("qwen_base2.npz", **out)
print(f"wrote qwen_base2.npz: {sum(v.size for k, v in out.items() if k.endswith('.w')) / 1e6:.1f}M weights (16-bit)")
