"""Convert GPT-2 to the all-base-2 machine. The only place a float exists: it reads GPT-2's fp32 weights once and
writes every number as a whole number with a power-of-two scale (SPEC.md).

  - each layer norm's DC block (x - mean = x C), gain and offset folded into the bank after it:
        W' = C diag(gain) W        bias' = offset W + bias
  - weights: 16-bit whole numbers, one power of two per column
  - biases, token and position tables: whole numbers at the stream's scale, 2^-16
  - tables: the share past the edge Phi(x); the arrival e^-d; log2 and 2^x of a 16-digit fine part
    python3 base2_convert.py
"""
import math, numpy as np, torch
from transformers import GPT2LMHeadModel

OUT = "base2_gpt2.npz"
ES = 16                                            # the stream's scale is 2^-ES
n = 768
sd = {k: v.detach().double().numpy() for k, v in GPT2LMHeadModel.from_pretrained("gpt2").state_dict().items()}
C = np.eye(n) - 1.0 / n                            # blocking the DC

def cols16(W):
    """16-bit whole numbers, one power of two per column: the smallest that holds the column's largest weight"""
    mx = np.maximum(np.abs(W).max(0), 1e-30)
    e = np.ceil(np.log2(mx / 32767)).astype(np.int64)
    q = np.round(W / 2.0 ** e).astype(np.int64)
    assert np.abs(q).max() <= 32767
    return q.astype(np.int16), e

def stream(v): return np.round(v * 2.0 ** ES).astype(np.int64)

out = {}
for k in range(12):
    p = f"transformer.h.{k}."
    g1, b1, g2, b2 = sd[p + "ln_1.weight"], sd[p + "ln_1.bias"], sd[p + "ln_2.weight"], sd[p + "ln_2.bias"]
    Wq, bq = sd[p + "attn.c_attn.weight"], sd[p + "attn.c_attn.bias"]
    Wf, bf = sd[p + "mlp.c_fc.weight"], sd[p + "mlp.c_fc.bias"]
    for name, W, b in [("qkv", C @ (g1[:, None] * Wq), b1 @ Wq + bq),
                       ("proj", sd[p + "attn.c_proj.weight"], sd[p + "attn.c_proj.bias"]),
                       ("fc", C @ (g2[:, None] * Wf), b2 @ Wf + bf),
                       ("mproj", sd[p + "mlp.c_proj.weight"], sd[p + "mlp.c_proj.bias"])]:
        out[f"{name}.{k}.w"], out[f"{name}.{k}.e"] = cols16(W)
        out[f"{name}.{k}.b"] = stream(b)
wte = sd["transformer.wte.weight"]
gf, bfin = sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"]
out["head.w"], out["head.e"] = cols16(C @ (gf[:, None] * wte.T))
out["head.b"] = stream(bfin @ wte.T)
out["wte"], out["wpe"] = stream(wte), stream(sd["transformer.wpe.weight"])

# tables
out["phi"] = np.array([round(0.5 * (1 + math.erf(((i - 32768) / 4096) / math.sqrt(2))) * 65536) for i in range(65536)], dtype=np.int64)
SC = 10                                            # scores on a scale of 2^-10
arr = []
d = 0
while True:
    a = round(math.exp(-d / 2 ** SC) * 2 ** 24)
    if a == 0: break
    arr.append(a); d += 1
out["arrival"] = np.array(arr, dtype=np.int64)
out["log2"] = np.array([round(math.log2(1 + m / 65536) * 65536) for m in range(65536)], dtype=np.int64)
out["exp2"] = np.array([round(2 ** (f / 65536) * 65536) for f in range(65536)], dtype=np.int64)
out["log2_768"] = np.array(round(math.log2(768) * 65536))
out["eps40"] = np.array(round(1e-5 * 768 * 768 * 2 ** 40))
np.savez(OUT, **out)
print(f"wrote {OUT}: {sum(v.size for k, v in out.items() if k.endswith('.w'))/1e6:.1f}M weights (16-bit), "
      f"arrival table {len(arr):,} entries, phi/log2/exp2 tables 65,536 each")
