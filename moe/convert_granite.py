"""QuantumGranite: Granite-3.1-1B-A400M (base) to the all-base-2 machine, the way qwen/convert.py did Qwen3-0.6B.
This is the only place a float exists.
  - Each gauge's gain (RMS, nothing taken out) is folded into the slabs after it: the input gauge into q|k|v; the
    after-arrival gauge into the router and into every expert's gate|up.
  - The score's 1/64 is folded into q's columns. It is a power of two, so q's whole numbers stay the same and only its
    power moves.
  - The 0.22 the machine puts on every write is folded into o and into every expert's down plate.
  - The 12 on the word rows is folded into the entry table. The 1/6 on the top scores is dropped: it does not change
    their order.
  - Weights: 16-bit whole numbers, one power of two per column. Slabs are laid [in][out]; expert plates
    [expert][in][out].
  - Tables: the turns for every position (64-wide heads, theta 1.5e6); the share sigma(x) for the gated bend; the
    arrival e^-d (also the router's share among its 8 lit experts); log2; 2^x.
  - One .npy file a piece in granite_base2/, so the machine reads only the plates a token lights.
    python3 convert_granite.py
"""
import os, glob, math, numpy as np
from safetensors import safe_open
SNAP = glob.glob("/Users/abundancemachine/.cache/huggingface/hub/models--ibm-granite--granite-3.1-1b-a400m-base/snapshots/*/")[0]
F = {}
for f in glob.glob(SNAP + "*.safetensors"):
    h = safe_open(f, "pt")
    for k in h.keys(): F[k] = h
def get(k): return F[k].get_tensor(k).double().numpy()
ES, L, D, HD, NE, I = 16, 24, 1024, 64, 32, 512
EMB_MUL, ATT_MUL, RES_MUL = 12.0, 0.015625, 0.22
OUT = "granite_base2"; os.makedirs(OUT, exist_ok=True)
def save(name, a): np.save(f"{OUT}/{name}.npy", a)
def cols16(W):                                   # [in][out] -> int16 and one power of two per column
    mx = np.maximum(np.abs(W).max(0), 1e-30); e = np.ceil(np.log2(mx / 32767)).astype(np.int64)
    q = np.round(W / 2.0 ** e).astype(np.int64); assert np.abs(q).max() <= 32767; return q.astype(np.int16), e
n = 0
for l in range(L):
    p = f"model.layers.{l}."
    g1, g2 = get(p + "input_layernorm.weight"), get(p + "post_attention_layernorm.weight")
    qkv = np.concatenate([get(p + "self_attn.q_proj.weight").T * ATT_MUL, get(p + "self_attn.k_proj.weight").T, get(p + "self_attn.v_proj.weight").T], 1)
    w, e = cols16(g1[:, None] * qkv); save(f"qkv.{l}.w", w); save(f"qkv.{l}.e", e); n += w.size
    w, e = cols16(get(p + "self_attn.o_proj.weight").T * RES_MUL); save(f"o.{l}.w", w); save(f"o.{l}.e", e); n += w.size
    w, e = cols16(g2[:, None] * get(p + "block_sparse_moe.router.layer.weight").T); save(f"router.{l}.w", w); save(f"router.{l}.e", e); n += w.size
    GU, DN = get(p + "block_sparse_moe.input_linear.weight"), get(p + "block_sparse_moe.output_linear.weight")   # [32][1024 out][1024 in], [32][1024 out][512 in]
    gw, ge, dw, de = np.zeros((NE, D, 2 * I), np.int16), np.zeros((NE, 2 * I), np.int64), np.zeros((NE, I, D), np.int16), np.zeros((NE, D), np.int64)
    for x in range(NE):
        gw[x], ge[x] = cols16(g2[:, None] * GU[x].T)          # columns: the gate's 512, then the up's 512
        dw[x], de[x] = cols16(DN[x].T * RES_MUL)
    save(f"gu.{l}.w", gw); save(f"gu.{l}.e", ge); save(f"down.{l}.w", dw); save(f"down.{l}.e", de); n += gw.size + dw.size
    print(f"layer {l} done", flush=True)
E = get("model.embed_tokens.weight")
w, e = cols16(get("model.norm.weight")[:, None] * E.T); save("head.w", w); save("head.e", e)
emb = np.round(E * EMB_MUL * 2.0 ** ES).astype(np.int64); assert np.abs(emb).max() < 2 ** 31; save("embed", emb.astype(np.int32))
# the turns: pair (i, i+32) of a head turns by pos * theta^(-2i/64); cos and sin at 2^-30
inv = 1500000.0 ** (-np.arange(0, HD, 2) / HD); ang = np.arange(512)[:, None] * inv[None, :]
save("cos", np.round(np.cos(ang) * 2 ** 30).astype(np.int64)); save("sin", np.round(np.sin(ang) * 2 ** 30).astype(np.int64))
# the gated bend's share sigma(x), x in [-16, 16) in steps of 2^-12, share at 2^-16
xs = (np.arange(131072) - 65536) / 4096.0; save("sig", np.round(1 / (1 + np.exp(-xs)) * 65536).astype(np.int64))
SC = 10; arr = []; d = 0
while True:
    a = round(math.exp(-d / 2 ** SC) * 2 ** 24)
    if a == 0: break
    arr.append(a); d += 1
save("arrival", np.array(arr, dtype=np.int64))
save("log2", np.array([round(math.log2(1 + m / 65536) * 65536) for m in range(65536)], dtype=np.int64))
save("exp2", np.array([round(2 ** (f / 65536) * 65536) for f in range(65536)], dtype=np.int64))
save("eps40", np.array(round(1e-6 * 2 ** 40)))
print(f"wrote {OUT}/: {(n + w.size) / 1e6:.1f}M weights (16-bit), entry table {emb.size / 1e6:.1f}M (32-bit)")
