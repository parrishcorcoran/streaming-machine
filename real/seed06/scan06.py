"""Qwen3-0.6B: is the body's blank in a candidate seed's CPU bf16 normal_ stream at a predicted place?"""
import glob, sys, time, itertools, numpy as np, torch
from safetensors import safe_open
torch.set_num_threads(1)
F = glob.glob("/Users/abundancemachine/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors")[0]
TYPES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
K = 1 << 18
f = safe_open(F, framework="pt")
shape = {t: f.get_slice(f"model.layers.0.{t}.weight").get_shape() for t in TYPES}
size = {t: shape[t][0] * shape[t][1] for t in TYPES}
LS = sum(size.values()); E = 151936 * 1024; NL = 28
def unit(y):
    y = y.astype(np.float64); y = y - y.mean(); return (y / np.linalg.norm(y)).astype(np.float32)
Y = {}                                            # (layer, type, layout) -> normalized first values
for l in range(NL):
    for t in TYPES:
        sl = f.get_slice(f"model.layers.{l}.{t}.weight"); o, i = shape[t]
        Y[(l, t, "stored")] = unit(sl[: K // i].float().numpy().reshape(-1))
        Y[(l, t, "transposed")] = unit(sl[:, : K // o].float().numpy().T.reshape(-1))
inner = {}
for t in TYPES:
    others = [size[u] for u in TYPES if u != t]
    inner[t] = sorted({sum(c) for r in range(7) for c in itertools.combinations(others, r)})
cands = []                                        # (offset, layer, type, order)
for order, b0 in (("layers first", 0), ("after word table", E)):
    for l in range(NL):
        for t in TYPES:
            for s in inner[t]:
                cands.append((b0 + l * LS + s, l, t, order))
CTRL = 16 * 1009 * 7                              # off-grid shift for the check
cands = sorted(cands + [(c[0] + CTRL, c[1], c[2], c[3] + " CHECK") for c in cands])
END = max(c[0] for c in cands) + K
print(f"plates {NL * 7}, candidate places {len(cands)//2} (+ the same number off-grid), stream length {END:,}", flush=True)

SEEDS = [3289271898, 2976995811, 232264057, 0, 1, 7, 42, 100, 123, 999, 1234, 1337, 2023, 2024, 2025, 2718, 3407, 3952, 12345, 31337]
C = 1 << 24
for seed in SEEDS:
    t0 = time.time(); torch.manual_seed(seed)
    buf = np.empty(0, np.float32); bstart = 0; best = {}; ci = 0
    while ci < len(cands):
        P = cands[ci][0]
        if P + K > bstart + len(buf):
            keep = buf[max(0, P - bstart):] if P >= bstart else buf
            nb = max(0, P - bstart); bstart += min(nb, len(buf))
            buf = np.concatenate([keep, torch.empty(C, dtype=torch.bfloat16).normal_(0, 1).float().numpy()])
            continue
        x = buf[P - bstart: P - bstart + K]
        x = x - x.mean(); x = x / np.linalg.norm(x)
        _, l, t, order = cands[ci]
        for lay in ("stored", "transposed"):
            y = Y[(l, t, lay)]; n = len(y)
            z = float(np.dot(x[:n] if n == K else (x[:n] - x[:n].mean()) / np.linalg.norm(x[:n] - x[:n].mean()), y)) * np.sqrt(n)
            key = (order.endswith("CHECK"), order.replace(" CHECK", ""))
            if key not in best or abs(z) > abs(best[key][0]): best[key] = (z, l, t, lay, P)
        ci += 1
    line = " | ".join(f"{o}: largest |z| {abs(best[(False,o)][0]):5.2f} (check {abs(best[(True,o)][0]):5.2f})"
                      for o in ("layers first", "after word table"))
    print(f"seed {seed:>10}: {line}   ({time.time()-t0:.0f}s)", flush=True)
    for o in ("layers first", "after word table"):
        z, l, t, lay, P = best[(False, o)]
        if abs(z) >= 8: print(f"   HIT? seed {seed} {o}: z {z:+.1f} layer {l} {t} {lay} at {P:,}", flush=True)
