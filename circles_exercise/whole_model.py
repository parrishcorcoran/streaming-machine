"""Whole Qwen3-0.6B: every row's length in one shared file; every weight's tilt off 90 degrees written as a whole
number of one shared piece of pi (pi / 2^b) for the whole model; tilts packed one table per matrix.
Against bf16 and 4-bit (Q4_0). Reports the size and how far each row turns."""
import glob, math, time, numpy as np, torch
from safetensors import safe_open
F = glob.glob("/Users/abundancemachine/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors")[0]
BS = (9, 10, 11)
def H_bits(hist):
    c = hist[hist > 0].astype(np.float64); p = c / c.sum(); return float(-(c * np.log2(p)).sum())
def kind(n):
    for k in ("embed", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if k in n: return k
tot = dict(rows=0, weights=0, other_bytes=0); packed = {b: 0.0 for b in BS}; q4bits = 0.0
turns = {}; t0 = time.time()
with safe_open(F, framework="pt") as f:
    for name in f.keys():
        if name == "lm_head.weight": continue          # the same table as embed_tokens (tied), stored once
        t = f.get_slice(name); shp = t.get_shape()
        if len(shp) != 2: tot["other_bytes"] += 2 * math.prod(shp); continue
        R, N = shp; k = kind(name); tot["rows"] += R; tot["weights"] += R * N
        hist = {b: {} for b in BS}; q4hist = np.zeros(16, np.int64)
        for r0 in range(0, R, 4096):
            X = t[r0:r0+4096].float().numpy().astype(np.float64)
            L = np.linalg.norm(X, axis=1, keepdims=True); C = X / L                       # cosines (tilt ~ cosine near 90)
            tilt = np.pi / 2 - np.arccos(C)                                               # exact tilt off 90 degrees
            for b in BS:
                s = np.pi / 2 ** b
                code = np.round(tilt / s).astype(np.int64)
                rb = np.sin(code * s)                                                     # the cosine back from the tilt
                nr = np.linalg.norm(rb, axis=1)
                cos = np.where(nr > 0, np.sum(C * rb, 1) / np.where(nr > 0, nr, 1), 0)
                turns.setdefault((k, b), []).append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
                u, c = np.unique(code, return_counts=True)
                for a, n in zip(u.tolist(), c.tolist()): hist[b][a] = hist[b].get(a, 0) + n
            # 4-bit Q4_0, blocks of 32 along the row
            Xf = X.astype(np.float32).reshape(-1, 32); j = np.argmax(np.abs(Xf), 1)
            d = Xf[np.arange(len(Xf)), j] / np.float32(-8)
            idd = np.where(d != 0, np.float32(1) / np.where(d != 0, d, 1), 0).astype(np.float32)
            xi = np.minimum(15, np.trunc(Xf * idd[:, None] + np.float32(8.5)).astype(np.int64))
            q4hist += np.bincount(xi.ravel(), minlength=16)
            Q = ((xi - 8).astype(np.float32) * np.float16(d).astype(np.float32)[:, None]).reshape(X.shape).astype(np.float64)
            nq = np.linalg.norm(Q, axis=1)
            cq = np.where(nq > 0, np.sum(X * Q, 1) / (L[:, 0] * np.where(nq > 0, nq, 1)), 0)
            turns.setdefault((k, "4-bit"), []).append(np.degrees(np.arccos(np.clip(cq, -1, 1))))
        for b in BS: packed[b] += H_bits(np.array(list(hist[b].values())))
        q4bits += H_bits(q4hist) + 16 * (R * N // 32)
        print(f"{name:45s} {R:>7d} x {N:<5d} {time.time()-t0:6.1f}s", flush=True)
W = tot["weights"]; MB = 1e6
print(f"\nrows {tot['rows']:,}   weights {W:,}   small 1-D pieces (norms) {tot['other_bytes']/MB:.2f} MB kept as they are")
print(f"lengths file: {tot['rows']:,} lengths x 32 bits = {tot['rows']*4/MB:.2f} MB  ({32*tot['rows']/W:.3f} bits a weight)")
print(f"bf16: {2*W/MB:.1f} MB   4-bit (Q4_0) packed: {q4bits/8/MB:.1f} MB ({q4bits/W:.2f} bits a weight; written 4.5 = {4.5*W/8/MB:.1f} MB)")
for b in BS:
    print(f"shared piece pi/2^{b}: tilts packed {packed[b]/8/MB:.1f} MB ({packed[b]/W:.2f} bits a weight) + lengths "
          f"{tot['rows']*4/MB:.2f} MB")
print("\nturn of each row (mean / worst), degrees")
ks = sorted({k for k, _ in turns}, key=["embed","q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"].index)
print(f"{'':10s}" + "".join(f"{('pi/2^'+str(b)) if b != '4-bit' else '4-bit':>16s}" for b in (*BS, "4-bit")))
for k in ks:
    print(f"{k:10s}" + "".join(f"{np.concatenate(turns[(k,b)]).mean():8.2f}/{np.concatenate(turns[(k,b)]).max():<7.2f}" for b in (*BS, "4-bit")))
