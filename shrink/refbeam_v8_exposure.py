"""REFERENCE BEAM family, version 8: each word's own exposure. Version 2 found that words carry the shared light at
different strengths (spread of the strength 41/256 vs 14.5/256 when the lines are shuffled among words: a real
property of the words), but subtracting strength x pattern hurt, because the lines that dominate the pattern are the
ones every word holds exactly. So here the word's strength is used the way a camera uses exposure: not subtracted, but
telling the coder how bright this word's residual runs. Per word, stored once (coded, counted):
    S_w = sum over its 768 lines of |residual|        (adds)
    cls_w = 2 x (bit length of S_w) + (the place just under its top photon)    (half-octave exposure steps)
The coder's odds for the top place and direction are then kept per (line, exposure step).
Prediction (analysis only, float): if a word's residual is its line's light scaled by b_w, the gain per value is
1/2 [log2 mean(b^2) - mean(log2 b^2)], b_w^2 = mean over lines of residual^2 / line variance.
CONTROL: lines shuffled among the words (seed 0): every word's exposure then comes out about the same.
    python3 refbeam_v8_exposure.py
"""
import time, numpy as np
from refbeam_lib import *

def exposure(R):
    S = np.abs(R).sum(1); bl = np.array([int(s).bit_length() for s in S])
    nxt = (S >> np.maximum(bl - 2, 0)) & 1
    c = 2 * bl + nxt; c = c - c.min()
    return c

def predict(R):
    v = R.astype(np.float64).var(0); b2 = ((R.astype(np.float64) ** 2) / v[None, :]).mean(1)
    return float(0.5 * (np.log2(b2.mean()) - np.log2(b2).mean()))

for name in ("E", "T"):
    A = load(name); n = A.size; t = time.time()
    ref = np.sort(A, axis=0)[A.shape[0] // 2]; sref = small(ref)
    rng = np.random.default_rng(0); S = A.copy()
    for j in range(A.shape[1]): S[:, j] = S[rng.permutation(A.shape[0]), j]
    for label, X in (("real", A), ("CONTROL lines shuffled", S)):
        R = X - ref[None, :]
        c = exposure(R); K = int(c.max()) + 1
        print(f"   {name} {label}: prediction {predict(R):+.4f} b/v; exposure steps used {K}, counts {np.bincount(c).tolist()}")
        ctx = (np.arange(768)[None, :] * K + c[:, None]).astype(np.uint32)
        nb0, ok0 = rc(R, "line", False)
        nb, ok = rc(R, ctx, False); side = sref + small(c)
        report("v8", name, f"{label}: line ref, rc line (no exposure)", nb0, sref, n, ok0)
        report("v8", name, f"{label}: line ref + word exposure, rc line x exposure", nb, side, n, ok)
        del ctx, R
    del S, A
    print(f"   ({time.time() - t:.0f} s)")
