"""REFERENCE BEAM family, version 3: after the line reference (version 1), an integer Haar lens across each word's
768 lines (lifting: exact in whole numbers). Each output position keeps its own odds in the coder (rc line+mant), so
the low and high bands are coded each with their own statistics.
  stored order, 1 level and 8 levels (768 -> 3 at the bottom)
  random line order (seed 0, a fixed permutation the decoder makes from the seed; lines are free to reorder), 8 levels
  CONTROL: each line shuffled among the words first (keeps every line's values, scrambles which word has which)
Before coding, the Gaussian prediction for one level from the pairs' own spreads s_a, s_b and their correlation r
(analysis only, float): change per value = 1/4 log2( ((s_a^2+s_b^2)^2 - 4 r^2 s_a^2 s_b^2) / (4 s_a^2 s_b^2) ).
It is >= 0 (a loss) unless the pair is correlated more than its spreads differ.
    python3 refbeam_v3_haar_lines.py
"""
import time, numpy as np
from refbeam_lib import *
from refbeam_lift import haar, ihaar

def predict_level1(R):
    a, b = R[:, 0::2].astype(np.float64), R[:, 1::2].astype(np.float64)
    a -= a.mean(0); b -= b.mean(0)
    va, vb = (a * a).mean(0), (b * b).mean(0); r = (a * b).mean(0) / np.sqrt(va * vb)
    ch = 0.25 * np.log2(((va + vb) ** 2 - 4 * r * r * va * vb) / (4 * va * vb))
    return float(ch.mean()), float((r * r).mean()), float(np.abs(r).max())

for name in ("E", "T"):
    A = load(name); n = A.size; t = time.time()
    ref = np.sort(A, axis=0)[A.shape[0] // 2]; side = small(ref); R = A - ref[None, :]
    p, r2, rmax = predict_level1(R)
    print(f"   {name}: level-1 Gaussian prediction {p:+.4f} b/v; adjacent pairs mean r^2 {r2:.5f}, largest |r| {rmax:.3f}")
    perm = np.random.default_rng(0).permutation(768)
    pr, r2r, rmr = predict_level1(R[:, perm])
    print(f"   {name}: random order: prediction {pr:+.4f} b/v; mean r^2 {r2r:.5f}, largest |r| {rmr:.3f}")
    for lv, order in ((1, "stored"), (8, "stored"), (8, "random")):
        X = R if order == "stored" else R[:, perm]
        H = haar(X, lv)
        ok_t = bool(np.array_equal(ihaar(H, lv), X))
        for ctx, mant in (("none", False), ("line", True)):
            nb, ok = rc(H, ctx, mant)
            report("v3", name, f"line ref + Haar across lines, {lv} lvl, {order} order, rc {ctx}{'+mant' if mant else ''}",
                   nb, side + (8 if order == "random" else 0), n, ok and ok_t)
        if lv == 8 and order == "stored":
            nb, ok = lz(H, "line")
            report("v3", name, "line ref + Haar across lines, 8 lvl, stored, lzma line-major", nb, side, n, ok and ok_t)
            o1, o2 = ones(R).mean(), ones(H).mean(); n1, n2 = naf(R).mean(), naf(H).mean()
            print(f"     photons/value: residual {o1:.3f} -> through the lens {o2:.3f}  (single wave {n1:.3f} -> {n2:.3f})")
            report("v3", name, "photons through the Haar lens (8 lvl)", 0, 0, n, None,
                   {"ones_resid": round(float(o1), 4), "ones_lens": round(float(o2), 4), "naf_resid": round(float(n1), 4), "naf_lens": round(float(n2), 4)})
        del H
    rng = np.random.default_rng(0); S = A.copy()
    for j in range(A.shape[1]): S[:, j] = S[rng.permutation(A.shape[0]), j]
    S = S - ref[None, :]; H = haar(S, 8)
    nb, ok = rc(H, "line", True)
    report("v3", name, "CONTROL lines shuffled, Haar 8 lvl stored, rc line+mant", nb, side + 8, n, ok)
    del S, H, R, A
    print(f"   ({time.time() - t:.0f} s)")
