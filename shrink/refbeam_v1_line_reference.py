"""REFERENCE BEAM family, version 1: the reference beam = one value per line, common to all 50,257 words.
Stored once (768 whole numbers, coded for real and counted); each word stores only its residual R = A - ref[line].
Three ways to pick the line's value (all whole numbers, no division beyond a floor):
   middle : the median of the line (50,257 is odd, so it is one of the stored values)
   mean   : floor(sum of the line / 50,257)
   common : the most common value on the line (ties: the smallest)
Coded with the same coders as version 0. Decoded: residual decoded, reference added back, compared to the table.
Photons: per value in the word and in the residual, and per word the share of its photons the reference removed.
CONTROL: each line shuffled among the words (seed 0) -- the per-line reference depends only on each line's own values,
so if the gain is the same, the gain is the lines' distributions, not any word structure.
    python3 refbeam_v1_line_reference.py
"""
import time, numpy as np
from refbeam_lib import *

def ref_of(A, how):
    if how == "middle": return np.sort(A, axis=0)[A.shape[0] // 2]
    if how == "mean": return A.sum(0) // A.shape[0]
    out = np.zeros(A.shape[1], np.int64)
    for j in range(A.shape[1]):
        u, c = np.unique(A[:, j], return_counts=True); out[j] = u[np.flatnonzero(c == c.max())[0]]
    return out

for name in ("E", "T"):
    A = load(name); n = A.size; t = time.time()
    for how in ("middle", "mean", "common"):
        ref = ref_of(A, how); side = small(ref); R = A - ref[None, :]
        for ctx, mant in (("none", False), ("line", False), ("line", True)):
            nb, ok = rc(R, ctx, mant)
            ok = ok and bool(np.array_equal(R + ref[None, :], A))
            report("v1", name, f"line reference ({how}), rc {ctx}{'+mant' if mant else ''}", nb, side, n, ok)
        nb, ok = lz(R, "line")
        report("v1", name, f"line reference ({how}), lzma pages line-major", nb, side, n, ok)
        if how == "middle":
            ph = photons(A, R)
            print("     photons:", ph)
            report("v1", name, "photons, line reference (middle)", 0, 0, n, None, {"photons": ph})
            # which lines carry the reference: how many photons per value the reference takes away, line by line
            dl = ones(A).mean(0) - ones(R).mean(0)
            top = np.argsort(-dl)[:8]
            print("     lines losing most photons/value:", [(int(j), round(float(dl[j]), 2)) for j in top],
                  f"; lines losing > 0.5: {(dl > 0.5).sum()}, lines gaining: {(dl < 0).sum()}")
        del R
    rng = np.random.default_rng(0); S = A.copy()
    for j in range(A.shape[1]): S[:, j] = S[rng.permutation(A.shape[0]), j]
    ref = ref_of(S, "middle"); side = small(ref); R = S - ref[None, :]
    nb, ok = rc(R, "line", True)
    report("v1", name, "CONTROL lines shuffled, line reference (middle), rc line+mant", nb, side + 8, n, ok)
    nb, ok = rc(R, "none", False)
    report("v1", name, "CONTROL lines shuffled, line reference (middle), rc none", nb, side + 8, n, ok)
    del S, R, A
    print(f"   ({time.time() - t:.0f} s)")
