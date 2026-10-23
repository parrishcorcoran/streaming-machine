"""REFERENCE BEAM family, version 4: after the line reference, the integer Walsh-Hadamard lens across each word's 768
lines (the 2-point lifting step on every band at every level: each output is a +/- sum over many lines -- the lens that
mixes most). 2 levels (each output mixes 4 lines) and 8 levels (each output mixes 256 lines). Odds per output position.
Prediction written before coding (analysis only, float): a lens that mixes lines with no correlation makes every output's
spread the AVERAGE of the spreads it mixes, so the cost changes by 1/2 log2( arithmetic mean of the variances / their
geometric mean ) per value -- a loss, unless the lines are correlated. Printed for groups of 4 and of 256 lines.
CONTROL: each line shuffled among the words first.
    python3 refbeam_v4_walsh_lines.py
"""
import time, numpy as np
from refbeam_lib import *
from refbeam_lift import wht, iwht

def predict(R, size):
    v = R.astype(np.float64).var(0)                 # each line's variance
    lv = np.log2(v).reshape(-1)
    # group lines the way the lens does after `levels` levels: consecutive blocks of `size`
    am = v.reshape(-1, size).mean(1); gm = lv.reshape(-1, size).mean(1)
    return float(0.5 * (np.log2(am) - gm).mean())

for name in ("E", "T"):
    A = load(name); n = A.size; t = time.time()
    ref = np.sort(A, axis=0)[A.shape[0] // 2]; side = small(ref); R = A - ref[None, :]
    print(f"   {name}: prediction (no correlation): 2 levels {predict(R, 4):+.4f} b/v, 8 levels {predict(R, 256):+.4f} b/v")
    for lv in (2, 8):
        H = wht(R, lv); ok_t = bool(np.array_equal(iwht(H, lv), R))
        nb, ok = rc(H, "line", True)
        report("v4", name, f"line ref + Walsh-Hadamard across lines, {lv} lvl, rc line+mant", nb, side, n, ok and ok_t)
        if lv == 8:
            nb, ok = rc(H, "none", False)
            report("v4", name, "line ref + Walsh-Hadamard across lines, 8 lvl, rc none", nb, side, n, ok and ok_t)
            nb, ok = lz(H, "line")
            report("v4", name, "line ref + Walsh-Hadamard across lines, 8 lvl, lzma line-major", nb, side, n, ok and ok_t)
            o1, o2 = ones(R).mean(), ones(H).mean(); n1, n2 = naf(R).mean(), naf(H).mean()
            print(f"     photons/value: residual {o1:.3f} -> through the lens {o2:.3f}  (single wave {n1:.3f} -> {n2:.3f})")
            report("v4", name, "photons through the Walsh lens (8 lvl)", 0, 0, n, None,
                   {"ones_resid": round(float(o1), 4), "ones_lens": round(float(o2), 4), "naf_resid": round(float(n1), 4), "naf_lens": round(float(n2), 4)})
            # the all-lines sum (the lens's first output column, x 2^-8 scale) -- the word's own DC
            print(f"     word DC through the lens: spread {H[:, 0].std():.1f}, other outputs median spread {np.median(H.std(0)):.1f}")
        del H
    rng = np.random.default_rng(0); S = A.copy()
    for j in range(A.shape[1]): S[:, j] = S[rng.permutation(A.shape[0]), j]
    S = S - ref[None, :]; H = wht(S, 8)
    nb, ok = rc(H, "line", True)
    report("v4", name, "CONTROL lines shuffled, Walsh 8 lvl, rc line+mant", nb, side + 8, n, ok)
    print(f"     CONTROL word DC spread {H[:, 0].std():.1f}")
    del S, H, R, A
    print(f"   ({time.time() - t:.0f} s)")
