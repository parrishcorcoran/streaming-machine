"""REFERENCE BEAM family, version 2: each word's own share of the common light, on top of version 1's line reference.
  (a) per-word offset: o_w = middle value of the word's residual row (one whole number per word, coded, counted)
        R = A - ref[line] - o_w
  (b) per-word brightness of the reference beam: the word holds the reference pattern ref at its own strength a_w
        a_w = (256 <A_w, ref> + <ref, ref>/2) // <ref, ref>      (whole-number sums, one floor division)
        R = A_w - ((a_w ref + 128) >> 8)                        (a_w is stored; the decoder rebuilds the same light)
  (c) both: brightness, then the offset of what is left
Side information coded for real (range coder) and counted: the 768 reference values, the 50,257 offsets / strengths.
CONTROL: each line shuffled among the words (seed 0), then (b): if the strengths a_w were word structure, the spread of
a_w collapses in the control.
    python3 refbeam_v2_word_offset.py
"""
import time, numpy as np
from refbeam_lib import *

def middle_rows(X): return np.sort(X, axis=1)[:, X.shape[1] // 2 - 1]     # 768 even: the lower middle

def bright(A, ref):
    num = (A * ref[None, :]).sum(1); den = int((ref * ref).sum())
    a = (num * 256 + den // 2) // den
    return a, A - ((a[:, None] * ref[None, :] + 128) >> 8)

for name in ("E", "T"):
    A = load(name); n = A.size; t = time.time()
    ref = np.sort(A, axis=0)[A.shape[0] // 2]; sref = small(ref)
    R1 = A - ref[None, :]
    o = middle_rows(R1); R = R1 - o[:, None]; side = sref + small(o)
    print(f"   {name}: per-word offsets o_w: |o| mean {np.abs(o).mean():.1f}, largest {np.abs(o).max()}")
    for ctx, mant in (("none", False), ("line", True)):
        nb, ok = rc(R, ctx, mant); ok = ok and bool(np.array_equal(R + o[:, None] + ref[None, :], A))
        report("v2", name, f"(a) line ref + word offset, rc {ctx}{'+mant' if mant else ''}", nb, side, n, ok)
    nb, ok = lz(R, "line"); report("v2", name, "(a) line ref + word offset, lzma pages line-major", nb, side, n, ok)
    ph = photons(A, R); print("     photons (a):", ph)
    del R, R1
    a, R = bright(A, ref); side = sref + small(a)
    q = np.percentile(a, [1, 10, 50, 90, 99]).astype(int)
    print(f"   {name}: reference strength a_w (x256): percentiles 1/10/50/90/99 = {q}, spread {a.std():.2f}")
    for ctx, mant in (("none", False), ("line", True)):
        nb, ok = rc(R, ctx, mant)
        ok = ok and bool(np.array_equal(R + ((a[:, None] * ref[None, :] + 128) >> 8), A))
        report("v2", name, f"(b) word brightness of the reference, rc {ctx}{'+mant' if mant else ''}", nb, side, n, ok)
    ph = photons(A, R); print("     photons (b):", ph)
    report("v2", name, "photons (b)", 0, 0, n, None, {"photons": ph, "a_percentiles": q.tolist()})
    o = middle_rows(R); R = R - o[:, None]; side2 = side + small(o)
    nb, ok = rc(R, "line", True)
    ok = ok and bool(np.array_equal(R + o[:, None] + ((a[:, None] * ref[None, :] + 128) >> 8), A))
    report("v2", name, "(c) brightness + word offset, rc line+mant", nb, side2, n, ok)
    del R
    rng = np.random.default_rng(0); S = A.copy()
    for j in range(A.shape[1]): S[:, j] = S[rng.permutation(A.shape[0]), j]
    a2, R = bright(S, ref); side = sref + small(a2)
    print(f"   {name}: CONTROL strength spread {a2.std():.2f} (real {a.std():.2f})")
    nb, ok = rc(R, "line", True)
    report("v2", name, "CONTROL lines shuffled, (b) word brightness, rc line+mant", nb, side + 8, n, ok)
    del S, R, A
    print(f"   ({time.time() - t:.0f} s)")
