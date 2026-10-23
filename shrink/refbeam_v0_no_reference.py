"""REFERENCE BEAM family, version 0: no reference -- the coders calibrated on the tables as stored.
  rc none      : one set of odds for every value (what a single histogram gives)
  rc line      : odds per line (each line's own spread; no stored side information, the coder learns it)
  rc line+mant : the line's own odds also for the 3 places under the top one
  lz line/word : sign-magnitude bit pages, lzma, the pages written line by line / word by word
  CONTROL      : every line's 50,257 values shuffled among the words (random, seed 0): each line keeps exactly its
                 values, the words are scrambled. A coder that only uses each line's statistics gives the same size.
Floors (analysis only, float logs, not a codec): one histogram for all values; one histogram per line.
    python3 refbeam_v0_no_reference.py
"""
import time, numpy as np
from refbeam_lib import *

def floor_bits(A):
    def H(x):
        _, c = np.unique(x, return_counts=True); p = c / c.sum(); return float(-(c * np.log2(p)).sum())
    one = H(A.ravel()) / A.size
    per = sum(H(A[:, j]) for j in range(A.shape[1])) / A.size
    return round(one, 4), round(per, 4)

for name in ("E", "T"):
    A = load(name); n = A.size; t = time.time()
    f1, fl = floor_bits(A)
    print(f"{name}: floor one histogram {f1} b/v, floor per-line histograms {fl} b/v (histograms themselves not counted)")
    for ctx, mant in (("none", False), ("line", False), ("line", True)):
        nb, ok = rc(A, ctx, mant)
        report("v0", name, f"no reference, rc {ctx}{'+mant' if mant else ''}", nb, 0, n, ok)
    for order in ("line", "word"):
        nb, ok = lz(A, order)
        report("v0", name, f"no reference, lzma pages {order}-major", nb, 0, n, ok)
    rng = np.random.default_rng(0); S = A.copy()
    for j in range(A.shape[1]): S[:, j] = S[rng.permutation(A.shape[0]), j]
    nb, ok = rc(S, "line", False)
    report("v0", name, "CONTROL each line shuffled among words, rc line", nb, 8, n, ok)
    nb, ok = lz(S, "line")
    report("v0", name, "CONTROL each line shuffled among words, lzma line-major", nb, 8, n, ok)
    del S, A
    print(f"   ({time.time() - t:.0f} s)")
