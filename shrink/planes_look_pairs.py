"""PLANES family, look (no coding): does one line of a word say anything about another line of the same word?
Random pairs of lines (seed 0, 4,000 pairs, nothing chosen by value), each line with its dark level taken off.
For each pair: the share of the second line's information the first could remove if it were used perfectly as a
straight-line guide, -1/2 log2(1 - r^2) bits (r = how much the two lines rise and fall together). Floats appear only
in these printed reference numbers. Control: the same pairs with the second line's words shuffled (seed 1).
    python3 planes_look_pairs.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from planes_lib import load
import numpy as np
rng = np.random.default_rng(0)
for tb in ["E", "T"]:
    A = load(tb).astype(np.float64); A -= A.mean(0); A /= A.std(0)
    i = rng.integers(0, A.shape[1], 4000); j = rng.integers(0, A.shape[1], 4000); keep = i != j; i, j = i[keep], j[keep]
    r = np.einsum("ni,ni->i", A[:, i], A[:, j]) / A.shape[0]
    g = -0.5 * np.log2(1 - r ** 2)
    S = A[rng.permutation(A.shape[0])]
    rs = np.einsum("ni,ni->i", A[:, i], S[:, j]) / A.shape[0]; gs = -0.5 * np.log2(1 - rs ** 2)
    print(f"{tb}: |r| median {np.median(np.abs(r)):.4f}, 99th pct {np.percentile(np.abs(r), 99):.3f}, max {np.abs(r).max():.3f};"
          f" bits one random line could save on another: mean {g.mean():.5f}, max {g.max():.4f}")
    print(f"   CONTROL (words of one line shuffled): |r| median {np.median(np.abs(rs)):.4f}, mean bits {gs.mean():.5f}")
    del A, S
