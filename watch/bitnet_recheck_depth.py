"""Checker: per depth, the add's top place stays put (real vs another line's stream, Sattolo cycle seed 77) and the
carry places per write photon; also the gate piles exactly on the edge. Lines also taken in a random order (seed 5)."""
import numpy as np
DATA = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/bwatch/"
def pc(v): return np.bitwise_count(v).astype(np.int64)
def blen(v):
    v = np.array(v, np.int64, copy=True)
    for s in (1, 2, 4, 8, 16, 32): v |= v >> s
    return pc(v)
def cycle(rng, m):
    p = np.arange(m)
    for i in range(m - 1, 0, -1):
        j = int(rng.integers(0, i)); p[i], p[j] = p[j], p[i]
    return p
def run(order):
    inv = np.argsort(order); T = np.zeros((60, 8), np.int64); edge = np.zeros(30, np.int64)
    for i in range(30):
        f = np.load(DATA + f"text_{i:02d}.npz"); S = f["states"].astype(np.int64)[:, :, order]; Wr = f["writes"].astype(np.int64)[:, :, order]
        rng = np.random.default_rng(77 + i)
        for w in range(60):
            sa, sb, sh = (int(v) for v in f["shifts"][w]); a, b = S[w] << sa, Wr[w] << sb; P = inv[cycle(rng, 2560)[order]]
            for j, aa in enumerate((a, a[:, P])):
                s = aa + b; lit = (aa != 0) & (s != 0); dT = (blen(np.abs(s)) - blen(np.abs(aa)))[lit]
                chg = np.abs(aa) ^ np.abs(s)
                T[w, 4 * j:4 * j + 4] += [int((dT == 0).sum()), int(lit.sum()), int(pc(chg & ~np.abs(b)).sum()), int(pc(np.abs(b)).sum())]
        dk = np.unpackbits(f["dark"], axis=-1, count=6912).astype(bool); bl = np.unpackbits(f["below"], axis=-1, count=6912).astype(bool)
        edge += (dk & ~bl).sum((1, 2))
    return T, edge
T, edge = run(np.arange(2560)); T2, edge2 = run(np.random.default_rng(5).permutation(2560))
print("random order identical:", np.array_equal(T, T2) and np.array_equal(edge, edge2))
q = lambda a, b: (2000 * a + b) // (2 * b)
for z in range(30):
    row = []
    for w in (2 * z, 2 * z + 1):
        row.append(f"stay {q(T[w,0],T[w,1])}/{q(T[w,4],T[w,5])} ({q(T[w,0],T[w,1]) - q(T[w,4],T[w,5]):+d})  carry/write {q(T[w,2],T[w,3])}/{q(T[w,6],T[w,7])}")
    print(f"depth {z:2d}: look {row[0]} | neuron {row[1]} | on the edge {edge[z]}")
