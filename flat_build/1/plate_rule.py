"""flat_build/1/plate_rule.py -- PLATE ONLY, before any light: the per-place rule (L_b < G x M -> signed digits) at
g = 11 and g = 12 on every proj and mproj plate, to see whether the readers-sized table moves any place between the
up/hole and signed-digit forms. Whole numbers only. Reads base2_gpt2.npz."""
import numpy as np
Z = np.load("/Users/abundancemachine/streaming-machine/base2_gpt2.npz")
def lit_by_place(w):
    r = np.asarray(w, np.int64).copy(); out = []
    for b in range(18):
        odd = (r & 1) == 1; d = np.where(odd, 2 - (r & 3), 0); out.append(int((d != 0).sum())); r = (r - d) >> 1
    assert not r.any(); return out
for kind in ("proj", "mproj", "qkv", "fc"):
    for l in range(12):
        w = Z[f"{kind}.{l}.w"]; N, M = w.shape; L = lit_by_place(w)
        row = []
        for g in ((11, 12) if kind in ("proj", "mproj") else (12,)):
            G = -(-N // g); U = G * M; sp = [p for p in range(16) if L[p] < U]
            row.append(f"g={g} G={G} U={U:,} signed places {sp}")
        print(f"{kind}.{l:<2d} N={N} M={M} L13..15={L[13:16]}  " + " | ".join(row))
