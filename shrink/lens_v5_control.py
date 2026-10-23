"""Shuffle control for LENS v5a (store T, rebuild E through the inverse lens with each word's DC: 30/30).
Same rebuild, but the 768 inverse gains put on randomly permuted lines (seed 0), so every gain value is still used
exactly once -- only the relation between a line and its gain is broken. If the entry cares about the lens, this
should fall well below 30.
"""
import os, sys, json, numpy as np
os.chdir("/Users/abundancemachine/streaming-machine")
sys.path.insert(0, "/Users/abundancemachine/streaming-machine"); sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from table_bits import E, T, TE, words
import lens_lib as LL

K, K2 = 24, 24
N, L = E.shape
G = LL.whole(LL.lens_float("ln_f.weight"), K)
H = np.array([LL.div_round(np.array([1 << (K + K2)], np.int64), int(g))[0] for g in G], dtype=np.int64)
D = np.empty(N, np.int64)
for a in range(0, N, 4096):
    D[a:a + 4096] = LL.rshift_round(LL.div_round((E[a:a + 4096] * G[None, :]).sum(1), 768), K - 4)


def inv_lens(T, D, H):
    out = np.empty(T.shape, np.int64)
    for a in range(0, N, 4096):
        sh = (TE[a:a + 4096] + 20).astype(np.int64)[:, None]
        out[a:a + 4096] = LL.rshift_round(((T[a:a + 4096].astype(np.int64) << sh) + D[a:a + 4096, None]) * H[None, :], K2 + 4)
    return out


Hs = H[np.random.default_rng(0).permutation(L)]
r = {"v5a_control_shuffled_inverse_gains": int(words(entry=inv_lens(T, D, Hs)))}
print(json.dumps(r))
