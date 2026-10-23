"""REFERENCE BEAM family, version 10 (the floor, measured as a real coder): after the line reference, each value is
split into its TOP PHOTON's place (k = bit length of |v|) and everything under it. Only the top place goes through the
range coder (odds per line); the direction and every place under the top photon are written as they are, one bit each,
no coder at all. If this costs the same as the full coder (version 1), the places under the top photon are coin flips:
no reference, lens, negative or context can shorten them, and they set the floor.
Also measured: how often a place is lit at depth d under the top photon (0.5 = a coin flip; a lit place and a dark
place then cost the same, so "storing only the 1s" cannot help there).
    python3 refbeam_v10_top_photon_raw.py
"""
import time, numpy as np
from refbeam_lib import *

def pack_fixed(v, w):
    if w == 0 or v.size == 0: return b""
    bits = ((v[:, None] >> np.arange(w - 1, -1, -1)[None, :]) & 1).astype(np.uint8)
    return np.packbits(bits.ravel()).tobytes()

def unpack_fixed(buf, n, w):
    if w == 0 or n == 0: return np.zeros(n, np.int64)
    bits = np.unpackbits(np.frombuffer(buf, np.uint8))[:n * w].reshape(n, w).astype(np.int64)
    return (bits << np.arange(w - 1, -1, -1)[None, :]).sum(1)

def bitlen(x):
    x = x.copy(); b = np.zeros(x.shape, np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        m = (x >> s) > 0; b += m * s; x = np.where(m, x >> s, x)
    return b + (x > 0)

for name in ("E", "T"):
    A = load(name); n = A.size; t = time.time()
    u, c = None, None
    ref = np.sort(A, axis=0)[A.shape[0] // 2]; side = small(ref); R = A - ref[None, :]
    m = np.abs(R).ravel(); k = bitlen(m)
    nk, okk = rc(k.reshape(R.shape), "line", False)
    sb = np.packbits(R.ravel()[m > 0] < 0).tobytes()
    groups = {}; nm = 0
    for j in range(2, int(k.max()) + 1):
        sel = k == j; groups[j] = pack_fixed(m[sel] - (1 << (j - 1)), j - 1); nm += len(groups[j])
    # decode
    back = np.zeros(m.size, np.int64); back[k >= 1] = 1 << (k[k >= 1] - 1)
    for j, b in groups.items():
        sel = k == j; back[sel] += unpack_fixed(b, int(sel.sum()), j - 1)
    lit = back > 0; sg = np.zeros(m.size, bool)
    sg[lit] = np.unpackbits(np.frombuffer(sb, np.uint8))[:int(lit.sum())].astype(bool)
    ok = bool(okk and np.array_equal(np.where(sg, -back, back).reshape(R.shape) + ref[None, :], A))
    report("v10", name, "top place coded, direction + places under it raw", nk + len(sb) + nm, side + 8, n, ok,
           {"top_place_bits": round(8 * nk / n, 4), "direction_bits": round(8 * len(sb) / n, 4),
            "under_top_bits": round(8 * nm / n, 4), "mean_places_under_top": round(float(np.maximum(k - 1, 0).mean()), 4)})
    # how often lit at depth d under the top photon
    rows = []
    for d in range(1, 9):
        sel = k - 1 - d >= 0
        p = float(((m[sel] >> (k[sel] - 1 - d)) & 1).mean()); rows.append(round(p, 4))
    print(f"     lit fraction at depth 1..8 under the top photon: {rows}")
    report("v10", name, "analysis: lit fraction at depth 1..8 under the top photon", 0, 0, n, None, {"lit_fraction": rows})
    del R, A, m, k, back
    print(f"   ({time.time() - t:.0f} s)")
