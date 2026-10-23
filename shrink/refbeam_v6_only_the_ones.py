"""REFERENCE BEAM family, version 6: store ONLY THE 1s of the residual (after the line reference of version 1):
each value is written as how many photons it has (k), and which places they sit in -- nothing about the dark places.
The places are stored as one whole number, the pattern's rank among all patterns with k photons in P places
(combinatorial numbering: the i-th photon from the bottom, at place p, adds C(p, i); decoding walks down the places
and lights place p when the rank is at least C(p, photons still to place)). A table of binomial counts, adds,
subtracts and compares only.
  k     : range coder, odds per line (rc line)
  rank  : written in exactly the bits it needs, ceil(log2 C(P, k)) (all values with the same k share one width)
  sign  : one bit per lit value, as is
Also (analysis only, float logs): the ideal for the places, sum of log2 C(P, k), and the same count for the single-wave
writing (up/hole digits, no two side by side: C(P-k+1, k) 2^k patterns for k photons) -- fewer photons, but each one
also says up or hole.
    python3 refbeam_v6_only_the_ones.py
"""
import time, math, numpy as np
from refbeam_lib import *

def binom_table(P):
    C = np.zeros((P + 2, P + 2), np.int64)
    for n in range(P + 2):
        C[n, 0] = 1
        for r in range(1, n + 1): C[n, r] = C[n - 1, r - 1] + C[n - 1, r]
    return C

def rank_of(m, P, C):
    cnt = np.zeros(m.shape, np.int64); rank = np.zeros(m.shape, np.int64)
    for p in range(P):
        bit = (m >> p) & 1; cnt += bit; rank += bit * C[p, cnt]
    return cnt, rank

def unrank(k, rank, P, C):
    m = np.zeros(k.shape, np.int64); left = k.copy(); r = rank.copy()
    for p in range(P - 1, -1, -1):
        c = C[p, left]; on = (left > 0) & (r >= c)
        m |= on.astype(np.int64) << p; r -= np.where(on, c, 0); left -= on
    return m

def pack_fixed(v, w):
    if w == 0 or v.size == 0: return b""
    bits = ((v[:, None] >> np.arange(w - 1, -1, -1)[None, :]) & 1).astype(np.uint8)
    return np.packbits(bits.ravel()).tobytes()

def unpack_fixed(buf, n, w):
    if w == 0 or n == 0: return np.zeros(n, np.int64)
    bits = np.unpackbits(np.frombuffer(buf, np.uint8))[:n * w].reshape(n, w).astype(np.int64)
    return (bits << np.arange(w - 1, -1, -1)[None, :]).sum(1)

for name in ("E", "T"):
    A = load(name); n = A.size; t = time.time()
    ref = np.sort(A, axis=0)[A.shape[0] // 2]; side = small(ref); R = A - ref[None, :]
    m = np.abs(R).ravel(); P = int(m.max()).bit_length(); C = binom_table(P)
    k, rank = rank_of(m, P, C)
    nk, ok_k = rc(k.reshape(R.shape), "line", False)
    width = np.array([int(C[P, j] - 1).bit_length() for j in range(P + 1)])
    groups = {}; nrank = 0
    for j in range(P + 1):
        sel = k == j; b = pack_fixed(rank[sel], int(width[j])); groups[j] = b; nrank += len(b)
    neg = (R.ravel() < 0)[m > 0]; sb = np.packbits(neg).tobytes()
    # decode
    back_rank = np.zeros(m.size, np.int64)
    for j in range(P + 1):
        sel = k == j; back_rank[sel] = unpack_fixed(groups[j], int(sel.sum()), int(width[j]))
    mm = unrank(k, back_rank, P, C)
    sg = np.zeros(m.size, bool); sg[mm > 0] = np.unpackbits(np.frombuffer(sb, np.uint8))[:int((mm > 0).sum())].astype(bool)
    back = np.where(sg, -mm, mm).reshape(R.shape) + ref[None, :]
    ok = bool(ok_k and np.array_equal(back, A)); del back, mm, back_rank
    tot_side = side + 8 * (P + 2)
    report("v6", name, f"only the 1s: count k (rc line) + places as one rank + signs", nk + nrank + len(sb), tot_side, n, ok,
           {"k_bits_per_value": round(8 * nk / n, 4), "rank_bits_per_value": round(8 * nrank / n, 4),
            "sign_bits_per_value": round(8 * len(sb) / n, 4), "places": P})
    lgC = np.array([math.log2(C[P, j]) for j in range(P + 1)])
    ideal = float(lgC[k].mean())
    kn = naf(R).ravel()
    lgN = np.array([math.log2(math.comb(P + 1 - j + 1, j)) + j if j <= (P + 2) // 2 else 0 for j in range(P + 2)])
    ideal_naf = float(lgN[kn].mean())
    print(f"     photons/value {k.mean():.3f}; ideal bits for their places {ideal:.3f} b/v; rank as stored {8 * nrank / n:.3f} b/v")
    print(f"     single wave: photons/value {kn.mean():.3f}; ideal bits for their places AND directions {ideal_naf:.3f} b/v "
          f"(plain: places {ideal:.3f} + signs {8 * len(sb) / n:.3f})")
    report("v6", name, "analysis: ideal bits for the photons' places (plain / single wave)", 0, 0, n, None,
           {"photons_plain": round(float(k.mean()), 4), "ideal_places_plain": round(ideal, 4),
            "photons_single_wave": round(float(kn.mean()), 4), "ideal_places_and_directions_single_wave": round(ideal_naf, 4)})
    del R, A, m, k, rank, kn
    print(f"   ({time.time() - t:.0f} s)")
