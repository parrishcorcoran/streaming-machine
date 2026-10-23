"""FAX family, versions 3 and 4: single-wave pages (each value written with ups and holes, no two lit side by side,
the fewest lit pixels a value can have). Each place's page: store only its lit pixels (Rice-coded gaps, scan along
the 768 lines of a word) plus one bit per lit pixel: up or hole.
  v3 alone: every page coded on its own
  v4 skip : pages coded from the top place down; a pixel whose place just above is lit is KNOWN dark (no two lit side
            by side), so it is left out of the scan -- the page above is the fax's reference line.
Decoded and checked equal.     python3 fax_v3_waves.py E|T|fc.0
"""
import sys, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from fax_common import *

def code(A, mode):
    S = A.ravel(); neg = S < 0; up, hole = naf_masks(S); del S
    up, hole = np.where(neg, hole, up), np.where(neg, up, hole)          # a negative value: ups and holes swap
    N = up.size; top = int((up | hole).max()).bit_length()
    chunks, side, nbytes = [], 2 * 32 + 5, 0
    above = np.zeros(N, bool)
    for k in range(top - 1, -1, -1):
        u = ((up >> k) & 1).astype(bool); h = ((hole >> k) & 1).astype(bool); lit = u | h
        if mode == "skip":
            keep = ~above; assert not (lit & above).any()
            buf, n, kk, s = gaps_encode(lit[keep]); above = lit
        else:
            buf, n, kk, s = gaps_encode(lit)
        d = raw_bits(h[lit]); chunks.append((buf, n, kk, d)); side += s; nbytes += len(buf) + len(d)
    return (chunks, top, N), nbytes, side

def decode(pkg, shape, mode):
    chunks, top, N = pkg; v = np.zeros(N, np.int32); above = np.zeros(N, bool)
    for i, (buf, n, kk, d) in enumerate(chunks):
        k = top - 1 - i
        if mode == "skip":
            keep = ~above; lit = np.zeros(N, bool); lit[keep] = gaps_decode(buf, n, kk, int(keep.sum())); above = lit
        else: lit = gaps_decode(buf, n, kk, N)
        sgn = np.where(unraw_bits(d, n), -1, 1).astype(np.int32)
        v[lit] += sgn << k
    return v.reshape(shape)

name = sys.argv[1] if len(sys.argv) > 1 else "E"
A = load(name); res = []
for mode, ver in [("alone", "v3"), ("skip", "v4")]:
    t = time.time()
    pkg, nbytes, side = code(A, mode)
    ok = bool(np.array_equal(decode(pkg, A.shape, mode), A)); del pkg
    res.append(report(f"{ver} single-wave pages, {mode}", name, nbytes, side, A.size, ok))
    print(f"     ({time.time() - t:.0f} s)", flush=True)
json.dump(res, open(f"/Users/abundancemachine/streaming-machine/shrink/fax_out_v3_{name}.json", "w"))
