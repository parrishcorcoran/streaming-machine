"""WAY 2'S STORAGE: EACH LINE'S GAPS ACROSS ITS NEURONS, PACKED, AND EACH LINE'S ORDER AT ITS FLOOR.

The record of way 2 (lineup2/way2.py, Record): for each line of a plate, its weights over all neurons lined up once;
stored are the gaps in rank order (the first the lowest weight itself, then the tiny steps, all >= 0) and the order
(which neuron stands at each rank).
  PACKED GAPS: the gap table (lines x ranks) coded by the same coder recording/C_store.py used, loaded as it is:
    shrink/planes_rc.dylib, its single-wave entry naf_planes (the up and down sheets, place by place from the top),
    with planes version 9's contexts (M_HIGH | M_WEXP | M_LEXP: the higher places of the same value, the line's and
    the rank's exposure meters), which gave the 153.06 MB exact floor of the 48 plates. Decoded back and checked equal,
    and the plate rebuilt from the decoded gaps and the order and checked equal to base2_gpt2.npz.
  THE ORDER AT ITS FLOOR: per line, log2(N! / product over equal weights of count!) in whole numbers (Python
    integers, floor of the log2, as recording/C_store.py): no coder of the order can go below it.
  RAW: the gaps' up and down sheets at 1 bit a place, the order at ceil(log2 N) bits an entry.
  CONTROL (storage shuffle): the plate's values shuffled inside the plate (same values, every pairing broken), then
    each line lined up and its gaps coded the same way.
Against: the 153.1 MB exact floor of the 48 plates (shrink/RESULTS.md), plain 169.87 MB, recording/C's line-up down
each neuron (134.81 MB packed + 93.05 MB order floor = 228.52 MB with powers), and the top table's 54.8 MB alone.
PREDICTION (way2.py's header, written before any run): orders about 8.1 bits a weight at 768 neurons a line and
9.9-10.1 at 2,304 and 3,072, less the ties; packed gaps 3.5-5 bits a weight; the 48 plates 150-175 MB, most likely at or
above the 153.1 MB floor; the top table at or above 54.8 MB.
    python3 lineup2/storage.py      -> lineup2/storage.log, lineup2/storage.json
"""
import sys, os, json, time, math, ctypes, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/lineup2"
LOG = open(f"{OUT}/storage.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
M = np.load(f"{HERE}/base2_gpt2.npz")
BANKS = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]; ALL = BANKS + ["head"]
M_HIGH, M_WEXP, M_LEXP = 1, 32, 64; MODE = M_HIGH | M_WEXP | M_LEXP
LIB = ctypes.CDLL(f"{HERE}/shrink/planes_rc.dylib")
_P, _I, _i = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
LIB.naf_planes.argtypes = [_i, _P, _I, _I, _i, _i, _P, _I, _P]; LIB.naf_planes.restype = _I
SIDE_BITS = 2 * 32 + 5 + 16
def code(A):
    A = np.ascontiguousarray(A, np.int32); R, C = A.shape; nb = int(np.abs(A).max()).bit_length() + 1
    cap = A.size * 3 + (1 << 20); buf = np.zeros(cap, np.uint8); mark = np.zeros(nb + 1, np.int64)
    n = LIB.naf_planes(0, A.ctypes.data, R, C, nb, MODE, buf.ctypes.data, cap, mark.ctypes.data); assert n <= cap
    data = buf[:n].copy(); del buf
    Bk = np.zeros_like(A); m2 = np.zeros(nb + 1, np.int64)
    LIB.naf_planes(1, Bk.ctypes.data, R, C, nb, MODE, data.ctypes.data, n, m2.ctypes.data)
    return int(n) + (SIDE_BITS + 7) // 8, Bk, nb
_F = {}
def fact(n):
    if n not in _F: _F[n] = math.factorial(n)
    return _F[n]
def order_floor_bits(ws):                                    # ws: (L, N) each line's weights in rank order
    L, N = ws.shape; tot = 0; FN = fact(N)
    brk = np.concatenate([np.ones((L, 1), bool), ws[:, 1:] != ws[:, :-1]], 1)
    for i in range(L):
        starts = np.flatnonzero(brk[i]); runs = np.diff(np.append(starts, N)); den = 1
        cs, ms = np.unique(runs[runs > 1], return_counts=True)            # tie counts, and how many runs have each
        for c, m in zip(cs.tolist(), ms.tolist()): den *= fact(c) ** m
        k = FN.bit_length() - den.bit_length()                            # floor(log2(N!/den)), whole numbers:
        if (den << k) > FN: k -= 1                                       # 2^k <= N!/den < 2^(k+1)
        assert (den << k) <= FN < (den << (k + 1))
        tot += k
    return tot
def lined_up(w):
    o = np.argsort(w, axis=1, kind="stable"); ws = np.take_along_axis(w.astype(np.int32), o, 1)
    return o, ws, np.diff(ws, axis=1, prepend=np.int32(0)).astype(np.int32)

rows = {}; acc = dict(weights=0, packed=0, floor_bits=0, raw_sheets_bits=0, raw_order_bits=0, shuffled=0)
for name in ALL:
    w = np.asarray(M[name + ".w"]); L, N = w.shape
    o, ws, g = lined_up(w)
    nbytes, dec, nb = code(g); ok = bool(np.array_equal(dec, g))
    back = np.empty((L, N), np.int64); np.put_along_axis(back, o, np.cumsum(dec, axis=1, dtype=np.int64), 1)
    ok_plate = bool(np.array_equal(back, w)); del dec, back
    fb = order_floor_bits(ws); obits = (N - 1).bit_length(); zg = int((g[:, 1:] == 0).sum()); del o, ws, g
    rng = np.random.default_rng(31 + ALL.index(name))
    sh = rng.permutation(w.reshape(-1)).reshape(L, N); _, _, gsh = lined_up(sh); nsh, dsh, _ = code(gsh)
    ok_sh = bool(np.array_equal(dsh, gsh)); del sh, gsh, dsh
    row = dict(lines=L, neurons=N, weights=L * N, packed_gaps=nbytes, gap_places=nb, roundtrip=ok, plate_rebuilt=ok_plate,
               order_floor_bits=fb, raw_sheets_bits=2 * nb * L * N, raw_order_bits=obits * L * N, shuffled_packed=nsh, shuffled_roundtrip=ok_sh,
               zero_gaps=zg)
    rows[name] = row; assert ok and ok_plate and ok_sh, name
    bw = lambda x: 8 * x / (L * N)
    log(f"  {name:9s} {L}x{N}: packed gaps {bw(nbytes):6.3f} b/w ({nb} places, round trip {ok}, plate rebuilt {ok_plate}) | order floor "
        f"{fb / (L * N):6.3f} b/w (raw {obits}) | total {bw(nbytes) + fb / (L * N):6.3f} b/w | zero gaps {100 * row['zero_gaps'] / (L * (N - 1)):5.1f}% | "
        f"SHUFFLE: gaps {bw(nsh):6.3f} b/w  ({time.time() - t0:.0f}s, {rss()} MB)")
    if name != "head":
        acc["weights"] += L * N; acc["packed"] += nbytes; acc["floor_bits"] += fb; acc["raw_sheets_bits"] += row["raw_sheets_bits"]
        acc["raw_order_bits"] += row["raw_order_bits"]; acc["shuffled"] += nsh
    del w
n = acc["weights"]; MB = lambda x: x / 1e6; POW = 663552                     # the 48 plates' powers (recording/C: 0.664 MB)
tot = acc["packed"] + acc["floor_bits"] / 8 + POW
log(f"\nTHE 48 PLATES ({n:,} weights; plain 16-bit {MB(2 * n):.2f} MB; exact floor shrink/RESULTS.md 153.1 MB)")
log(f"  RAW: gap sheets {MB(acc['raw_sheets_bits'] / 8):.2f} MB ({acc['raw_sheets_bits'] / n:.2f} bits a weight); orders "
    f"{MB(acc['raw_order_bits'] / 8):.2f} MB ({acc['raw_order_bits'] / n:.2f})")
log(f"  PACKED gaps {MB(acc['packed']):.2f} MB ({8 * acc['packed'] / n:.3f} bits a weight) + orders at their floor {MB(acc['floor_bits'] / 8):.2f} MB "
    f"({acc['floor_bits'] / n:.3f} bits a weight) + powers {MB(POW):.3f} MB = {MB(tot):.2f} MB ({8 * tot / n:.3f} bits a weight)")
log(f"  against: the 153.1 MB floor ({MB(tot) - 153.1:+.2f} MB); recording/C's line-up down each neuron 228.52 MB; plain 169.87 MB")
log(f"  SHUFFLE (values shuffled inside each plate, then each line lined up): gaps {MB(acc['shuffled']):.2f} MB ({8 * acc['shuffled'] / n:.3f} "
    f"bits a weight) against {MB(acc['packed']):.2f} MB")
h = rows["head"]; hw = h["weights"]
log(f"\nTHE TOP TABLE ({hw:,} weights; plain 16-bit {MB(2 * hw):.2f} MB; T alone packed 54.8 MB; T given the entry table 0.34 MB)")
log(f"  packed gaps {MB(h['packed_gaps']):.2f} MB ({8 * h['packed_gaps'] / hw:.3f} b/w) + orders at their floor {MB(h['order_floor_bits'] / 8):.2f} MB "
    f"({h['order_floor_bits'] / hw:.3f} b/w) = {MB(h['packed_gaps'] + h['order_floor_bits'] / 8):.2f} MB; SHUFFLE gaps {MB(h['shuffled_packed']):.2f} MB")
json.dump(dict(plates=rows, totals=acc, total_48_MB=MB(tot)), open(f"{OUT}/storage.json", "w"), indent=1)
log(f"saved lineup2/storage.json ({time.time() - t0:.0f}s, max memory {rss()} MB)")
