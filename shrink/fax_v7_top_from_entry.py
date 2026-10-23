"""FAX family, version 7: the top table T coded against the entry table E as the REFERENCE PAGE (a fax's 2-D mode, JBIG2's
refinement coding: a page coded as its difference from a page the reader already has).
The top table is the entry table seen through the last water level's gain glass and its centring, at 2^-12. So the
reader, holding E and the 768 gains (whole numbers at 2^24, fax_gain24.npy), makes the top page itself:
    s = g x E                        (whole-number products, shift-and-add)
    c = 768 s - (sum of s over the word's 768 lines)          (the centring, no division)
    y = (c + 2^23) >> 24             (back to 768 x 2^16)
    P = (y x 1431655766 + 2^43) >> 44                    (/ 3 / 4096 by a whole-number reciprocal, rounded)
and only R = T - P is stored. R is -1, 0 or +1 and almost always 0: it is lit only where E's rounding at 2^-16 hid
which side of a rounding edge the top value falls. The coder knows where those edges are -- the bits P dropped say how
far the prediction sits from the edge -- so the lit pixels of R are coded with context:
    lit?   : (gain size: bit length of g) x (distance to the rounding edge: bit length of the dropped bits)
    up/hole: (which side of the edge) x (distance)
    size   : |R| >= 2 is possible in general: its bit length in unary (one page per step), then its lower bits as they
             are (so any whole-number R is coded)
Side info: the 768 gains (29 bits each) and the stream lengths. Decoded and checked equal.
    python3 fax_v7_top_from_entry.py [shuffle=1]
  shuffle=1 : control, the gains given to the wrong lines (random order, seed 0): the reference is then wrong
"""
import sys, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from fax_common import *

G = np.load("/Users/abundancemachine/streaming-machine/shrink/fax_gain24.npy")
ONE3 = 1431655766                                     # 2^32 / 3, rounded up

def bitlen(x):                                        # bit length of non-negative whole numbers (a count, by compares)
    x = x.astype(np.int64); b = np.zeros(x.shape, np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        m = (x >> s) > 0; b += m * s; x = np.where(m, x >> s, x)
    return b + (x > 0)

def predict(E, g):
    """the top page as the reader makes it from E and the gains; also the context of every pixel"""
    P = np.empty(E.shape, np.int64); dc = np.empty(E.shape, np.uint32); side = np.empty(E.shape, np.uint32)
    gcls = np.clip(bitlen(g) - 16, 0, 15).astype(np.uint32)
    for r in range(0, E.shape[0], 4096):                          # in slices, to stay small in memory
        s = E[r:r + 4096].astype(np.int64) * g[None, :]
        c = 768 * s - s.sum(1, keepdims=True); y = (c + (1 << 23)) >> 24
        z = y * ONE3 + (1 << 43); P[r:r + 4096] = z >> 44
        f = z & ((1 << 44) - 1); up = f >= (1 << 43)                  # the dropped bits: where the prediction sits
        dist = np.where(up, (1 << 44) - 1 - f, f)                     # how far from the rounding edge
        dc[r:r + 4096] = gcls[None, :] * 45 + bitlen(dist).astype(np.uint32)
        side[r:r + 4096] = up * 45 + bitlen(dist).astype(np.uint32)
    return P, dc, side

def code(T, P, dc, side):
    R = (T.astype(np.int64) - P).ravel(); nz = R != 0
    b1 = rc_encode(nz, dc.ravel(), 16 * 45)
    b2 = rc_encode(R[nz] > 0, side.ravel()[nz], 90)
    m = np.abs(R[nz]); L = bitlen(m); rounds = []
    for j in range(1, int(L.max(initial=1))):                         # how long is |R|? one page per length step
        rounds.append(rc_encode(L[L >= j] > j, np.zeros(int((L >= j).sum()), np.uint32), 1))
    mant = raw_bits(np.concatenate([(m[L - 2 >= p] >> p) & 1 for p in range(int(L.max(initial=1)) - 1)] or [np.zeros(0, np.int64)]))
    return b1, b2, rounds, mant

def decode(b1, b2, rounds, mant, P, dc, side):
    nz = rc_decode(b1, dc.ravel(), 16 * 45).astype(bool)
    pos = rc_decode(b2, side.ravel()[nz], 90).astype(bool)
    n = int(nz.sum()); L = np.ones(n, np.int64)
    for j, b in enumerate(rounds, start=1):
        alive = L >= j; L[np.flatnonzero(alive)[rc_decode(b, np.zeros(int(alive.sum()), np.uint32), 1).astype(bool)]] = j + 1
    m = np.left_shift(np.int64(1), L - 1); u = unraw_bits(mant, int((L - 1).sum())).astype(np.int64); o = 0
    for p in range(int(L.max(initial=1)) - 1):
        w = L - 2 >= p; c = int(w.sum()); m[w] |= u[o:o + c] << p; o += c
    R = np.zeros(P.size, np.int64); R[nz] = np.where(pos, m, -m)
    return (P.ravel() + R).reshape(P.shape)

if __name__ == "__main__":
    opt = dict(x.split("=") for x in sys.argv[1:])
    E = load("E"); T = load("T"); g = G.copy(); tag = ""
    if opt.get("shuffle"): g = np.random.default_rng(0).permutation(g); tag = " CONTROL: gains on random lines"
    t = time.time()
    P, dc, side = predict(E, g); del E
    b1, b2, rounds, mant = code(T, P, dc, side)
    b3 = len(mant) + sum(len(b) for b in rounds)
    nbytes = len(b1) + len(b2) + b3; sidebits = 768 * 29 + (4 + len(rounds)) * 32 + 5
    print(f"     coded ({time.time() - t:.0f} s): lit? {len(b1) * 8 / T.size:.4f}  up/hole {len(b2) * 8 / T.size:.4f}  "
          f"size of the rest {b3 * 8 / T.size:.4f} bits/value", flush=True)
    ok = bool(np.array_equal(decode(b1, b2, rounds, mant, P, dc, side), T))
    res = [report(f"v7 T given E (reference page){tag}", "T|E", nbytes, sidebits, T.size, ok)]
    print(f"     ({time.time() - t:.0f} s)")
    json.dump(res, open(f"/Users/abundancemachine/streaming-machine/shrink/fax_out_v7{'_shuffle' if tag else ''}.json", "w"))
