"""LENS v3: v1 (store E, recompute T through the lens) but the residual coded knowing WHERE it can be lit.

The decoder has the whole number X before T^ is rounded, so it knows how far each place sits from a rounding edge:
   u = X - (T^ << s)  in [-half, half),   d = half - |u|  (distance to the nearest edge),  side = +1 if u >= 0 else -1.
E's own rounding moves X by at most G_i/2, and the word's DC estimate moves it by a little more (rms Q, from the lens:
Q = isqrt(sum G^2 / 12) / 768). So a place far from the edge cannot flip, and a flip can only go toward the near edge.
The residual becomes one yes/no per place: "did it cross the near edge?" Coded with the whole-number range coder
(lens_rc.py), the context = how far from the edge in units of (G_i + 4Q)/32, and whether the line's gain is tiny.
Anything else (a flip away from the near edge, or by 2) goes in an exception list.
Predicted before running: about 2a = gf_i/16 of places sit inside the zone where a flip is possible (~8%), with flip
chance falling from 1/2 at the edge to 0 at G_i/2 -> ideal ~ 0.07 bits/place -> ~0.35 MB, vs 1.36 MB for v1's lzma.
"""
import sys, time, json, math, lzma, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from lens_lib import *
import lens_rc

K, NB = 24, 48
t0 = time.time()
E, T, TE, HB, WPE = load()
N, L = E.shape
G = whole(lens_float("ln_f.weight"), K)
Q = math.isqrt(int((G.astype(object) ** 2).sum()) // 12) // 768
LOW = (G < 8 * Q).astype(np.int64)                       # lines where the DC error outweighs the line's own rounding
SCALE = G + 4 * Q
print(f"Q = {Q} ({Q / 2**K:.5f} of a gain unit); low-gain lines: {LOW.sum()}", flush=True)


def lens_edges(E, G, TE):
    """T^, plus for every place: side (+1/-1) and context (distance-to-edge bin, low-gain line)"""
    That = np.empty(E.shape, np.int16); side = np.empty(E.shape, np.int8); ctx = np.empty(E.shape, np.int16)
    for a in range(0, E.shape[0], 4096):
        Y = E[a:a + 4096] * G[None, :]
        X = Y - div_round(Y.sum(1), 768)[:, None]
        s = (K + 16 + TE[a:a + 4096]).astype(np.int64)[:, None]
        half = np.int64(1) << (s - 1)
        th = (X + half) >> s
        u = X - (th << s)
        d = half - np.abs(u)
        That[a:a + 4096] = np.clip(th, -32767, 32767)
        side[a:a + 4096] = np.where(u >= 0, 1, -1)
        ctx[a:a + 4096] = np.minimum(NB - 1, (d << 5) // SCALE[None, :]) + NB * LOW[None, :]
    return That, side, ctx


That, side, ctx = lens_edges(E, G, TE)
R = T.astype(np.int64) - That
flip = (R == side)
exc = (R != 0) & ~flip
print(f"residual nonzero {np.mean(R != 0)*100:.3f}%  toward the near edge {flip.sum():,}  exceptions {exc.sum():,} [{time.time()-t0:.0f}s]", flush=True)

# coding order: line-major (each line one scan line across all words)
bits = np.ascontiguousarray(flip.T).ravel().astype(np.uint8)
cx = np.ascontiguousarray(ctx.T).ravel()
# reference floor: each context's own flip rate, static
H = 0.0
print("context (distance bin, low line): places, flip rate")
for c in np.unique(cx):
    m = cx == c; n = int(m.sum()); k = int(bits[m].sum())
    if 0 < k < n: q = k / n; H += n * -(q * math.log2(q) + (1 - q) * math.log2(1 - q))
    if c % 4 == 0 and (c < NB and c < 28 or c >= NB and c < NB + 28): print(f"   bin {c % NB:2d} low {c // NB}: {n:10,d}  {k / max(n, 1) * 100:7.3f}%")
print(f"ideal (static per-context entropy): {H / 8 / 1e6:.4f} MB  [{time.time()-t0:.0f}s]", flush=True)
data = lens_rc.encode(bits.tolist(), cx.tolist(), 2 * NB)
ei = np.flatnonzero(np.ascontiguousarray(exc.T).ravel())
ev = np.ascontiguousarray(R.T).ravel()[ei]
blob_exc = lzma.compress(np.concatenate([np.diff(ei, prepend=0), ev]).astype(np.int64).tobytes())
bE, hE = pack(E.T); bG, hG = pack(G); bTE, hTE = pack(TE)
res = {"E_film_MB": mb(nbytes(bE, hE)), "flags_rc_MB": len(data) / 1e6, "ideal_flags_MB": H / 8 / 1e6,
       "exceptions": int(exc.sum()), "exceptions_bytes": len(blob_exc), "side_bytes": nbytes(bG, hG) + nbytes(bTE, hTE) + 8}
print(f"flags coded: {len(data):,} bytes ({res['flags_rc_MB']:.4f} MB) vs ideal {res['ideal_flags_MB']:.4f} MB  [{time.time()-t0:.0f}s]", flush=True)

# decode everything from the stored bytes
del R, flip, exc, bits, cx, That, side, ctx, ev, ei
E2 = np.ascontiguousarray(unpack(bE, hE).T); G2 = unpack(bG, hG); TE2 = unpack(bTE, hTE)
Th2, side2, ctx2 = lens_edges(E2, G2, TE2)
fl2 = np.frombuffer(lens_rc.decode(data, np.ascontiguousarray(ctx2.T).ravel().tolist(), 2 * NB), np.uint8)
T2 = Th2.astype(np.int64) + fl2.reshape(L, N).T * side2.astype(np.int64)
z = np.frombuffer(lzma.decompress(blob_exc), np.int64); k = len(z) // 2
pos = np.cumsum(z[:k]); T2l = np.ascontiguousarray(T2.T).ravel()
T2l[pos] = np.ascontiguousarray(Th2.T).ravel()[pos].astype(np.int64) + z[k:]
T2 = T2l.reshape(L, N).T
res["roundtrip_ok"] = bool(np.array_equal(E2, E) and np.array_equal(T2, T.astype(np.int64)))
res["residual_MB"] = res["flags_rc_MB"] + len(blob_exc) / 1e6
res["total_MB"] = res["E_film_MB"] + res["residual_MB"] + res["side_bytes"] / 1e6
res["T_bits_per_word_given_E"] = len(data) * 8 / N
print(json.dumps(res, indent=1)); print(f"done [{time.time()-t0:.0f}s]")
