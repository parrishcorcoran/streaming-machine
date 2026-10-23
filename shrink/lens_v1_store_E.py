"""LENS v1: store the entry film E; recompute the top film T through the whole-number lens; store only what it misses.

Lens (all whole numbers, k = 24):  G_i = gf_i at 2^-24 (768 numbers)
   Y = E * G                      (each line its gain: one product per value)
   M_w = round(sum_i Y / 768)     (the word's DC, long division)
   T^ = (Y - M_w) shifted down by 24 + 16 + TE_w, rounded
   R = T - T^                     the residual, stored
Predicted before running: E is finer than T on every line where gf_i < 16 (766 of 768 lines): E's rounding is
+-gf_i/32 of a T step, so a T rounding flips with chance about gf_i/64 -> ~2% of places are +-1, the rest 0;
on the two lines with gf ~17 about a quarter flip. Residual ~0.15 bits/place, under ~1.5 MB coded.
Shuffle control: the same lens with the 768 gains put on randomly permuted lines (seed 0).
"""
import sys, time, json, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from lens_lib import *

K = 24
t0 = time.time()
E, T, TE, HB, WPE = load()
N, L = E.shape
G = whole(lens_float("ln_f.weight"), K)


def lens(E, G, TE, K, chunk=4096):
    out = np.empty(E.shape, dtype=np.int16)
    for a in range(0, E.shape[0], chunk):
        Y = E[a:a + chunk] * G[None, :]
        Mw = div_round(Y.sum(1), 768)
        s = (K + 16 + TE[a:a + chunk]).astype(np.int64)[:, None]
        X = Y - Mw[:, None]
        out[a:a + chunk] = np.clip((X + (np.int64(1) << (s - 1))) >> s, -32767, 32767)
    return out


res = {}
# --- the films on their own, same coder (line-major: each line is one scan line across all words) ---
bE, hE = pack(E.T); res["E_alone_MB"] = mb(nbytes(bE, hE)); print(f"E film lzma (line-major byte planes): {res['E_alone_MB']:.3f} MB  [{time.time()-t0:.0f}s]", flush=True)
bT, hT = pack(T.T); res["T_alone_MB"] = mb(nbytes(bT, hT)); print(f"T film lzma: {res['T_alone_MB']:.3f} MB  [{time.time()-t0:.0f}s]", flush=True)
bTE, hTE = pack(TE); side_TE = nbytes(bTE, hTE)

# --- the lens ---
That = lens(E, G, TE, K)
R = T.astype(np.int64) - That
nz = R != 0
print(f"residual: nonzero {nz.mean()*100:.3f}%  photons/value {ones(R).mean():.4f}  single-wave {naf(R).mean():.4f}")
u, c = np.unique(R, return_counts=True)
print("residual histogram:", dict(zip(u.tolist(), c.tolist())))
frac_line = nz.mean(0)
gf = G / 2.0 ** K                                         # for the report only (not in any codec)
order = np.argsort(gf)
print("per-line flip fraction vs gf/64 (predicted):")
for idx in list(order[:3]) + list(order[380:383]) + list(order[-3:]):
    print(f"   line {idx:3d}  gf {gf[idx]:8.4f}  flips {frac_line[idx]*100:6.3f}%   gf/64 {gf[idx]/64*100:6.3f}%")
print(f"   mean over lines: flips {frac_line.mean()*100:.3f}%  predicted mean gf/64 {(np.minimum(gf/64, 1)).mean()*100:.3f}%")
bR, hR = pack(R.T)
bG, hG = pack(G)
res["residual_MB"] = mb(nbytes(bR, hR)); res["gains_bytes"] = nbytes(bG, hG); res["TE_bytes"] = side_TE
res["residual_nonzero_pct"] = float(nz.mean() * 100)
res["residual_order0_bits_per_value"] = order0_bits(R); res["residual_photons_per_value"] = float(ones(R).mean())
print(f"residual lzma: {res['residual_MB']:.3f} MB   gains {res['gains_bytes']} B   TE {side_TE} B  [{time.time()-t0:.0f}s]", flush=True)

# --- roundtrip through the stored bytes only ---
E2 = np.ascontiguousarray(unpack(bE, hE).T)
G2 = unpack(bG, hG); TE2 = unpack(bTE, hTE); R2 = np.ascontiguousarray(unpack(bR, hR).T)
T2 = (lens(E2, G2, TE2, K).astype(np.int64) + R2)
ok = bool(np.array_equal(E2, E) and np.array_equal(T2, T.astype(np.int64)))
res["roundtrip_ok"] = ok
print("roundtrip E and T from stored bytes:", ok, flush=True)
del E2, T2, R2

# --- shuffle control: the same lens, gains on randomly permuted lines ---
Gs = G[np.random.default_rng(0).permutation(L)]
Rs = T.astype(np.int64) - lens(E, Gs, TE, K)
bRs, hRs = pack(Rs.T)
res["control_shuffled_gains_residual_MB"] = mb(nbytes(bRs, hRs))
res["control_nonzero_pct"] = float((Rs != 0).mean() * 100)
print(f"CONTROL (gains on shuffled lines): residual lzma {res['control_shuffled_gains_residual_MB']:.3f} MB, nonzero {res['control_nonzero_pct']:.2f}%", flush=True)

res["total_store_E_plus_lens_MB"] = res["E_alone_MB"] + res["residual_MB"] + (res["gains_bytes"] + side_TE) / 1e6
res["both_films_lzma_MB"] = res["E_alone_MB"] + res["T_alone_MB"] + side_TE / 1e6
res["both_films_raw_MB"] = 86.8 + 77.2
print(json.dumps(res, indent=1))
print(f"done [{time.time()-t0:.0f}s]")
