"""LENS v2 (the mirror of v1): store the top film T; recompute the entry film E through the inverse lens; store the residual.

Inverse lens, all whole numbers (K = 24, K2 = 24):
   G_i = gf_i at 2^-24 (the same 768 numbers as v1)
   H_i = round(2^(K+K2) / G_i)            (long division, shift-and-subtract: the inverse gain, no float)
   D_w = the word's DC that the centring blocked, at 2^-20 (stored: 50,257 numbers -- side information)
   v = (T << 8) + D_w                      (the word back on its own level, at 2^-20)
   E^ = v * H_i shifted down by K2 + 4, rounded
   RE = E - E^                             stored
Predicted before running: T is coarser than E on 766 lines by 16/gf_i places of E, so RE is spread over ~16/gf_i
values: ~3.7 bits on a gf ~1.25 line, ~12 bits on the three gf < 0.01 lines (where T holds almost nothing of E).
Mean of log2(16/gf_i) over lines ~ 3.9 bits -> RE ~ 19 MB, so v2 total ~ 57 + 19 = 76 MB: about v1 (74.6 MB), because
the pair holds one picture; whichever film is stored, the other's extra detail must be paid once.
Shuffle control: inverse gains on randomly permuted lines (seed 0).
"""
import sys, time, json, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from lens_lib import *

K, K2 = 24, 24
t0 = time.time()
E, T, TE, HB, WPE = load()
N, L = E.shape
G = whole(lens_float("ln_f.weight"), K)
H = np.array([div_round(np.array([1 << (K + K2)], np.int64), int(g))[0] for g in G], dtype=np.int64)

# the DC the centring blocked, measured from E with the v1 lens (whole numbers), kept at 2^-20
D = np.empty(N, np.int64)
for a in range(0, N, 4096):
    D[a:a + 4096] = rshift_round(div_round((E[a:a + 4096] * G[None, :]).sum(1), 768), K - 4)


def inv_lens(T, D, H, TE, chunk=4096):
    out = np.empty(T.shape, np.int64)
    for a in range(0, T.shape[0], chunk):
        sh = (TE[a:a + chunk] + 20).astype(np.int64)[:, None]            # T at 2^TE -> 2^-20: shift up by TE + 20
        v = (T[a:a + chunk].astype(np.int64) << sh) + D[a:a + chunk, None]
        out[a:a + chunk] = rshift_round(v * H[None, :], K2 + 4)
    return out


res = {}
bT, hT = pack(T.T); res["T_film_MB"] = mb(nbytes(bT, hT))
bTE, hTE = pack(TE); bG, hG = pack(G); bD, hD = pack(D)
res["side_bytes"] = nbytes(bTE, hTE) + nbytes(bG, hG); res["DC_MB"] = mb(nbytes(bD, hD))
print(f"T film {res['T_film_MB']:.3f} MB, DC per word {res['DC_MB']:.3f} MB, gains+TE {res['side_bytes']} B [{time.time()-t0:.0f}s]", flush=True)
RE = E - inv_lens(T, D, H, TE)
gf = G / 2.0 ** K                                                       # report only
print("RE spread per line (max |RE|) vs predicted 8/gf_i:")
for idx in list(np.argsort(gf)[:3]) + list(np.argsort(gf)[380:382]) + list(np.argsort(gf)[-2:]):
    print(f"   line {idx:3d}  gf {gf[idx]:8.4f}  max|RE| {np.abs(RE[:, idx]).max():6d}   8/gf {8/gf[idx]:9.1f}   order-0 {order0_bits(RE[:, idx]):.2f} bits")
res["RE_order0_bits_per_value"] = order0_bits(RE)
res["RE_photons_per_value"] = float(ones(RE).mean()); res["RE_singlewave_per_value"] = float(naf(RE).mean())
bR, hR = pack(RE.T); res["RE_MB"] = mb(nbytes(bR, hR))
print(f"RE lzma {res['RE_MB']:.3f} MB  order-0 {res['RE_order0_bits_per_value']:.3f} bits/value  photons {res['RE_photons_per_value']:.3f} [{time.time()-t0:.0f}s]", flush=True)

T2 = np.ascontiguousarray(unpack(bT, hT).T).astype(np.int16); TE2 = unpack(bTE, hTE); G2 = unpack(bG, hG); D2 = unpack(bD, hD)
H2 = np.array([div_round(np.array([1 << (K + K2)], np.int64), int(g))[0] for g in G2], dtype=np.int64)
E2 = inv_lens(T2, D2, H2, TE2) + np.ascontiguousarray(unpack(bR, hR).T)
res["roundtrip_ok"] = bool(np.array_equal(E2, E) and np.array_equal(T2, T))
print("roundtrip:", res["roundtrip_ok"], flush=True)
del E2, T2

Hs = H[np.random.default_rng(0).permutation(L)]
REs = E - inv_lens(T, D, Hs, TE)
bRs, hRs = pack(REs.T); res["control_shuffled_RE_MB"] = mb(nbytes(bRs, hRs))
print(f"CONTROL shuffled inverse gains: RE lzma {res['control_shuffled_RE_MB']:.3f} MB", flush=True)
res["total_MB"] = res["T_film_MB"] + res["DC_MB"] + res["RE_MB"] + res["side_bytes"] / 1e6
print(json.dumps(res, indent=1)); print(f"done [{time.time()-t0:.0f}s]")
