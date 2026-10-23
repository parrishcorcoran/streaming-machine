"""LENS v9: how many photons does the lens itself need? Each line's gain G_i (gf_i at 2^-24) written in the single-wave
form (signed places, no two lit side by side) and cut to its top P photons. P = 1 is a lens made only of shifts
(magnification by whole octaves, no product at all).
Relation tested (predicted before running): cutting the gain to P signed photons leaves a relative gain error of about
2^-(3P) (single-wave places are lit about one in three). T^ is then off by about 2^-(3P) x |T|; the residual stays at
the edge flips only once that is below E's own grain gf_i/32 of a T step. |T| reaches ~2^9 on a typical line and ~2^14.6
on the brightest, so P must reach ~5-6 before the residual falls to v1's 1.36 MB; at P = 1-2 the residual holds
nearly all of T (tens of MB). Lossless every time (roundtrip checked on the last one); lzma-coded like v1.
"""
import sys, time, json, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from lens_lib import *

K = 24
t0 = time.time()
E, T, TE, HB, WPE = load()
N, L = E.shape
G = whole(lens_float("ln_f.weight"), K)


def naf_digits(x):                                        # single-wave digits of a whole number: list of (place, +1/-1), top first
    d = []; i = 0
    while x:
        if x & 1:
            s = 2 - (x & 3); d.append((i, s)); x -= s
        x >>= 1; i += 1
    return d[::-1]


def cut(g, P): return sum(s << p for p, s in naf_digits(int(g))[:P])


def lens(E, G):
    out = np.empty(E.shape, dtype=np.int16)
    for a in range(0, N, 4096):
        Y = E[a:a + 4096] * G[None, :]
        X = Y - div_round(Y.sum(1), 768)[:, None]
        s = (K + 16 + TE[a:a + 4096]).astype(np.int64)[:, None]
        out[a:a + 4096] = np.clip((X + (np.int64(1) << (s - 1))) >> s, -32767, 32767)
    return out


full_photons = np.array([len(naf_digits(int(g))) for g in G])
print(f"the full lens: {full_photons.sum()} photons over 768 gains ({full_photons.mean():.2f} a gain)", flush=True)
res = {"full_lens_photons": int(full_photons.sum())}
for P in (1, 2, 3, 4, 5, 6, 8, 99):
    GP = np.array([cut(g, P) for g in G], dtype=np.int64)
    R = T.astype(np.int64) - lens(E, GP)
    b, h = pack(R.T)
    r = {"photons_in_lens": int(sum(min(P, n) for n in full_photons)), "residual_MB": nbytes(b, h) / 1e6,
         "nonzero_pct": float((R != 0).mean() * 100), "max_abs": int(np.abs(R).max()), "max_rel_gain_err": float(np.max(np.abs(GP - G) / G))}
    if P == 99:
        R2 = unpack(b, h).reshape(L, N).T
        r["roundtrip_ok"] = bool(np.array_equal(lens(E, GP).astype(np.int64) + R2, T.astype(np.int64)))
    res[f"P{P}"] = r
    print(f"P={P:2d}: lens photons {r['photons_in_lens']:5d}  max rel gain err {r['max_rel_gain_err']:.2e}  residual {r['residual_MB']:8.3f} MB  "
          f"nonzero {r['nonzero_pct']:6.2f}%  max|R| {r['max_abs']}  [{time.time()-t0:.0f}s]", flush=True)
print(json.dumps(res, indent=1))
