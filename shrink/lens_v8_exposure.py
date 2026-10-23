"""LENS v8: each line its own exposure. The film that stays (E for the lossless system, T for the one-film system) has a
very different spread on each line -- T's spread IS the lens: gf_i times E's spread. One histogram for all lines (and
lzma on byte planes) pays for the mismatch. Here each line is split by a fixed whole-number rule:
    m_i = (sum of the line's zigzagged values) >> 15        (about 1.5 x the line's mean size)
    k_i = max(0, bitlength(m_i) - 3)                        (the places below ~1/8 of the line's size: grain)
    top part  z >> k_i   -> lzma (line-major: the picture)
    low k_i places        -> stored raw, packed bit plane by bit plane (grain: every pattern equally likely)
No search: k_i comes from one formula. Predicted: close to the per-line floor (E 71.6 MB, T 54.7 MB from v6),
i.e. ~1.5 MB off E and ~2 MB off T compared with plain byte planes (73.2 / 56.7 MB).
Then the totals: v3 lossless (E film + T's edge flags) and v5a one-film (T film + DC + gains, 30/30).
"""
import sys, time, json, lzma, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from lens_lib import *

t0 = time.time()
E, T, TE, HB, WPE = load()
N, L = E.shape


def bitlen(x):                                            # position of the top 1, whole numbers only
    x = np.asarray(x, dtype=np.uint64).copy(); n = np.zeros(x.shape, np.int64)
    while (x > 0).any(): nz = x > 0; n += nz; x = x >> np.uint64(1)
    return n


def pack_lines(A):
    z = zigzag(np.ascontiguousarray(A.T))                 # (lines, words)
    m = z.sum(axis=1) >> np.uint64(15)
    k = np.maximum(0, bitlen(m) - 3)
    hi = z >> k.astype(np.uint64)[:, None]
    bhi, hhi = pack_u(hi)
    lo = []
    for i in range(z.shape[0]):
        for b in range(int(k[i])):
            lo.append(np.packbits(((z[i] >> np.uint64(b)) & np.uint64(1)).astype(np.uint8)).tobytes())
    lo = b"".join(lo)
    return dict(hi=bhi, hhi=hhi, lo=lo, k=lzma.compress(k.astype(np.uint8).tobytes()), shape=z.shape)


def size(P): return nbytes(P["hi"], P["hhi"]) + len(P["lo"]) + len(P["k"]) + 8


def unpack_lines(P):
    nl, nw = P["shape"]
    k = np.frombuffer(lzma.decompress(P["k"]), np.uint8).astype(np.int64)
    hi = unpack_u(P["hi"], P["hhi"])
    z = hi << k.astype(np.uint64)[:, None]
    lo = np.frombuffer(P["lo"], np.uint8); pos = 0; nb = (nw + 7) // 8
    for i in range(nl):
        for b in range(int(k[i])):
            z[i] |= np.unpackbits(lo[pos:pos + nb])[:nw].astype(np.uint64) << np.uint64(b); pos += nb
    return np.ascontiguousarray(unzigzag(z).T)


res = {}
for name, A in (("E", E), ("T", T)):
    P = pack_lines(A)
    back = unpack_lines(P)
    ok = bool(np.array_equal(back, A.astype(np.int64)))
    res[f"{name}_film_MB_per_line_exposure"] = size(P) / 1e6; res[f"{name}_roundtrip_ok"] = ok
    res[f"{name}_top_part_MB"] = nbytes(P["hi"], P["hhi"]) / 1e6; res[f"{name}_raw_grain_MB"] = len(P["lo"]) / 1e6
    print(f"{name}: per-line exposure {size(P)/1e6:.3f} MB (top part lzma {res[f'{name}_top_part_MB']:.3f} + raw grain "
          f"{res[f'{name}_raw_grain_MB']:.3f}), roundtrip {ok}  [{time.time()-t0:.0f}s]", flush=True)
    del back, P
# totals with the measured side pieces from v3 (flags 337,799 B, exceptions 280 B, gains+TE 2,900 B) and v2/v5a (DC 86,940 B, gains+TE 2,892 B)
res["v3_lossless_total_MB"] = res["E_film_MB_per_line_exposure"] + (337799 + 280 + 2900) / 1e6
res["v5a_one_film_total_MB"] = res["T_film_MB_per_line_exposure"] + (86940 + 2892) / 1e6
print(json.dumps(res, indent=1)); print(f"done [{time.time()-t0:.0f}s]")
