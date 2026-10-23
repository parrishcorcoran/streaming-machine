"""LENS v10: the film that stays, coded near its floor with a whole-number coder of my own (v8's per-line exposure was
right in idea but lzma wastes ~0.27 bit a value on a small-alphabet stream; v8 also cut the grain too high).

Per line i (line-major, the order is the line number -- no sorting):
   z = zigzag(value)                        sign into the lowest place (shift, XOR)
   m_i = (sum z) >> bitlength(count),   k_i = max(0, bitlength(m_i) - 4),   group g_i = (m_i >> k_i) >> 1   (8 groups)
   top part z >> k_i  (0..254, 255 = escape)  -> interleaved rANS, 768 lanes (one per line), one frequency table per
                                                 group (8 tables x 256 counts on 2^14, stored)
   low k_i places                            -> raw, bit plane by bit plane (grain)
rANS: state x on 32 places. Encode: if x >= 2^18 f: emit low 16 places, x >>= 16; x = ((x // f) << 14) + x % f + c.
      Decode: slot = x AND (2^14 - 1); symbol = table[group][slot] (a table read); x = f (x >> 14) + slot - c;
      if x < 2^16: x = (x << 16) | next 16 places.   Whole numbers only.
Predicted: within ~0.1 bit a value of the per-line floors from v6: E -> ~72.4 MB (lzma 73.22), T -> ~55.5 MB (lzma 56.69).
"""
import sys, time, json, lzma, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from lens_lib import *

NM = 14; MM = 1 << NM; LR = np.uint64(1 << 16)
t0 = time.time()


def bitlen(x):
    x = np.asarray(x, dtype=np.uint64).copy(); n = np.zeros(x.shape, np.int64)
    while (x > 0).any(): n += (x > 0); x = x >> np.uint64(1)
    return n


def normalize(counts):                                    # counts -> frequencies summing to 2^14, every seen symbol >= 1
    tot = int(counts.sum())
    if tot == 0: f = np.zeros(256, np.int64); f[0] = MM; return f
    f = (counts.astype(np.int64) * MM) // tot
    f[(counts > 0) & (f == 0)] = 1
    f[int(np.flatnonzero(f == f.max())[0])] += MM - int(f.sum())   # the rounding remainder goes to the fullest bin
    return f


def encode_film(A):
    z = zigzag(np.ascontiguousarray(A.T)); Ln, N = z.shape
    m = z.sum(1) >> np.uint64(int(N).bit_length())       # about the line's mean size (sum shifted by the count's length)
    k = np.maximum(0, bitlen(m) - 4)
    grp = ((m >> k.astype(np.uint64)) >> np.uint64(1)).astype(np.int64)
    top = (z >> k.astype(np.uint64)[:, None]).astype(np.int64)
    esc = top >= 255
    sym = np.where(esc, 255, top)
    ei = np.flatnonzero(esc.ravel()); ev = top.ravel()[ei] - 255
    F = np.stack([normalize(np.bincount(sym[grp == g].ravel(), minlength=256)) for g in range(8)])
    C = np.concatenate([np.zeros((8, 1), np.int64), np.cumsum(F, 1)[:, :-1]], 1)
    Fl, Cl = F[grp].astype(np.uint64), C[grp].astype(np.uint64)
    lane = np.arange(Ln); x = np.full(Ln, LR, np.uint64)
    out = np.zeros((Ln, N + 2), np.uint16); ptr = np.zeros(Ln, np.int64)
    for t in range(N - 1, -1, -1):
        s = sym[:, t]; f = Fl[lane, s]; c = Cl[lane, s]
        need = x >= ((LR >> np.uint64(NM)) << np.uint64(16)) * f
        if need.any():
            ln = lane[need]; out[ln, ptr[ln]] = (x[ln] & np.uint64(0xFFFF)).astype(np.uint16); ptr[ln] += 1; x[ln] >>= np.uint64(16)
        x = ((x // f) << np.uint64(NM)) + (x % f) + c
    grain = []
    for i in range(Ln):
        for b in range(int(k[i])):
            grain.append(np.packbits(((z[i] >> np.uint64(b)) & np.uint64(1)).astype(np.uint8)).tobytes())
    words = b"".join(out[i, :ptr[i]].tobytes() for i in range(Ln))
    P = dict(words=words, ptr=lzma.compress(ptr.astype(np.uint32).tobytes()), states=x.astype(np.uint32).tobytes(),
             tables=lzma.compress(F.astype(np.uint16).tobytes()), kg=lzma.compress((k * 8 + grp).astype(np.uint16).tobytes()),
             grain=b"".join(grain), esc=lzma.compress(np.concatenate([np.diff(ei, prepend=0), ev]).astype(np.int64).tobytes()),
             shape=(Ln, N))
    return P


def size_parts(P): return {key: len(v) for key, v in P.items() if key != "shape"}


def decode_film(P):
    Ln, N = P["shape"]
    ptr = np.frombuffer(lzma.decompress(P["ptr"]), np.uint32).astype(np.int64)
    kg = np.frombuffer(lzma.decompress(P["kg"]), np.uint16).astype(np.int64); k, grp = kg >> 3, kg & 7
    F = np.frombuffer(lzma.decompress(P["tables"]), np.uint16).astype(np.int64).reshape(8, 256)
    C = np.concatenate([np.zeros((8, 1), np.int64), np.cumsum(F, 1)[:, :-1]], 1)
    D = np.stack([np.repeat(np.arange(256), F[g]) for g in range(8)])            # slot -> symbol, table read
    Fl, Cl = F[grp].astype(np.uint64), C[grp].astype(np.uint64)
    W = np.frombuffer(P["words"], np.uint16); off = np.concatenate([[0], np.cumsum(ptr)[:-1]])
    out = np.zeros((Ln, int(ptr.max()) + 1), np.uint16)
    for i in range(Ln): out[i, :ptr[i]] = W[off[i]:off[i] + ptr[i]]
    x = np.frombuffer(P["states"], np.uint32).astype(np.uint64); lane = np.arange(Ln); p = ptr.copy()
    top = np.empty((Ln, N), np.uint64); mask = np.uint64(MM - 1)
    for t in range(N):
        slot = (x & mask).astype(np.int64); s = D[grp, slot]
        x = Fl[lane, s] * (x >> np.uint64(NM)) + slot.astype(np.uint64) - Cl[lane, s]
        need = x < LR
        if need.any():
            ln = lane[need]; p[ln] -= 1; x[ln] = (x[ln] << np.uint64(16)) | out[ln, p[ln]].astype(np.uint64)
        top[:, t] = s
    z2 = np.frombuffer(lzma.decompress(P["esc"]), np.int64); ne = len(z2) // 2
    if ne:
        ei = np.cumsum(z2[:ne]); tf = top.ravel(); tf[ei] = (255 + z2[ne:]).astype(np.uint64); top = tf.reshape(Ln, N)
    z = top << k.astype(np.uint64)[:, None]
    g = np.frombuffer(P["grain"], np.uint8); pos = 0; nb = (N + 7) // 8
    for i in range(Ln):
        for b in range(int(k[i])):
            z[i] |= np.unpackbits(g[pos:pos + nb])[:N].astype(np.uint64) << np.uint64(b); pos += nb
    return np.ascontiguousarray(unzigzag(z).T)


E, T, TE, HB, WPE = load()
res = {}
for name, A in (("E", E), ("T", T), ("wpe", WPE)):
    P = encode_film(A); parts = size_parts(P); tot = sum(parts.values())
    back = decode_film(P); ok = bool(np.array_equal(back, A.astype(np.int64)))
    res[name] = {"MB": tot / 1e6, "bits_per_value": tot * 8 / A.size, "roundtrip_ok": ok, "parts_bytes": parts}
    print(f"{name}: {tot/1e6:.3f} MB  ({tot*8/A.size:.3f} bits/value)  roundtrip {ok}  parts {parts}  [{time.time()-t0:.0f}s]", flush=True)
    del back, P
res["best_lossless_E_film_plus_T_edge_flags_MB"] = res["E"]["MB"] + (337799 + 280 + 2900) / 1e6
res["best_one_film_T_plus_DC_plus_gains_MB_30of30"] = res["T"]["MB"] + (86940 + 2892) / 1e6
print(json.dumps(res, indent=1)); print(f"done [{time.time()-t0:.0f}s]")
