"""FAX family, versions 1 and 2: plain base-2 pages, store ONLY the lit pixels. Each place's page is scanned like a fax
line; each lit pixel stores how many dark pixels came before it (the gap). The sign of each lit value is one more bit.
  scan row: along the 768 lines of a word, word after word        (v1)
  scan col: along the 50,257 words of a line, line after line     (v2)
  gaps written as a Rice code (q = gap >> k in unary, then k low bits; k = floor(log2(dark per lit)), a fixed rule
  from the page's two counts) -- or as 7-bit groups (a whole-number varint), each page through its own lzma.
Side info counted: per page the lit count, k, split point; per table its shape. Decoded and checked equal.
    python3 fax_v1_gaps.py E|T|fc.0
"""
import sys, json, time, lzma, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from fax_common import *

def varint(g):                                   # 7 bits per byte, top bit = more to come
    g = g.astype(np.int64)
    L = np.ones(len(g), np.int64); x = g >> 7
    while x.any(): L += x > 0; x >>= 7
    out = np.zeros(int(L.sum()), np.uint8); start = np.cumsum(L) - L
    for j in range(int(L.max())):
        m = L > j; v = (g[m] >> (7 * j)) & 127
        out[start[m] + j] = v | np.where(L[m] > j + 1, 128, 0)
    return out.tobytes()

def unvarint(buf):
    b = np.frombuffer(buf, np.uint8)
    end = b < 128; first = np.flatnonzero(np.concatenate(([True], end[:-1]))).astype(np.int32)
    grp = np.repeat(np.arange(len(first), dtype=np.int32), np.diff(np.append(first, len(b))))
    j = (np.arange(len(b), dtype=np.int32) - first[grp]); del grp
    return np.add.reduceat((b & 127).astype(np.int32) << (7 * j), first).astype(np.int64)

def code(A, scan, how):
    S = (A if scan == "row" else A.T).ravel(); a = np.abs(S); N = S.size; nb = nplanes(S)
    chunks, side = [], 2 * 32 + 5
    for k in range(nb):
        lit = ((a >> k) & 1).astype(bool)
        if how == "rice":
            buf, n, kk, s = gaps_encode(lit); chunks.append((buf, n, kk)); side += s
        else:
            pos = np.flatnonzero(lit); g = np.diff(pos, prepend=-1) - 1
            chunks.append((lzma.compress(varint(g), preset=9), len(pos), 0)); side += 32 + 32   # each page its own lzma
    sg = raw_bits(S[S != 0] < 0)
    return (chunks, sg, nb, N), sum(len(c[0]) for c in chunks) + len(sg), side

def decode(pkg, shape, scan, how):
    ch, sg, nb, N = pkg; chunks = [(c[0], c[1], c[2]) for c in ch]
    a = np.zeros(N, np.int64)
    for k, c in enumerate(chunks):
        if how == "rice": lit = gaps_decode(c[0], c[1], c[2], N)
        else:
            lit = np.zeros(N, bool)
            if c[1]: lit[np.cumsum(unvarint(lzma.decompress(c[0])) + 1) - 1] = True
        a |= lit.astype(np.int64) << k
    nz = np.flatnonzero(a); a[nz[unraw_bits(sg, len(nz))]] *= -1
    R, C = shape
    return a.reshape(R, C) if scan == "row" else a.reshape(C, R).T

name = sys.argv[1] if len(sys.argv) > 1 else "E"
A = load(name); res = []
for scan, how, v in [("row", "rice", "v1"), ("col", "rice", "v2"), ("row", "lzma", "v1"), ("col", "lzma", "v2")]:
    t = time.time()
    pkg, nbytes, side = code(A, scan, how)
    ok = bool(np.array_equal(decode(pkg, A.shape, scan, how), A)); del pkg
    res.append(report(f"{v} lit-pixel gaps, scan {scan}, {how}", name, nbytes, side, A.size, ok))
    print(f"     ({time.time() - t:.0f} s)", flush=True)
json.dump(res, open(f"/Users/abundancemachine/streaming-machine/shrink/fax_out_v1_{name}.json", "w"))
