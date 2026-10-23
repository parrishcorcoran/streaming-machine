"""TABLEDIFFUSE family -- the word tables as ONE real file, and a reader that sees only that file.

File (little endian):
  header  "<4sBBBBHIH": b"TDF1", kind (0 cube = round one's grid, 1 D8 parity screen, 2 E8 parity screen), c, cap,
          symbol width, the coder's NMAX, rows (50,257), lines (768)                                   16 bytes
  the lens: 768 whole-number gains at 2^24 (int32)                                                    3,072 bytes
  kind 1/2: the 96 groups of 8 lines (768 line numbers, uint16)                                       1,536 bytes
  two stream lengths (uint32)                                                                             8 bytes
  the stored whole numbers, range coded (counting coder, tablediffuse_rc.c), contexts = the line (+768 for D8's halved
  8th number); kind 2: the code words, 4 bits a group, range coded with the group as context.
The reader rebuilds each line's exposure e_i = min(max(c - place(gain_i), 0), cap) from the gains it read, the developed
entry table D = k << e_i, and prints the top table from D through the same gains (whole numbers: G x D, centre, round,
x 1431655766 >> 44). Nothing else is read: no table, no npy, no seed.
"""
import os, sys, struct, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import develop_lib as dl
import tablediffuse_lib as td

HDR = "<4sBBBBHIH"
KIND = {"cube": 0, "D8": 1, "E8": 2}

def print_top_with(G, D):
    """the top table printed from D through the lens G (the same whole-number steps as develop_lib.print_top)"""
    out = np.empty(D.shape, np.int32)
    for r in range(0, D.shape[0], dl.CH):
        sv = D[r:r + dl.CH].astype(np.int64) * G[None, :]
        c = 768 * sv - sv.sum(1, keepdims=True); y = (c + (1 << 23)) >> 24
        out[r:r + dl.CH] = (y * 1431655766 + (1 << 43)) >> 44
    return out

def write(path, D, c, kind, grp=None, cw=None, cap=12, nmax=255):
    """D: the developed entry table (on the grid of exposure c). Returns the file size in bytes."""
    G = dl.gains().astype(np.int64); n = D.shape[0]
    e, _ = td.exposure(c, cap); e = e.astype(np.int64)
    if kind == "cube":
        sym = np.empty(D.size, np.uint16)
        for r in range(0, n, dl.CH):
            q = D[r:r + dl.CH].astype(np.int64) >> e[None, :]; assert np.array_equal(q << e[None, :], D[r:r + dl.CH])
            z = dl.zigzag(q).ravel(); sym[r * 768:r * 768 + z.size] = z
        ctx = np.tile(np.arange(768, dtype=np.uint16), n); nctx = 768
    else:
        sym, ctx = td.lattice_streams(D, e, grp, kind, cw); nctx = 1536 if kind == "D8" else 768
    w = dl.nb(int(sym.max())); main = td.rc2_encode(sym, ctx, w, nctx, nmax); del sym, ctx
    cws = b""
    if kind == "E8":
        cws = td.rc2_encode(cw.ravel().astype(np.uint16), np.tile(np.arange(96, dtype=np.uint16), n), 4, 96, nmax)
    with open(path, "wb") as f:
        f.write(struct.pack(HDR, b"TDF1", KIND[kind], c, cap, w, nmax, n, 768))
        f.write(G.astype(np.int32).tobytes())
        if kind != "cube": f.write(grp.astype(np.uint16).tobytes())
        f.write(struct.pack("<II", len(main), len(cws))); f.write(main); f.write(cws)
    return os.path.getsize(path)

def read(path):
    """the reader: only the file. Returns (developed entry table D int32, printed top table T int32, info)."""
    data = open(path, "rb").read(); p = 0
    magic, kind, c, cap, w, nmax, n, L = struct.unpack_from(HDR, data, p); p += struct.calcsize(HDR)
    assert magic == b"TDF1" and L == 768
    G = np.frombuffer(data, np.int32, 768, p).astype(np.int64); p += 768 * 4
    place = np.array([int(g).bit_length() - 1 for g in G], np.int64)
    e = np.minimum(np.maximum(c - place, 0), cap)
    grp = None
    if kind != 0: grp = np.frombuffer(data, np.uint16, 768, p).astype(np.int32); p += 768 * 2
    lm, lc = struct.unpack_from("<II", data, p); p += 8
    main = data[p:p + lm]; p += lm; cws = data[p:p + lc]; p += lc
    assert p == len(data)
    if kind == 0:
        ctx = np.tile(np.arange(768, dtype=np.uint16), n)
        sym = td.rc2_decode(main, ctx, w, 768, nmax); del ctx
        D = np.empty((n, 768), np.int32)
        for r in range(0, n, dl.CH):
            D[r:r + dl.CH] = (dl.unzigzag(sym[r * 768:(r + dl.CH) * 768].astype(np.int64)).reshape(-1, 768) << e[None, :])
    else:
        name = "D8" if kind == 1 else "E8"
        cx = np.broadcast_to(grp.reshape(96, 8).astype(np.int64), (n, 96, 8)).copy()
        if name == "D8": cx[:, :, 7] += 768
        ctx = cx.ravel().astype(np.uint16); del cx
        sym = td.rc2_decode(main, ctx, w, 1536 if name == "D8" else 768, nmax); del ctx
        cwd = None
        if name == "E8": cwd = td.rc2_decode(cws, np.tile(np.arange(96, dtype=np.uint16), n), 4, 96, nmax).reshape(n, 96)
        D = td.lattice_rebuild(sym, n, e, grp, name, cwd)
    del sym
    T = print_top_with(G, D)
    return D, T, dict(kind=["cube", "D8", "E8"][kind], c=c, cap=cap, nmax=nmax, bytes=len(data))
