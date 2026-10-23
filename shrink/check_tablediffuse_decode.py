"""CHECK of the tablediffuse (diffuse_tables) family -- step 1, the reader alone.
Sees only each .tdf file and a coder written by the checker (check_tablediffuse_rc.c, compiled to the scratchpad).
Imports nothing of the machine, nothing of develop_lib / tablediffuse_*. Parses every byte, rebuilds the developed entry
table D and prints the top table T through the gains read from the file (whole numbers: shifts, adds, ANDs).
Writes scratchpad/chk_tdf_<tag>.npz (D int32, T, G, c) for step 2 (the words)."""
import sys, os, struct, ctypes, json, numpy as np
SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad"
lib = ctypes.CDLL(os.path.join(SP, "chk_tdf_rc.dylib")); lib.chk_decode.restype = ctypes.c_int64
P = ctypes.c_void_p

def rc(buf, ctx, nbits, nctx, nmax):
    b = np.frombuffer(buf, np.uint8).copy(); ctx = np.ascontiguousarray(ctx, np.uint16); out = np.empty(len(ctx), np.uint16)
    used = lib.chk_decode(b.ctypes.data_as(P), ctypes.c_int64(len(b)), ctx.ctypes.data_as(P), ctypes.c_int64(len(ctx)),
                          ctypes.c_int(nbits), ctypes.c_int64(nctx), ctypes.c_int(nmax), out.ctypes.data_as(P))
    assert used >= 0
    return out, int(used)

def unzz(z): z = z.astype(np.int64); return (z >> 1) ^ (-(z & 1))

def rm13():                       # the 16 words of the [8,4,4] code, index a*2+b: bit x = (popcount(a&x) + b) & 1
    B = np.zeros((16, 8), np.int64)
    for a in range(8):
        for b in range(2):
            for x in range(8): B[a * 2 + b, x] = (bin(a & x).count("1") + b) & 1
    return B

def read(path):
    data = open(path, "rb").read(); p = 0; side = {}
    magic, kind, c, cap, w, nmax, n, L = struct.unpack_from("<4sBBBBHIH", data, p); p += 16; side["header"] = 16
    assert magic == b"TDF1" and L == 768, (magic, L)
    G = np.frombuffer(data, np.int32, 768, p).astype(np.int64); p += 3072; side["gains"] = 3072
    assert (G > 0).all()
    place = np.array([int(g).bit_length() - 1 for g in G], np.int64)
    e = np.minimum(np.maximum(c - place, 0), cap)
    grp = None
    if kind:
        grp = np.frombuffer(data, np.uint16, 768, p).astype(np.int64); p += 1536; side["groups"] = 1536
        assert np.array_equal(np.sort(grp), np.arange(768))
    lm, lc = struct.unpack_from("<II", data, p); p += 8; side["lengths"] = 8
    main = data[p:p + lm]; p += lm; cws = data[p:p + lc]; p += lc
    assert p == len(data), (p, len(data))
    N = n * 768; D = np.empty((n, 768), np.int32)
    if kind == 0:
        sym, used = rc(main, np.tile(np.arange(768, dtype=np.uint16), n), w, 768, nmax)
        for r in range(0, n, 4096):
            q = unzz(sym[r * 768:(r + 4096) * 768]).reshape(-1, 768); D[r:r + 4096] = q << e[None, :]
    else:
        pat = grp.copy()
        if kind == 1: pat[7::8] += 768
        sym, used = rc(main, np.tile(pat.astype(np.uint16), n), w, 1536 if kind == 1 else 768, nmax)
        cw = None
        if kind == 2:
            cw, used2 = rc(cws, np.tile(np.arange(96, dtype=np.uint16), n), 4, 96, nmax); cw = cw.reshape(n, 96).astype(np.int64)
            assert cw.max() < 16
        B = rm13()
        for r in range(0, n, 4096):
            m = unzz(sym[r * 768:(r + 4096) * 768]).reshape(-1, 96, 8)
            if kind == 1:
                K = m.copy(); K[:, :, 7] = 2 * m[:, :, 7] + (K[:, :, :7].sum(2) & 1)
                assert not (K.sum(2) & 1).any()
            else:
                K = 2 * m + B[cw[r:r + 4096]]
            Kl = np.empty((K.shape[0], 768), np.int64); Kl[:, grp] = K.reshape(-1, 768)
            D[r:r + 4096] = Kl << e[None, :]
    del sym
    # print the top table through the lens (whole numbers)
    T = np.empty((n, 768), np.int32); tmax = 0
    for r in range(0, n, 4096):
        sv = D[r:r + 4096].astype(np.int64) * G[None, :]
        cc = 768 * sv - sv.sum(1, keepdims=True); y = (cc + (1 << 23)) >> 24
        assert int(np.abs(y).max()) < (1 << 32)
        t = (y * 1431655766 + (1 << 43)) >> 44; T[r:r + 4096] = t; tmax = max(tmax, int(np.abs(t).max()))
    side["streams_main"] = lm; side["streams_codewords"] = lc
    info = dict(file=os.path.basename(path), kind=["cube", "D8", "E8"][kind], c=c, cap=cap, width=w, nmax=nmax, rows=n,
                file_bytes=len(data), parsed_bytes=p, parts=side, parts_sum=sum(side.values()),
                MB=round(len(data) / 1e6, 3), bits_per_value=round(8 * len(data) / N, 4),
                T_max_abs=tmax, T_fits_int16=tmax < 32768, e_min=int(e.min()), e_max=int(e.max()))
    return D, T, G, info

if __name__ == "__main__":
    jobs = [("E8c32", "tables_E8_c32.tdf"), ("D8c34", "tables_D8_c34.tdf"), ("cubec34", "tables_cube_c34.tdf")]
    for tag, fn in jobs:
        if sys.argv[1:] and tag not in sys.argv[1:]: continue
        D, T, G, info = read(os.path.join(SP, fn))
        np.savez(os.path.join(SP, f"chk_tdf_{tag}.npz"), D=D, T=T, G=G, c=np.int64(info["c"]))
        print(tag, json.dumps(info), flush=True)
        del D, T
