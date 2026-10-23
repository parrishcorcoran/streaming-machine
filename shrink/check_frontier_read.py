"""CHECKER for the frontier/combined family's stored machine files (.cmf). Independent of the family's reader:
my own container parser, my own rebuild of steps / exposures / zigzag / printed top table, and decoders compiled by me
from plates_rc.c and combined_rc.c into the scratchpad (no file I/O in either source). The original machine
(base2_gpt2.npz) is touched only AFTER every part is decoded, to (a) compare the parts claimed exact, (b) get the
original next words (the truth) the harness compares against.
    python3 shrink/check_frontier_read.py FILE [FILE ...]"""
import sys, os, struct, ctypes, lzma, json, time, hashlib, numpy as np
SM = "/Users/abundancemachine/streaming-machine"
SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/chk_frontier/"
P_ = ctypes.c_void_p
PL = ctypes.CDLL(SP + "plates_rc_mine.dylib"); PL.plate_decode.restype = ctypes.c_int
PL.plate_decode.argtypes = [P_, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, P_, ctypes.c_int]
CR = ctypes.CDLL(SP + "combined_rc_mine.dylib"); CR.crc_decode.restype = ctypes.c_int64
NAMES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]

def photons(buf, R, C):
    b = np.frombuffer(buf, np.uint8).copy(); v = np.zeros((R, C), np.int32)
    assert PL.plate_decode(b.ctypes.data, len(b), R, C, v.ctypes.data, 1) == 0
    return v

def lines_dec(buf, n, nbits, lim):
    ctx = (np.arange(n, dtype=np.int64) % 768).astype(np.uint16); b = np.frombuffer(buf, np.uint8).copy()
    out = np.empty(n, np.uint16)
    CR.crc_decode(b.ctypes.data_as(P_), ctypes.c_int64(len(b)), ctx.ctypes.data_as(P_), ctypes.c_int64(n),
                  ctypes.c_int(nbits), ctypes.c_int64(768), ctypes.c_int(lim), out.ctypes.data_as(P_))
    return out

def top_place(x):                                    # place of the top 1 (bit length - 1), by shifts and compares
    x = np.abs(np.asarray(x, np.int64)); n = np.zeros(x.shape, np.int64)
    for _ in range(64):
        m = x > 0
        if not m.any(): break
        n += m; x = x >> 1
    return n - 1

def decode_file(path):
    raw = open(path, "rb").read(); assert raw[:4] == b"CMF1"
    secs = {}; p = 4
    while p < len(raw):
        tag = raw[p:p + 4].decode(); (ln,) = struct.unpack_from("<I", raw, p + 4)
        assert tag not in secs; secs[tag] = raw[p + 8:p + 8 + ln]; p += 8 + ln
    assert p == len(raw), "trailing bytes"
    out = dict(file_bytes=len(raw), sections={k: len(v) + 8 for k, v in secs.items()})
    # plates
    d = secs["PLAT"]; q = 0; plates = {}; Aof = {}
    while q < len(d):
        idx, R, C, A, t0, o0, nb = struct.unpack_from("<BHHbhhI", d, q); q += 14
        e = np.frombuffer(d, np.int8, C, q).astype(np.int64); q += C
        t = np.frombuffer(d, np.uint8, C, q).astype(np.int64) + t0; q += C
        off = np.frombuffer(d, np.uint8, R, q).astype(np.int64) + o0; q += R
        v = photons(d[q:q + nb], R, C).astype(np.int64); q += nb
        s = (t[None, :] + off[:, None] - A) >> 1; s[s < 0] = 0
        w = v << s; w = np.where(w > 32767, 32767, np.where(w < -32768, -32768, w)).astype(np.int16)
        n = NAMES[idx]; assert n not in plates; plates[n] = (w, e); Aof[n] = int(A)
        del v, s
    assert len(plates) == 48
    out["A"] = Aof
    # biases
    d = secs["BIAS"]; q = 0; bias = {}
    for n in NAMES:
        (ln,) = struct.unpack_from("<I", d, q); q += 4
        bias[n] = photons(d[q:q + ln], 1, plates[n][0].shape[1])[0].astype(np.int64); q += ln
    assert q == len(d)
    # entry table
    d = secs["TABL"]; c, cap, half, nbits, lim, rows, nbq = struct.unpack_from("<BBBBBII", d, 0); q = 13
    if half: ex = np.frombuffer(d, np.uint8, 768, q).astype(np.int64); q += 768
    (lg,) = struct.unpack_from("<I", d, q); q += 4
    G = np.frombuffer(lzma.decompress(d[q:q + lg]), np.int32).astype(np.int64); q += lg; assert G.size == 768
    if not half:
        ex = c - top_place(G); ex[ex < 0] = 0; ex[ex > cap] = cap
    z = lines_dec(d[q:q + nbq], rows * 768, nbits, lim); q += nbq; assert q == len(d)
    D = np.empty((rows, 768), np.int32)
    for r in range(0, rows, 4096):
        zz = z[r * 768:(r + 4096) * 768].astype(np.int64).reshape(-1, 768)
        k = np.where(zz & 1, -((zz + 1) >> 1), zz >> 1)          # un-zigzag
        D[r:r + 4096] = k << ex[None, :]
    del z
    Tp = np.empty((rows, 768), np.int64)                          # the top table printed through the stored lens
    for r in range(0, rows, 4096):
        sv = D[r:r + 4096].astype(np.int64) * G[None, :]
        cc = 768 * sv - sv.sum(1, keepdims=True); y = (cc + (1 << 23)) >> 24
        assert int(np.abs(y).max()) < (1 << 32)
        Tp[r:r + 4096] = (y * 1431655766 + (1 << 43)) >> 44
    out.update(c=int(c), cap=int(cap), half=int(half), nbits=int(nbits), lim=int(lim), rows=int(rows),
               exposure_hist=np.bincount(ex).tolist(), Tp_range=[int(Tp.min()), int(Tp.max())])
    Tp = Tp.astype(np.int32)
    # head
    d = secs["HEAD"]; (ln,) = struct.unpack_from("<I", d, 0)
    hb = photons(d[4:4 + ln], 1, rows)[0].astype(np.int64); (he,) = struct.unpack_from("<b", d, 4 + ln); assert 4 + ln + 1 == len(d)
    # positions
    d = secs["POSN"]; R, C = struct.unpack_from("<HH", d, 0); wpe = np.cumsum(photons(d[4:], R, C).astype(np.int64), axis=0)
    # arithmetic tables
    d = secs["ARIT"]; q = 0; ar = {}
    for k in ("phi", "arrival", "log2", "exp2"):
        a0, n, ln = struct.unpack_from("<qII", d, q); q += 16
        ar[k] = np.concatenate([[a0], a0 + np.cumsum(photons(d[q:q + ln], 1, n - 1)[0].astype(np.int64))]); q += ln
    l768, eps = struct.unpack_from("<qq", d, q); q += 16; assert q == len(d)
    out.update(plates=plates, bias=bias, D=D, Tp=Tp, G=G, hb=hb, he=int(he), wpe=wpe, ar=ar, l768=l768, eps=eps)
    return out

if __name__ == "__main__":
    files = sys.argv[1:]
    sys.path.insert(0, SM); os.chdir(SM)
    import table_bits as tb
    tb.E = None; tb.T = None
    M = tb.M
    assert tb.words() == 30                                       # truth: the original machine's own next words
    tb._WTE0 = None; H0 = tb.W["head"]; tb.W["head"] = (None, H0[1], H0[2]); del H0   # (memory)
    for path in files:
        t0 = time.time(); m = decode_file(path); tdec = time.time() - t0   # decode_file sees only the file
        exact = dict(
            neuron_powers=all(np.array_equal(m["plates"][n][1], M[n + ".e"]) for n in NAMES),
            plate_biases=all(np.array_equal(m["bias"][n], M[n + ".b"]) for n in NAMES),
            head_bias=bool(np.array_equal(m["hb"], M["head.b"])), head_power=bool((M["head.e"] == m["he"]).all()),
            positions=bool(np.array_equal(m["wpe"], M["wpe"])),
            arith=all(np.array_equal(m["ar"][k], M[k]) for k in m["ar"]) and m["l768"] == int(M["log2_768"]) and m["eps"] == int(M["eps40"]),
            lens_equals_round_one=bool(np.array_equal(m["G"], np.load(SM + "/shrink/develop_gain24.npy"))))
        changed_pl = sum(int((m["plates"][n][0] != M[n + ".w"]).sum()) for n in NAMES); tot_pl = sum(m["plates"][n][0].size for n in NAMES)
        E0 = M["wte"]; changed_E = int((m["D"] != E0).sum()); del E0
        hp = hashlib.sha256()
        for n in NAMES: hp.update(m["plates"][n][0].tobytes())
        ht = hashlib.sha256()
        for r in range(0, m["D"].shape[0], 4096): ht.update(np.ascontiguousarray(m["D"][r:r + 4096], np.int32).tobytes())
        assert len(files) == 1, "one file per process (memory): the originals are not kept for restoring"
        for n in NAMES: tb.W[n] = None                            # (memory) the original plates out first
        try:
            for n in NAMES: tb.W[n] = (m["plates"][n][0].astype(np.float64), m["plates"][n][1], m["bias"][n]); m["plates"][n] = None
            tb.W["head"] = (None, np.full(m["rows"], m["he"], np.int64), m["hb"])
            D, Tp = m["D"], m["Tp"]; del m["plates"]; m["D"] = m["Tp"] = None
            wd = int(tb.words(entry=D, top=Tp))
            tb.W["head"] = (Tp.astype(np.int64).T.astype(np.float64), np.full(m["rows"], m["he"], np.int64), m["hb"])
            D64 = D.astype(np.int64)
            res = [tb._run(ids, D64) == t for ids, t in zip(tb.inputs, tb._TRUTH)]
            miss = [i for i, ok in enumerate(res) if not ok]; del D64
        finally:
            pass
        rec = dict(file=os.path.basename(path), file_bytes=m["file_bytes"], MB=round(m["file_bytes"] / 1e6, 3),
                   sections=m["sections"], sum_sections_plus_magic=sum(m["sections"].values()) + 4,
                   A_per_plate=m["A"], c=m["c"], cap=m["cap"], half=m["half"], nbits=m["nbits"], lim=m["lim"],
                   exposure_hist=m["exposure_hist"], Tp_range=m["Tp_range"], exact=exact,
                   plates_changed_frac=round(changed_pl / tot_pl, 4), entry_changed_frac=round(changed_E / D.size, 4),
                   plates_sha=hp.hexdigest()[:16], entry_sha=ht.hexdigest()[:16], words=wd, words_recount=30 - len(miss),
                   missed=miss, decode_secs=round(tdec, 1))
        print(json.dumps(rec), flush=True)
        del D, Tp, m
