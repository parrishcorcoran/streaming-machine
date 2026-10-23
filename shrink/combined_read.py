"""COMBINED -- the reader. Sees ONLY the machine file (combined_store.py's layout) and rebuilds the whole machine from it:
the 48 developed plates (their steps from the stored terms), the plates' biases and powers of two, the developed entry
table, the top table PRINTED from it through the stored lens, the head's bias and power of two, the position table and
the arithmetic tables. It does not import the writer or any developing code; it uses the two range decoders
(plates_rc.dylib, combined_rc.dylib, decode only) and its own whole-number steps.
Every exact part is compared equal to the original machine (base2_gpt2.npz). The developed parts are installed and
the harness's own door, words(entry=D, top=T), gives the right next word out of 30.
    python3 shrink/combined_read.py FILE [FILE ...]"""
import sys, os, struct, ctypes, lzma, json, time, hashlib, numpy as np
SM = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, SM); os.chdir(SM)
import table_bits as tb
tb.E = None; tb.T = None                                  # the reader brings its own tables
HERE = SM + "/shrink"
NAMES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
P_ = ctypes.c_void_p
plib = ctypes.CDLL(HERE + "/plates_rc.dylib")
plib.plate_decode.restype = ctypes.c_int
plib.plate_decode.argtypes = [P_, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, P_, ctypes.c_int]
dlib = ctypes.CDLL(HERE + "/combined_rc.dylib"); dlib.crc_decode.restype = ctypes.c_int64

def pdec(buf, R, C):
    b = np.frombuffer(buf, np.uint8).copy(); v = np.zeros((R, C), np.int32)
    assert plib.plate_decode(b.ctypes.data, len(buf), R, C, v.ctypes.data, 1) == 0
    return v

def ldec(data, n, nbits, lim):                            # line-context symbols, line = i mod 768
    ctx = np.tile(np.arange(768, dtype=np.uint16), n // 768); buf = np.frombuffer(data, np.uint8).copy()
    sym = np.empty(n, np.uint16)
    dlib.crc_decode(buf.ctypes.data_as(P_), ctypes.c_int64(len(buf)), ctx.ctypes.data_as(P_), ctypes.c_int64(n),
                    ctypes.c_int(nbits), ctypes.c_int64(768), ctypes.c_int(lim), sym.ctypes.data_as(P_))
    return sym

def blen(x):                                              # place of the top 1, plus one
    x = np.abs(np.asarray(x, np.int64)).copy(); n = np.zeros(x.shape, np.int64)
    while (x > 0).any(): n += (x > 0); x >>= 1
    return n

def print_top(G, D):                                      # T from D through the lens: whole numbers only
    out = np.empty(D.shape, np.int32)
    for r in range(0, D.shape[0], 4096):
        sv = D[r:r + 4096].astype(np.int64) * G[None, :]
        c = 768 * sv - sv.sum(1, keepdims=True); y = (c + (1 << 23)) >> 24
        out[r:r + 4096] = (y * 1431655766 + (1 << 43)) >> 44
    return out

def read(path):
    data = open(path, "rb").read(); assert data[:4] == b"CMF1"; p = 4; S = {}
    while p < len(data):
        tag = data[p:p + 4]; ln = struct.unpack_from("<I", data, p + 4)[0]; S[tag.decode()] = data[p + 8:p + 8 + ln]; p += 8 + ln
    assert p == len(data)
    M = tb.M; out = {}
    # PLAT
    d = S["PLAT"]; q = 0; plates = {}; hs = struct.calcsize("<BHHbhhI")
    for _ in NAMES:
        idx, R, C, A, tb0, ob0, nb = struct.unpack_from("<BHHbhhI", d, q); q += hs
        e = np.frombuffer(d, np.int8, C, q).astype(np.int64); q += C
        t = np.frombuffer(d, np.uint8, C, q).astype(np.int64) + tb0; q += C
        off = np.frombuffer(d, np.uint8, R, q).astype(np.int64) + ob0; q += R
        v = pdec(d[q:q + nb], R, C).astype(np.int64); q += nb
        s = np.maximum(0, (t[None, :] + off[:, None] - A) >> 1)
        w = np.clip(v << s, -32768, 32767).astype(np.int16)
        plates[NAMES[idx]] = (w, e)
    assert q == len(d)
    # BIAS
    d = S["BIAS"]; q = 0; bias = {}
    for n in NAMES:
        ln = struct.unpack_from("<I", d, q)[0]; q += 4
        bias[n] = pdec(d[q:q + ln], 1, plates[n][0].shape[1])[0].astype(np.int64); q += ln
    # TABL
    d = S["TABL"]; c, cap, half, w, lim, rows, nbq = struct.unpack_from("<BBBBBII", d, 0); q = struct.calcsize("<BBBBBII")
    if half: e = np.frombuffer(d, np.uint8, 768, q).astype(np.int64); q += 768
    lg = struct.unpack_from("<I", d, q)[0]; q += 4
    G = np.frombuffer(lzma.decompress(d[q:q + lg]), np.int32).astype(np.int64); q += lg
    if not half: e = np.minimum(np.maximum(c - (blen(G) - 1), 0), cap)
    z = ldec(d[q:q + nbq], rows * 768, w, lim).reshape(rows, 768); q += nbq; assert q == len(d)
    D = np.empty((rows, 768), np.int32)
    for r in range(0, rows, 4096):                         # a slice at a time (memory)
        zz = z[r:r + 4096].astype(np.int64); D[r:r + 4096] = ((zz >> 1) ^ (-(zz & 1))) << e[None, :]
    del z
    Tp = print_top(G, D)
    # HEAD
    d = S["HEAD"]; ln = struct.unpack_from("<I", d, 0)[0]
    hbias = pdec(d[4:4 + ln], 1, rows)[0].astype(np.int64); he = struct.unpack_from("<b", d, 4 + ln)[0]
    # POSN
    d = S["POSN"]; R, C = struct.unpack_from("<HH", d, 0); wpe = np.cumsum(pdec(d[4:], R, C).astype(np.int64), axis=0)
    # ARIT
    d = S["ARIT"]; q = 0; ar = {}
    for kname in ("phi", "arrival", "log2", "exp2"):
        a0, n, ln = struct.unpack_from("<qII", d, q); q += 16
        ar[kname] = np.concatenate([[a0], a0 + np.cumsum(pdec(d[q:q + ln], 1, n - 1)[0].astype(np.int64))]); q += ln
    l768, eps = struct.unpack_from("<qq", d, q); q += 16; assert q == len(d)
    # the exact parts must be the machine's own, number for number
    exact = dict(
        neuron_powers=all(np.array_equal(plates[n][1], M[n + ".e"]) for n in NAMES),
        plate_biases=all(np.array_equal(bias[n], M[n + ".b"]) for n in NAMES),
        head_bias=bool(np.array_equal(hbias, M["head.b"])), head_power=bool((M["head.e"] == he).all()),
        position_table=bool(np.array_equal(wpe, M["wpe"])),
        arith_tables=all(np.array_equal(ar[k], M[k]) for k in ar) and l768 == int(M["log2_768"]) and eps == int(M["eps40"]),
        lens=bool(np.array_equal(G, np.load(HERE + "/develop_gain24.npy"))))
    hp = hashlib.sha256()
    for n in NAMES: hp.update(plates[n][0].tobytes())
    ht = hashlib.sha256()
    for r in range(0, D.shape[0], 4096): ht.update(np.ascontiguousarray(D[r:r + 4096], np.int32).tobytes())
    return dict(plates=plates, bias=bias, D=D, Tp=Tp, hbias=hbias, exact=exact, c=c, cap=cap, half=half,
                plates_sha=hp.hexdigest()[:16], entry_sha=ht.hexdigest()[:16], sections={k: len(v) + 8 for k, v in S.items()},
                file_bytes=len(data))

if __name__ == "__main__":
    assert tb.words() == 30                               # the machine's own answers first
    tb._WTE0 = None                                       # (memory) every judge below brings its own entry table
    tb.W["head"] = (None, tb.W["head"][1], tb.W["head"][2])   # (memory) and its own top table
    for path in sys.argv[1:]:
        t0 = time.time(); m = read(path)
        for n in NAMES:                                    # the developed plates, their powers and biases go in
            tb.W[n] = (m["plates"][n][0].astype(np.float64), m["plates"][n][1], m["bias"][n])
        hw = tb.W["head"]; tb.W["head"] = (hw[0], hw[1], m["hbias"])
        D, Tp = m["D"], m["Tp"]; del m["plates"], m["D"], m["Tp"]
        try:
            wd = int(tb.words(entry=D, top=Tp))
        finally:
            for n in NAMES: tb.W[n] = (tb.M[n + ".w"].astype(np.float64), tb.M[n + ".e"], tb.M[n + ".b"])
            tb.W["head"] = hw
        del D, Tp
        rec = dict(version=f"READ {os.path.basename(path)}", file=path, file_bytes=m["file_bytes"],
                   total_MB=round(m["file_bytes"] / 1e6, 3), section_bytes=m["sections"], exact_parts_equal=m["exact"],
                   plates_sha=m["plates_sha"], entry_sha=m["entry_sha"], c=m["c"], cap=m["cap"], half=m["half"],
                   words=wd, decoded_from_file_only=True, secs=round(time.time() - t0, 1))
        print(json.dumps(rec), flush=True)
        allr = json.load(open(HERE + "/combined_results.json")); allr.append(rec)
        json.dump(allr, open(HERE + "/combined_results.json", "w"), indent=1)
