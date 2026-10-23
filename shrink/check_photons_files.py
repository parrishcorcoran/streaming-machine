"""CHECKER for the photons family's files. Reads each file with its own parser and its own build of the coder
(scratchpad/chk_photons/photons_rc_chk.dylib, compiled from photons_rc.c); the reader sees only the file.
Then (after decoding) compares to round one's developed values, and asks the machine for the words.
    python3 shrink/check_photons_files.py
"""
import os, sys, struct, lzma, ctypes, time, json, numpy as np

SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/"
LIB = ctypes.CDLL(SP + "chk_photons/photons_rc_chk.dylib")
LIB.ph_decode.restype = ctypes.c_int
LIB.ph_decode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
NAMES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
HS = struct.calcsize("<BHHbhhI")


def _dec(buf, R, C, o, colcls=None, colterm=None, rowterm=None):
    z = lambda x, n: np.ascontiguousarray(np.zeros(n, np.int32) if x is None else np.asarray(x, np.int32))
    cc, ct, rt = z(colcls, C), z(colterm, C), z(rowterm, R)
    b = np.frombuffer(buf, np.uint8).copy(); v = np.zeros((R, C), np.int32); oo = np.ascontiguousarray(o, np.int32)
    rc = LIB.ph_decode(b.ctypes.data, len(buf), R, C, oo.ctypes.data, cc.ctypes.data, ct.ctypes.data, rt.ctypes.data, v.ctypes.data)
    assert rc == 0, rc
    return v


def decode_plates(path):
    """yields (name, w int16 plate, e, v) from the file alone"""
    data = open(path, "rb").read(); assert data[:4] == b"PHP1"
    o = np.frombuffer(data[4:36], np.int16).astype(np.int32).copy(); p = 36; seen = []
    while p < len(data):
        idx, R, C, A, tb, ob, nb = struct.unpack_from("<BHHbhhI", data, p); p += HS
        e = np.frombuffer(data, np.int8, C, p).astype(np.int64); p += C
        t = np.frombuffer(data, np.uint8, C, p).astype(np.int64) + tb; p += C
        off = np.frombuffer(data, np.uint8, R, p).astype(np.int64) + ob; p += R
        op = o.copy(); op[5] = A
        v = _dec(data[p:p + nb], R, C, op, colterm=t, rowterm=off); p += nb
        s = np.maximum(0, (t[None, :] + off[:, None] - A) >> 1)
        w = np.clip(np.left_shift(v.astype(np.int64), s), -32768, 32767).astype(np.int16)
        seen.append(NAMES[idx])
        yield NAMES[idx], w, e, v
    assert p == len(data) and seen == NAMES


def decode_tables(path):
    """-> (D entry table, T printed top table, c, cap, e, G) from the file alone"""
    data = open(path, "rb").read(); assert data[:4] == b"PHT1"
    o = np.frombuffer(data[4:36], np.int16).astype(np.int32).copy(); p = 36
    c, cap, R, C = struct.unpack_from("<BBIH", data, p); p += struct.calcsize("<BBIH")
    e = np.frombuffer(data, np.uint8, C, p).astype(np.int64); p += C
    (ng,) = struct.unpack_from("<I", data, p); p += 4
    G = np.frombuffer(lzma.decompress(data[p:p + ng]), np.int32).astype(np.int64); p += ng
    (nb,) = struct.unpack_from("<I", data, p); p += 4
    Q = _dec(data[p:p + nb], R, C, o, colcls=np.arange(C)); p += nb
    assert p == len(data)
    D = np.empty((R, C), np.int32); T = np.empty((R, C), np.int32)
    for r in range(0, R, 4096):
        d = Q[r:r + 4096].astype(np.int64) << e[None, :]; D[r:r + 4096] = d
        sv = d * G[None, :]; cc = 768 * sv - sv.sum(1, keepdims=True); y = (cc + (1 << 23)) >> 24
        assert int(np.abs(y).max()) < (1 << 32)
        T[r:r + 4096] = (y * 1431655766 + (1 << 43)) >> 44
    del Q
    return D, T, c, cap, e, G


if __name__ == "__main__":
    out = {}
    PF = {"v12_A8": SP + "photons_plates_v12_A8.php", "v14_A6": SP + "photons_plates_v14_A6.php"}
    TF = {32: SP + "photons_tables_c32.pht", 34: SP + "photons_tables_c34.pht"}
    for k, f in list(PF.items()) + list(TF.items()): out[f"bytes {os.path.basename(f)}"] = os.path.getsize(f)
    print(json.dumps(out), flush=True)

    sys.path.insert(0, "/Users/abundancemachine/streaming-machine"); sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
    os.chdir("/Users/abundancemachine/streaming-machine")
    import table_bits as tb
    w0 = int(tb.words()); print("unchanged machine:", w0, flush=True)
    print("head.e unique:", np.unique(tb.M["head.e"]).tolist(), " head.b nonzero:", int((tb.M["head.b"] != 0).sum()), flush=True)
    tb.E = None; tb.T = None; tb._WTE0 = tb._WTE0.astype(np.int32)
    import gc; gc.collect()
    _prc = ctypes.CDLL("/Users/abundancemachine/streaming-machine/shrink/plates_rc.dylib")
    _prc.plate_decode.restype = ctypes.c_int
    _prc.plate_decode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int]

    def round_one_plates(path):            # round one's file through round one's own reader (after our decode)
        data = open(path, "rb").read(); p = 0
        for _ in NAMES:
            idx, R, C, A, tb_, ob, nb = struct.unpack_from("<BHHbhhI", data, p); p += HS
            e = np.frombuffer(data, np.int8, C, p).astype(np.int64); p += C
            t = np.frombuffer(data, np.uint8, C, p).astype(np.int64) + tb_; p += C
            off = np.frombuffer(data, np.uint8, R, p).astype(np.int64) + ob; p += R
            b = np.frombuffer(data[p:p + nb], np.uint8).copy(); v = np.zeros((R, C), np.int32)
            assert _prc.plate_decode(b.ctypes.data, nb, R, C, v.ctypes.data, 1) == 0; p += nb
            s = np.maximum(0, (t[None, :] + off[:, None] - A) >> 1)
            yield NAMES[idx], np.clip(np.left_shift(v.astype(np.int64), s), -32768, 32767).astype(np.int16)

    for c in (34, 32):
        t0 = time.time(); D, T, cc, cap, e, G = decode_tables(TF[c]); dt = round(time.time() - t0, 1)
        import develop_lib as dl
        E0 = np.asarray(tb.M["wte"], dtype=np.int64); D0, e0 = dl.develop_lines(E0, c, 12); del E0
        sameD = bool(np.array_equal(D, D0)); sameT = bool(np.array_equal(T, dl.print_top(D0))); del D0
        rec = {"table": c, "decode_secs": dt, "c_cap": [cc, cap], "D_equals_round_one": sameD, "T_equals_print_top": sameT,
               "sha": dl.sha(D)}
        rec["words_tables_alone"] = int(tb.words(entry=D, top=T))
        for lab, pf in PF.items():
            t1 = time.time(); same_r1 = True
            r1 = dict(round_one_plates(SP + f"plates_{lab}.bin"))
            for n, w, e_, v in decode_plates(pf):
                assert np.array_equal(e_, tb.M[n + ".e"].astype(np.int64))
                same_r1 &= bool(np.array_equal(w, r1[n]))
                tb.W[n] = (w.astype(np.float64), tb.W[n][1], tb.W[n][2])
            del r1
            rec[f"{lab} plates_equal_round_one"] = same_r1
            rec[f"{lab} words_plates_alone"] = int(tb.words())
            rec[f"{lab} + c{c} words"] = int(tb.words(entry=D, top=T))
            rec[f"{lab} + c{c} bytes"] = os.path.getsize(pf) + os.path.getsize(TF[c])
            rec[f"{lab} secs"] = round(time.time() - t1, 1)
            for n in NAMES: tb.W[n] = (tb.M[n + ".w"].astype(np.float64), tb.W[n][1], tb.W[n][2])
            print(json.dumps(rec), flush=True)
        del D, T; gc.collect()
    print("restored machine:", int(tb.words()), flush=True)
