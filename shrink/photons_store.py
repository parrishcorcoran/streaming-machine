"""PHOTONS family, the winners written to ONE real file each and read back by a reader that sees only that file.
  plates  (PHP1): a 32-byte head of coder settings; per plate round one's own side information (header, each neuron's
          power of two e_j, step terms t_j, line offsets off_i) and the photon-coded stored numbers: each value's
          whole photon pattern as one symbol down an 8-place tree, read against the two exposure meters and the
          exposure residual H - 2s that the reader computes from t_j, off_i and A.
  tables  (PHT1): a 32-byte head of coder settings, c, the cap, the shape, the 768 line cut places, the 768 lens gains
          (lzma), and the photon-coded Q = D >> e: each value's whole photon pattern as one symbol down a 10-place
          tree, one set of chances per line. The reader rebuilds D = Q << e and prints T from D through the gains.
The file size IS the stored size. Every read-back is checked np.array_equal against the developed values.
    python3 shrink/photons_store.py
"""
import lzma
from photons_lib import *

PLATE_OPTS = dict(coder="pattern", wave=1, meters=1, clsmode=2, ncls=5, tbits=8)
TABLE_OPTS = dict(coder="pattern", wave=1, meters=0, clsmode=1, ncls=768, tbits=10)

def _opt_bytes(o): return np.asarray(o[:16], np.int16).tobytes()      # 32 bytes of settings
def _opt_from(b): return np.frombuffer(b, np.int16, 16).astype(np.int32).copy()

def write_plates(P, path):
    o = opts(**PLATE_OPTS)
    with open(path, "wb") as f:
        f.write(b"PHP1"); f.write(_opt_bytes(o))
        for p in P:
            op = o.copy(); op[5] = p["A"]
            buf = ph_encode(p["v"], op, colterm=p["t"], rowterm=p["off"])
            tb = int(p["t"].min()); ob = int(p["off"].min())
            f.write(struct.pack("<BHHbhhI", NAMES.index(p["name"]), p["R"], p["C"], p["A"], tb, ob, len(buf)))
            f.write(p["e"].astype(np.int8).tobytes()); f.write((p["t"] - tb).astype(np.uint8).tobytes())
            f.write((p["off"] - ob).astype(np.uint8).tobytes()); f.write(buf)
    return os.path.getsize(path)

def read_plates(path): return dict(iter_plates(path))

def iter_plates(path):
    """yields (name, (v stored whole numbers, s steps, e neuron powers)) one plate at a time, from the file alone"""
    data = open(path, "rb").read(); assert data[:4] == b"PHP1"; o = _opt_from(data[4:36]); p = 36
    while p < len(data):
        idx, R, C, A, tb, ob, nb = struct.unpack_from("<BHHbhhI", data, p); p += HS
        e = np.frombuffer(data, np.int8, C, p).astype(np.int64); p += C
        t = np.frombuffer(data, np.uint8, C, p).astype(np.int64) + tb; p += C
        off = np.frombuffer(data, np.uint8, R, p).astype(np.int64) + ob; p += R
        op = o.copy(); op[5] = A
        v = ph_decode(data[p:p + nb], R, C, op, colterm=t, rowterm=off); p += nb
        yield NAMES[idx], (v, steps(t, off, A), e)
    assert p == len(data)

def write_tables(Q, e, c, cap, gains, path):
    o = opts(**TABLE_OPTS); R, C = Q.shape
    buf = ph_encode(Q, o, colcls=np.arange(C)); g = lzma.compress(np.asarray(gains, np.int32).tobytes(), preset=6)
    with open(path, "wb") as f:
        f.write(b"PHT1"); f.write(_opt_bytes(o)); f.write(struct.pack("<BBIH", c, cap, R, C))
        f.write(np.asarray(e, np.uint8).tobytes()); f.write(struct.pack("<I", len(g))); f.write(g)
        f.write(struct.pack("<I", len(buf))); f.write(buf)
    return os.path.getsize(path)

def print_top_from(D, G):
    """T printed from the entry table through the lens (develop_lib.print_top, with the gains read from the file)"""
    out = np.empty(D.shape, np.int32)
    for r in range(0, D.shape[0], CH):
        sv = D[r:r + CH].astype(np.int64) * G[None, :]
        cc = 768 * sv - sv.sum(1, keepdims=True); y = (cc + (1 << 23)) >> 24
        out[r:r + CH] = (y * 1431655766 + (1 << 43)) >> 44
    return out

def read_tables(path):
    """-> (D developed entry table, T printed top table, c) from the file alone"""
    data = open(path, "rb").read(); assert data[:4] == b"PHT1"; o = _opt_from(data[4:36]); p = 36
    c, cap, R, C = struct.unpack_from("<BBIH", data, p); p += struct.calcsize("<BBIH")
    e = np.frombuffer(data, np.uint8, C, p).astype(np.int64); p += C
    (ng,) = struct.unpack_from("<I", data, p); p += 4
    G = np.frombuffer(lzma.decompress(data[p:p + ng]), np.int32).astype(np.int64); p += ng
    (nb,) = struct.unpack_from("<I", data, p); p += 4
    Q = ph_decode(data[p:p + nb], R, C, o, colcls=np.arange(C)); p += nb; assert p == len(data)
    D = np.empty(Q.shape, np.int32)
    for r in range(0, R, CH): D[r:r + CH] = Q[r:r + CH].astype(np.int64) << e[None, :]
    del Q
    return D, print_top_from(D, G), c

if __name__ == "__main__":
    sys.path.insert(0, HERE); import develop_lib as dl
    for src, label in [(SP + "plates_v12_A8.bin", "v12_A8"), (SP + "plates_v14_A6.bin", "v14_A6")]:
        t0 = time.time(); P = read_round_one_plates(src); r1 = os.path.getsize(src)
        path = SP + f"photons_plates_{label}.php"; size = write_plates(P, path)
        back = read_plates(path); ok = all(np.array_equal(back[p["name"]][0], p["v"]) and np.array_equal(back[p["name"]][2], p["e"])
                                           and np.array_equal(back[p["name"]][1], steps(p["t"], p["off"], p["A"])) for p in P)
        pp = pw = 0
        for p in P: a, b = count_photons(p["v"]); pp += a; pw += b
        n = sum(p["v"].size for p in P)
        log({"version": f"FILE plates {label}", "file": path, "file_bytes": size, "total_MB": round(size / 1e6, 3),
             "bits_per_value": round(8 * size / n, 4), "bits_per_wave_photon": round(8 * size / pw, 4),
             "bits_per_plain_photon": round(8 * size / pp, 4), "round_one_file_bytes": r1,
             "minus_round_one_MB": round((size - r1) / 1e6, 3), "decoded_from_file_only": True, "roundtrip_ok": bool(ok),
             "secs": round(time.time() - t0, 1)})
        del P, back
    G = dl.gains()
    for c in (32, 34):
        t0 = time.time(); Q, e, h = tables_Q(c); assert h == ROUND1_TABLE_HASH[c]
        pp, pw = count_photons(Q)
        path = SP + f"photons_tables_c{c}.pht"; size = write_tables(Q, e, c, 12, G, path); del Q
        D, Tp, cc = read_tables(path)
        M = np.load(NPZ); E = np.asarray(M["wte"], dtype=np.int64); del M
        D0, e0 = dl.develop_lines(E, c, 12); del E
        okD = dl.sha(D) == h and np.array_equal(D, D0); okT = np.array_equal(Tp, dl.print_top(D0)); del D0
        log({"version": f"FILE tables c={c}", "file": path, "file_bytes": size, "total_MB": round(size / 1e6, 3),
             "bits_per_value": round(8 * size / D.size, 4), "bits_per_wave_photon": round(8 * size / pw, 4),
             "bits_per_plain_photon": round(8 * size / pp, 4), "round_one_bytes": ROUND1_TABLE_BYTES[c],
             "minus_round_one_MB": round((size - ROUND1_TABLE_BYTES[c]) / 1e6, 3), "decoded_from_file_only": True,
             "roundtrip_ok": bool(okD and okT), "T_printed_equal": bool(okT), "secs": round(time.time() - t0, 1)})
        del D, Tp
