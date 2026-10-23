"""CHECK of the PLATES family's best claim, independent of plates_lib / plates_best_store.
Reads each claimed file with its own parser (nothing from the encoder's process), decodes the photon streams with
the family's range decoder (plates_rc.dylib, decode only), rebuilds steps and plates with its own whole-number code,
and asks the machine for its words through the harness's own words(plates=...) door.
For the v12 file it also rebuilds the v12 development from the original plates with its own half-place meter
(bit lengths, sums, shifts) and checks the file holds exactly that.  Then a lossless round trip of v1 (identity).
    python3 shrink/check_plates_file.py
"""
import sys, os, struct, ctypes, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine")
import table_bits as tb
tb.E = None; tb.T = None                                  # the word tables are not used here
HERE = os.path.dirname(os.path.abspath(__file__))
SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad"
NAMES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
NVAL = sum(tb.plate(n).size for n in NAMES)
lib = ctypes.CDLL(os.path.join(HERE, "plates_rc.dylib"))
lib.plate_decode.restype = ctypes.c_int
lib.plate_decode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int]
lib.plate_encode.restype = ctypes.c_int64
lib.plate_encode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]

def dec(buf, R, C):
    b = np.frombuffer(buf, np.uint8).copy(); v = np.zeros((R, C), np.int32)
    assert lib.plate_decode(b.ctypes.data, len(buf), R, C, v.ctypes.data, 1) == 0
    return v

def enc(v):
    v = np.ascontiguousarray(v, np.int32); cap = v.size * 5 + 4096; out = np.zeros(cap, np.uint8)
    n = lib.plate_encode(v.ctypes.data, v.shape[0], v.shape[1], out.ctypes.data, cap, 1); assert n > 0
    return out[:n].tobytes()

def bitlen(x):                                            # my own: count of shifts until zero
    x = np.abs(np.asarray(x, np.int64)).copy(); n = np.zeros(x.shape, np.int64)
    while (x > 0).any(): n += (x > 0); x >>= 1
    return n

def hp(x):                                                # half place: 2*(top place) + next place down
    x = np.asarray(x, np.int64); n = bitlen(x)
    nx = np.where(n >= 2, (x >> np.maximum(n - 2, 0)) & 1, 0)
    return np.where(x > 0, 2 * (n - 1) + nx, -1000)

def most_common(a):
    u, c = np.unique(a, return_counts=True); return int(u[np.argmax(c)])

def naf_photons(v):
    a = np.abs(v.astype(np.int64)); t = a ^ (3 * a); n = 0
    while (t > 0).any(): n += int((t & 1).sum()); t >>= 1
    return n

def read_file(path, check_v12=False, check_nearest=True):
    data = open(path, "rb").read(); p = 0; HS = struct.calcsize("<BHHbhhI")
    plates = {}; parts = {"header": 0, "neuron_power": 0, "neuron_term": 0, "line_offset": 0, "photons": 0}
    ph = 0; seen = set(); mism = 0
    while p < len(data):
        idx, R, C, A, tbase, obase, nb = struct.unpack_from("<BHHbhhI", data, p); p += HS; parts["header"] += HS
        e = np.frombuffer(data, np.int8, C, p).astype(np.int64); p += C; parts["neuron_power"] += C
        t = np.frombuffer(data, np.uint8, C, p).astype(np.int64) + tbase; p += C; parts["neuron_term"] += C
        off = np.frombuffer(data, np.uint8, R, p).astype(np.int64) + obase; p += R; parts["line_offset"] += R
        v = dec(data[p:p + nb], R, C); p += nb; parts["photons"] += nb
        n = NAMES[idx]; seen.add(n); q = tb.plate(n).astype(np.int64)
        assert q.shape == (R, C) and np.array_equal(e, tb.M[n + ".e"].astype(np.int64)), n
        s = np.maximum(0, (t[None, :] + off[:, None] - A) >> 1)
        if check_nearest:                                 # the stored numbers are the nearest development at those steps
            half = np.where(s > 0, np.left_shift(np.int64(1), np.maximum(s - 1, 0)), 0)
            if not np.array_equal(v.astype(np.int64), (q + half) >> s): mism += 1
        if check_v12:                                     # the steps are v12's rule, rebuilt from the original plate
            qa = np.abs(q); t12 = hp(qa.sum(0)) + 2 * (2 - bitlen(np.array(R)))
            r12 = hp(qa.sum(1)); o12 = r12 - most_common(r12)
            assert np.array_equal(t, t12) and np.array_equal(off, o12), n
        w = np.clip(np.left_shift(v.astype(np.int64), s), -32768, 32767).astype(np.int16)
        plates[n] = w; ph += naf_photons(v)
    assert p == len(data) and seen == set(NAMES)
    return plates, parts, ph, mism, len(data)

out = {}
assert tb.words() == 30
for label, path, c12, cn in [("v12 A=8", "plates_v12_A8.bin", True, True), ("v14 A=6", "plates_v14_A6.bin", False, True),
                             ("v14 A=6 + 2 photons", "plates_v14_A6_p_2_photons.bin", False, False)]:
    t0 = time.time()
    plates, parts, ph, mism, size = read_file(os.path.join(SP, path), c12, cn)
    changed = sum(int((plates[n] != tb.plate(n)).sum()) for n in NAMES)
    wd = int(tb.words(plates=plates)); del plates
    rec = {"file": path, "file_bytes": size, "parts_bytes": parts, "sum_parts": sum(parts.values()),
           "MB": round(size / 1e6, 3), "bits_per_weight": round(8 * size / NVAL, 4),
           "single_wave_photons_per_weight": round(ph / NVAL, 3), "nearest_mismatch_plates": mism if cn else None,
           "changed_share": round(changed / NVAL, 4), "words": wd, "secs": round(time.time() - t0, 1)}
    out[label] = rec; print(label, json.dumps(rec), flush=True)

# lossless v1: identity through the same coder, decoded back and compared; side = 1 byte/neuron power + 4 bytes/plate
t0 = time.time(); tot = 0; ok = True; ncols = 0
for n in NAMES:
    q = tb.plate(n).astype(np.int32); b = enc(q); tot += len(b); ncols += q.shape[1]
    ok &= bool(np.array_equal(dec(b, *q.shape), q))
side = ncols + 4 * len(NAMES)
out["v1 lossless"] = {"coded_bytes": tot, "side_bytes": side, "MB": round((tot + side) / 1e6, 3),
                      "bits_per_weight": round(8 * (tot + side) / NVAL, 4), "roundtrip_ok": ok, "secs": round(time.time() - t0, 1)}
print("v1", json.dumps(out["v1 lossless"]), flush=True)
json.dump(out, open(os.path.join(HERE, "check_plates_file.json"), "w"), indent=1)
