"""PHOTONS family: store ONLY the photons (lit places) of the DEVELOPED values, never the dark places.

The developed values are exactly round one's:
  plates  v12 A=8 : read back from round one's own file (scratchpad/plates_v12_A8.bin) with round one's decoder
                    (plates_rc.dylib), so they are the very numbers that gave 30/30. Each plate also carries its
                    neurons' powers of two e_j, step terms t_j and line offsets off_i (side information, counted).
  tables  c=32/34 : E developed for the lens (develop_lib.develop_lines(E, c, 12)), stored as Q = D >> e_i per line;
                    T is printed from D through the 768 gains (2,548 bytes). Checked by the hash round one judged.

Every coder here is in photons_rc.c: whole numbers only (shifts, adds, compares, ANDs, table reads, and the range
coder's one whole-number product). Every stored stream is decoded back and checked np.array_equal.
The machine (table_bits) is NOT loaded here; only the npz arrays and round one's files.
"""
import os, sys, ctypes, struct, json, hashlib, time, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = "/Users/abundancemachine/streaming-machine/base2_gpt2.npz"
SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/"
OUTJ = os.path.join(HERE, "photons_results.json")
NAMES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
CH = 4096

# ------------------------------------------------------------------ whole-number helpers
_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)
def popcount(x):
    x = np.asarray(x, dtype=np.int64); n = np.zeros(x.shape, np.int64)
    while x.any(): n += _POP[x & 0xFFFF]; x = x >> 16
    return n
def naf_split(a):                                       # a >= 0 -> (up, hole) masks, a = up - hole
    a = np.asarray(a, dtype=np.int64); ah = a >> 1; a3 = a + ah; c = ah ^ a3
    return a3 & c, ah & c
def photons_plain(V): return popcount(np.abs(np.asarray(V, np.int64)))
def photons_wave(V):
    p, m = naf_split(np.abs(np.asarray(V, np.int64))); return popcount(p | m)

# ------------------------------------------------------------------ round one's plates file -> the developed plates
_prc = ctypes.CDLL(os.path.join(HERE, "plates_rc.dylib"))
_prc.plate_decode.restype = ctypes.c_int
_prc.plate_decode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int]
_prc.plate_encode.restype = ctypes.c_int64
_prc.plate_encode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
HS = struct.calcsize("<BHHbhhI")

def read_round_one_plates(path=SP + "plates_v12_A8.bin"):
    """-> list of dicts: name, R, C, A, e (neuron powers), t, off (step terms), v (stored whole numbers, int32),
    head (the bytes of header + e + t + off: the side information round one stores), nbytes_coded"""
    data = open(path, "rb").read(); p = 0; out = []
    for _ in NAMES:
        idx, R, C, A, tb, ob, nb = struct.unpack_from("<BHHbhhI", data, p); head = data[p:p + HS + 2 * C + R]; p += HS
        e = np.frombuffer(data, np.int8, C, p).astype(np.int64); p += C
        t = np.frombuffer(data, np.uint8, C, p).astype(np.int64) + tb; p += C
        off = np.frombuffer(data, np.uint8, R, p).astype(np.int64) + ob; p += R
        b = np.frombuffer(data[p:p + nb], np.uint8).copy(); v = np.zeros((R, C), np.int32)
        assert _prc.plate_decode(b.ctypes.data, nb, R, C, v.ctypes.data, 1) == 0
        p += nb
        out.append(dict(name=NAMES[idx], R=R, C=C, A=A, e=e, t=t, off=off, v=v, head=head, coded=nb))
    assert p == len(data)
    return out

def steps(t, off, A): return np.maximum(0, (t[None, :] + off[:, None] - A) >> 1)

def exposure_residual(t, off, A):
    """what the stored exposure says about each stored value's brightness, in half places: H - 2s (s = max(0, H>>1)),
    H = t_j + off_i - A. When the step is above 0 this is 0 or 1 (the meter's half place); when the step is 0 it is
    H itself (negative: the value is darker than the meter asked for). Whole numbers, known to the reader."""
    H = t[None, :] + off[:, None] - A
    return H - 2 * np.maximum(0, H >> 1)

# ------------------------------------------------------------------ the tables: developed for the lens
def tables_Q(c):
    """-> (Q int32 (50257,768) = D >> e, e (768,), sha of D as round one hashed it)"""
    sys.path.insert(0, HERE)
    import develop_lib as dl
    M = np.load(NPZ); E = np.asarray(M["wte"], dtype=np.int64); del M
    D, e = dl.develop_lines(E, c, 12); del E
    h = dl.sha(D)
    Q = np.empty(D.shape, np.int32)
    for r in range(0, D.shape[0], CH):
        d = D[r:r + CH].astype(np.int64); q = d >> e[None, :]; assert np.array_equal(q << e[None, :], d); Q[r:r + CH] = q
    return Q, e, h

ROUND1_TABLE_HASH = {32: "891b9f2ae87c71b1", 34: "374901a5776a3f25", 37: "02d5ee9e1092a717"}
ROUND1_TABLE_BYTES = {32: 34638171 + 2548, 34: 24961571 + 2548, 37: 14995130 + 2548}   # E coded + side, + the 768 gains

def log(rec):
    try: allr = json.load(open(OUTJ))
    except Exception: allr = []
    allr.append(rec); json.dump(allr, open(OUTJ, "w"), indent=1)
    print(json.dumps(rec), flush=True)

# ------------------------------------------------------------------ the photon coders (photons_rc.c)
_ph = ctypes.CDLL(os.path.join(HERE, "photons_rc.dylib"))
_ph.ph_encode.restype = ctypes.c_int64
_ph.ph_encode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
_ph.ph_decode.restype = ctypes.c_int
_ph.ph_decode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
CODER = {"fax": 0, "elias": 1, "rice": 2, "pattern": 3, "pages": 4}

def opts(coder="fax", wave=1, meters=1, clsmode=0, ncls=1, A=0, signcls=0, lim=11, tbits=8, lowcls=0, rowreset=0):
    return np.array([CODER[coder], wave, meters, clsmode, ncls, A, signcls, lim, tbits, lowcls, rowreset] + [0] * 5, np.int32)

def _arr(x, n):
    return np.ascontiguousarray(np.zeros(n, np.int32) if x is None else np.asarray(x, np.int32))

def ph_encode(v, o, colcls=None, colterm=None, rowterm=None):
    v = np.ascontiguousarray(v, dtype=np.int32); R, C = v.shape
    cc, ct, rt = _arr(colcls, C), _arr(colterm, C), _arr(rowterm, R)
    cap = v.size * 8 + 65536; out = np.zeros(cap, np.uint8)
    n = _ph.ph_encode(v.ctypes.data, R, C, o.ctypes.data, cc.ctypes.data, ct.ctypes.data, rt.ctypes.data, out.ctypes.data, cap)
    assert n > 0, n
    return out[:n].tobytes()

def ph_decode(buf, R, C, o, colcls=None, colterm=None, rowterm=None):
    cc, ct, rt = _arr(colcls, C), _arr(colterm, C), _arr(rowterm, R)
    b = np.frombuffer(buf, np.uint8).copy(); v = np.zeros((R, C), np.int32)
    assert _ph.ph_decode(b.ctypes.data, len(buf), R, C, o.ctypes.data, cc.ctypes.data, ct.ctypes.data, rt.ctypes.data, v.ctypes.data) == 0
    return v

def ph_store(v, o, **kw):
    """real bytes of v through one photon coder; decoded back and checked np.array_equal"""
    buf = ph_encode(v, o, **kw); back = ph_decode(buf, *v.shape, o, **kw)
    return len(buf), bool(np.array_equal(back, v))

def count_photons(v):
    """(plain photons, single-wave photons) of a whole-number array, summed, a slice at a time"""
    v = np.asarray(v); pp = pw = 0
    for r in range(0, v.shape[0], 1024):
        a = np.abs(v[r:r + 1024].astype(np.int64)); pp += int(popcount(a).sum()); p, m = naf_split(a); pw += int(popcount(p | m).sum())
    return pp, pw
