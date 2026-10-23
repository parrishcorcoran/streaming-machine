"""COMBINED family: the whole developed machine at once -- the entry table E (exposed for the lens, T printed from it),
the 48 plates (each plate its own exposure A, rule v12 or v14 from round one), and every other stored number
(the plates' biases, the head's bias, the position table, the arithmetic tables, the lens) -- judged ONLY by
words(...) out of 30 on the whole machine, sized ONLY by real bytes that decode back.

Pieces reused unchanged from round one:
  plates : plates_best_store.terms / steps (the v12 / v14 meters, whole numbers), plates_lib.cut (nearest),
           plates_lib.encode / decode (plates_rc: the photon coder, whole numbers)
  tables : develop_lib.develop_lines(E, c, cap) (line i keeps places >= min(max(c - gainplace_i, 0), cap), nearest),
           develop_lib.print_top (T printed from the developed E through 768 whole-number gains)
Kept exact (lossless, decoded and compared equal): the plates' biases, each neuron's power of two, the head's bias,
the head's power of two, the position table, the arithmetic tables (bend share, arrival, log2, 2^x), two constants,
the lens gains.

A plate configuration is a dict {plate name: (rule, A)}. Sizes of each (plate, rule, A) are cached (bytes of the
coded photons + that plate's side numbers exactly as the file carries them), so a configuration's size is known
before it is judged.
"""
import os, sys, struct, json, time, lzma, numpy as np
SM = "/Users/abundancemachine/streaming-machine"
sys.path.insert(0, SM); sys.path.insert(0, SM + "/shrink"); os.chdir(SM)
_src = open(SM + "/shrink/plates_best_store.py").read()
exec(_src[:_src.index("assert words() == 30")])           # plates_lib + terms/steps/write/read_into_machine + NP
import develop_lib as dl
import table_bits as _tb

HERE = SM + "/shrink"
SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/"
OUTJ = HERE + "/combined_results.json"
CACHE = SP + "combined_plate_bytes.json"
KINDS = ("qkv", "proj", "fc", "mproj")
HS = struct.calcsize("<BHHbhhI")                          # 14 bytes: a plate's header in the file

# ------------------------------------------------------------------ plates
_cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

def dev_plate(n, rule, A):
    q = plate(n); t, off = terms(n, q, rule); s = steps(t, off, A)
    return cut(q, s), s

def plate_bytes(n, rule, A):
    """real bytes this plate takes in the file: header + neuron powers + neuron terms + line offsets + coded photons.
    The photons are coded once per (plate, rule, A), decoded and checked equal, and the count is cached."""
    key = f"{n}|{rule}|{A}"
    if key not in _cache:
        v, s = dev_plate(n, rule, A); buf = encode(v)
        assert np.array_equal(decode(buf, *v.shape), v)
        R, C = v.shape
        _cache[key] = HS + 2 * C + R + len(buf)
        json.dump(_cache, open(CACHE, "w"))
    return _cache[key]

def plates_size(cfg): return sum(plate_bytes(n, *cfg[n]) for n in NAMES)

def install_plates(cfg):
    for n in NAMES:
        rule, A = cfg[n]; v, s = dev_plate(n, rule, A); w = shift_back(v, s)
        _tb.W[n] = (w.astype(np.float64), _tb.W[n][1], _tb.W[n][2])

def cfg_all(rule="v12", A=8): return {n: (rule, A) for n in NAMES}

def kind(n): return n.split(".")[0]
def layer(n): return int(n.split(".")[1])

def cfg_by(fn):
    """fn(name) -> (rule, A)"""
    return {n: fn(n) for n in NAMES}

# ------------------------------------------------------------------ tables (E developed, T printed)
assert int(np.abs(_tb._WTE0).max()) < (1 << 30)
_tb._WTE0 = _tb._WTE0.astype(np.int32)                   # same whole numbers in 32 bits (memory)
assert _tb.words() == 30                                  # the machine's own answers, before anything moves
E0 = _tb._WTE0                                            # the entry table, whole numbers at 2^16
_HEAD = _tb.W["head"]; _tb.W["head"] = (None, _HEAD[1], _HEAD[2]); _HEAD = (None, _HEAD[1], _HEAD[2])   # free 309 MB
G = dl.gains().astype(np.int64)
LENS_BYTES = len(lzma.compress(G.astype(np.int32).tobytes(), preset=9)) + 4
_tab = {}
TCACHE = SP + "combined_table_bytes.json"
_tcache = json.load(open(TCACHE)) if os.path.exists(TCACHE) else {}

def exposure(c, cap=12, half=None):
    """line i keeps places >= e_i = min(max(c - gainplace_i, 0), cap). half = a seed: a random half of the lines
    (fixed seed, no similarity) one place coarser -- the exposure c + 1/2 laid as a halftone across the lines."""
    h = 0 if half is None else (np.random.default_rng(half).permutation(768) < 384).astype(np.int64)
    return np.minimum(np.maximum(c - (dl.bitlen(G) - 1) + h, 0), cap).astype(np.int64)

def tables(c, cap=12, half=None):
    """the developed entry table D (int32) and the top table printed from it, laid out as the machine reads it
    (lines x words, float64 holding whole numbers exactly), kept for the last (c, cap) asked"""
    key = (c, cap, half)
    if key not in _tab:
        _tab.clear()
        if half is None:
            D, e = dl.develop_lines(E0, c, cap)
            assert np.array_equal(e, exposure(c, cap))
        else:
            D = develop_e(E0, exposure(c, cap, half))
        HT = np.empty((768, D.shape[0]), np.float64)
        for r in range(0, D.shape[0], dl.CH):
            HT[:, r:r + dl.CH] = dl.print_top(D[r:r + dl.CH]).T
        _tab[key] = (D, HT)
    return _tab[key]

def develop_e(A, e):
    """develop_lines with a given exposure per line (nearest, the same whole-number steps)"""
    q = (np.int64(1) << e)[None, :]; keep = ~(q - 1); out = np.empty(A.shape, np.int32)
    for r in range(0, A.shape[0], dl.CH):
        v = A[r:r + dl.CH].astype(np.int64); a = np.abs(v); a2 = (a + (q >> 1)) & keep
        out[r:r + dl.CH] = np.where(v < 0, -a2, a2)
    return out

def table_symbols(D, e):
    z = np.empty(D.size, np.uint16)
    for r in range(0, D.shape[0], dl.CH):
        v = D[r:r + dl.CH].astype(np.int64); q = v >> e[None, :]
        assert np.array_equal(q << e[None, :], v)
        zz = dl.zigzag(q).ravel(); assert int(zz.max()) < 65536; z[r * 768:r * 768 + zz.size] = zz
    return z

# the entry table's coder: combined_rc.c (a counting range coder, line as context, LIM = 10). Round one's develop_rc
# (chances nudged by 1/32, never settle) is kept as coder="rc1" for comparison: same symbols, ~3% more bytes.
import ctypes
_crc = ctypes.CDLL(HERE + "/combined_rc.dylib"); _crc.crc_encode.restype = ctypes.c_int64; _crc.crc_decode.restype = ctypes.c_int64
LIM = 10
def crc_encode(sym, ctx, nbits, lim=LIM):
    P_ = ctypes.c_void_p; cap = len(sym) * nbits // 8 + 65536; out = np.empty(cap, np.uint8)
    n = _crc.crc_encode(sym.ctypes.data_as(P_), ctx.ctypes.data_as(P_), ctypes.c_int64(len(sym)), ctypes.c_int(nbits),
                        ctypes.c_int64(768), ctypes.c_int(lim), out.ctypes.data_as(P_), ctypes.c_int64(cap))
    assert n <= cap; return out[:n].tobytes()
def crc_decode(b, ctx, nbits, lim=LIM):
    P_ = ctypes.c_void_p; buf = np.frombuffer(b, np.uint8).copy(); sym = np.empty(len(ctx), np.uint16)
    _crc.crc_decode(buf.ctypes.data_as(P_), ctypes.c_int64(len(buf)), ctx.ctypes.data_as(P_), ctypes.c_int64(len(ctx)),
                    ctypes.c_int(nbits), ctypes.c_int64(768), ctypes.c_int(lim), sym.ctypes.data_as(P_))
    return sym

def table_bytes(c, cap=12, half=None, coder="crc10"):
    """the entry table's stored numbers k = D >> e_i, zigzagged, range coded with the line as context, decoded and
    checked; plus the table header (c, cap, width, rows, LIM) and the lens (+ the 768 exposures if halftoned)."""
    key = f"{c}|{cap}" + ("" if half is None else f"|h{half}") + ("" if coder == "rc1" else "|" + coder)
    if key not in _tcache:
        D, _ = tables(c, cap, half); e = exposure(c, cap, half)
        z = table_symbols(D, e); w = dl.nb(int(z.max())); line = np.tile(np.arange(768, dtype=np.uint16), D.shape[0])
        if coder == "rc1": bq = dl.rc_encode(z, line, w, 768); zb = dl.rc_decode(bq, line, w, 768)
        else: bq = crc_encode(z, line, w); zb = crc_decode(bq, line, w)
        assert np.array_equal(zb, z); del z, zb, line
        _tcache[key] = len(bq) + 12 + LENS_BYTES + (0 if half is None else 768) + (0 if coder == "rc1" else 1)
        json.dump(_tcache, open(TCACHE, "w"))
    return _tcache[key]

# ------------------------------------------------------------------ the lossless rest
def _rc(a2):                                              # the photon coder on a 2-D whole-number array (|v| < 2^22)
    a2 = np.ascontiguousarray(a2, np.int32); assert int(np.abs(a2.astype(np.int64)).max()) < (1 << 22)
    b = encode(a2); assert np.array_equal(decode(b, *a2.shape), a2); return b

def rest_parts():
    """every other number the machine holds, coded exactly. Returns {part: bytes object} (each decodes back equal)."""
    M = _tb.M; out = {}
    # the plates' biases, one array per plate, in the stream's scale (whole numbers at 2^16)
    out["plate_biases"] = b"".join(struct.pack("<I", len(b)) + b for b in
                                   (_rc(M[n + ".b"].reshape(1, -1)) for n in NAMES))
    # the head's bias: one whole number per word; the head's power of two: one number for every word (checked)
    he = M["head.e"]; assert (he == he[0]).all()
    out["head_bias"] = _rc(M["head.b"].reshape(1, -1)) + struct.pack("<b", int(he[0]))
    # the position table: 1024 positions x 768 lines; each position written as its change from the one before
    P = M["wpe"].astype(np.int64); dP = P.copy(); dP[1:] = P[1:] - P[:-1]
    out["position_table"] = _rc(dP)
    # the arithmetic tables, each as its first value and its changes (they are smooth)
    def diffs(a): a = a.astype(np.int64); d = np.diff(a); return struct.pack("<qI", int(a[0]), len(a)) + _rc(d.reshape(1, -1))
    out["arith_tables"] = b"".join(diffs(M[k]) for k in ("phi", "arrival", "log2", "exp2")) + \
        struct.pack("<qq", int(M["log2_768"]), int(M["eps40"]))
    return out

_rest = None
def rest_bytes():
    global _rest
    if _rest is None:
        parts = rest_parts(); _rest = {k: len(v) for k, v in parts.items()}
    return _rest

# ------------------------------------------------------------------ judge and log
def log(rec):
    try: allr = json.load(open(OUTJ))
    except Exception: allr = []
    allr.append(rec); json.dump(allr, open(OUTJ, "w"), indent=1)
    print(json.dumps(rec), flush=True)

def which_missed():
    """which of the 30 texts lose their next word (same judge as words(): the machine as installed now)"""
    return [i for i, (ids, t) in enumerate(zip(_tb.inputs, _tb._TRUTH)) if _tb._run(ids, _tb._WTE0) != t]

def judge(version, cfg, c, cap=12, note="", predicted=None, extra=None, half=None):
    """the whole machine: developed plates (cfg) + developed tables (c) + everything else exact. Words / 30 and the
    total stored size (plates + tables + rest), each part real bytes."""
    t0 = time.time()
    pb = plates_size(cfg); tb_ = table_bytes(c, cap, half); rb = sum(rest_bytes().values())
    D, HT = tables(c, cap, half)
    install_plates(cfg)
    try:
        _tb._WTE0 = D; _tb.W["head"] = (HT, _HEAD[1], _HEAD[2])
        w = int(_tb.words()); miss = which_missed()
    finally:
        restore(); _tb._WTE0 = E0; _tb.W["head"] = _HEAD
    byk = {k: round(sum(plate_bytes(n, *cfg[n]) for n in NAMES if kind(n) == k) / 1e6, 3) for k in KINDS}
    rec = dict(version=version, note=note, predicted=predicted, c=c, cap=cap, half=half,
               A_by_plate={n: cfg[n][1] for n in NAMES} if len({cfg[n] for n in NAMES}) > 1 else cfg[NAMES[0]],
               plates_MB=round(pb / 1e6, 3), plates_by_kind_MB=byk, tables_MB=round(tb_ / 1e6, 3),
               rest_MB=round(rb / 1e6, 3), total_MB=round((pb + tb_ + rb) / 1e6, 3),
               plates_bits_per_weight=round(8 * pb / NVAL, 4), words=w, missed=miss, secs=round(time.time() - t0, 1))
    if extra: rec.update(extra)
    log(rec); return rec

tb = _tb; rc = _rc                                        # public names for the version scripts

# ------------------------------------------------------------------ the screen's phase (a control for the judge)
# Nearest rounding lays every developed value on the grid k x step. The same exposure on a grid shifted by a fraction
# phi of a step (one phi per neuron of a plate, one per line of the entry table, phi a whole number in [0, 2^16), from
# a fixed seed) is an equally fine development with the same grain power and an unrelated grain pattern -- the same
# film, another frame. Words over several phases show how much of a words count is the exposure and how much is the
# one pattern that happened to fall. phase seed 0 = no shift = the stored development. Not stored: a control only.
def _phi(n, seed): return np.random.default_rng(seed).integers(0, 1 << 16, size=n, dtype=np.int64)

def cut_phase(q, s, phi):
    q = q.astype(np.int64); s = np.broadcast_to(np.asarray(s, np.int64), q.shape); p = phi[None, :] << s
    k = ((q << 16) + p + (np.int64(1) << (15 + s))) >> (16 + s)
    w = ((k << (16 + s)) - p + (1 << 15)) >> 16
    return np.clip(w, -32768, 32767)

def tables_phase(c, cap, seed, half=None):
    e = exposure(c, cap, half); phi = _phi(768, 7000 + seed); p = (phi << e)[None, :]
    D = np.empty(E0.shape, np.int32); HT = np.empty((768, E0.shape[0]), np.float64)
    for r in range(0, E0.shape[0], dl.CH):
        v = E0[r:r + dl.CH].astype(np.int64)
        k = ((v << 16) + p + (np.int64(1) << (15 + e))[None, :]) >> (16 + e)[None, :]
        D[r:r + dl.CH] = ((k << (16 + e)[None, :]) - p + (1 << 15)) >> 16
        HT[:, r:r + dl.CH] = dl.print_top(D[r:r + dl.CH]).T
    return D, HT

def judge_phase(version, cfg, c, cap=12, seeds=(1, 2, 3, 4), note="", half=None):
    """words of the same exposures (plates cfg, tables c/cap) on shifted grids, one phase seed at a time"""
    out = []
    for sd in seeds:
        t0 = time.time(); _tab.clear()
        D, HT = tables_phase(c, cap, sd, half)
        for i, n in enumerate(NAMES):
            rule, A = cfg[n]; q = plate(n); t, off = terms(n, q, rule); s = steps(t, off, A)
            w = cut_phase(q, s, _phi(q.shape[1], 1000 * sd + i))
            _tb.W[n] = (w.astype(np.float64), _tb.W[n][1], _tb.W[n][2])
        try:
            _tb._WTE0 = D; _tb.W["head"] = (HT, _HEAD[1], _HEAD[2])
            w_ = int(_tb.words()); miss = which_missed()
        finally:
            restore(); _tb._WTE0 = E0; _tb.W["head"] = _HEAD
        del D, HT
        out.append(w_)
        log(dict(version=version + f" [phase seed {sd}]", note=note, c=c, cap=cap, phase_seed=sd,
                 A_by_plate={n: cfg[n][1] for n in NAMES} if len({cfg[n] for n in NAMES}) > 1 else cfg[NAMES[0]],
                 words=w_, missed=miss, control=True, secs=round(time.time() - t0, 1)))
    return out
