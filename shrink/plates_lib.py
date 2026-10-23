"""PLATES family: shared pieces. The 48 plates (qkv/proj/fc/mproj x 12), int16 (lines in, neurons out), one power of
two per neuron (column).  Whole numbers only in every codec step; sizes are real bytes from plates_rc (an integer
range coder over the photons of each weight) plus every piece of side information.
"""
import sys, os, json, time, ctypes, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine")
from table_bits import *                                   # E, T, plate, ones, naf, entropy_bits, words, M
HERE = os.path.dirname(os.path.abspath(__file__))
OUTJ = os.path.join(HERE, "plates_results.json")
import table_bits as _tb
NAMES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
NVAL = sum(plate(n).size for n in NAMES)                   # 84,934,656
_lib = ctypes.CDLL(os.path.join(HERE, "plates_rc.dylib"))
_lib.plate_encode.restype = ctypes.c_int64
_lib.plate_encode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
_lib.plate_decode.restype = ctypes.c_int
_lib.plate_decode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int]

def colexp(name): return M[name + ".e"].astype(np.int64)  # each neuron's power of two

def encode(v, ctx=1):
    v = np.ascontiguousarray(v, dtype=np.int32); R, C = v.shape
    cap = v.size * 5 + 4096; out = np.zeros(cap, np.uint8)
    n = _lib.plate_encode(v.ctypes.data, R, C, out.ctypes.data, cap, ctx)
    assert n > 0, n
    return out[:n].tobytes()

def decode(buf, R, C, ctx=1):
    b = np.frombuffer(buf, np.uint8).copy(); v = np.zeros((R, C), np.int32)
    assert _lib.plate_decode(b.ctypes.data, len(buf), R, C, v.ctypes.data, ctx) == 0
    return v

def store(v, ctx=1, check=True):
    """real coded bytes of one plate's stored whole numbers; decodes back and checks"""
    buf = encode(v, ctx); ok = None
    if check: ok = bool(np.array_equal(decode(buf, *v.shape, ctx=ctx), v))
    return len(buf), ok

def bl(a):                                                 # bit length of each |a| (a count of lit-or-above places)
    a = np.abs(np.asarray(a, np.int64)); n = np.zeros(a.shape, np.int64)
    for b in range(40):
        m = (a >> b) > 0
        if not m.any(): break
        n += m
    return n

def shift_back(v, s):                                      # the developed weight on the plate: v x 2^s, saturated
    w = np.left_shift(v.astype(np.int64), np.asarray(s, np.int64))
    return np.clip(w, -32768, 32767).astype(np.int16)

def cut(q, s, chance=False, rng=None):
    """develop at place s (per column array or scalar): keep places >= s.  nearest: add half, shift.
    by chance: add a random whole number in [0, 2^s), shift -- each weight rounds up with chance = its remainder."""
    q = q.astype(np.int64); s = np.broadcast_to(np.asarray(s, np.int64), q.shape)
    if chance:
        u = rng.integers(0, 1 << 30, size=q.shape, dtype=np.int64) & ((np.int64(1) << s) - 1)
    else:
        u = np.where(s > 0, np.left_shift(np.int64(1), np.maximum(s - 1, 0)), 0)
    return np.right_shift(q + u, s).astype(np.int32)

def topk(q, k, chance=True, rng=None, floor=0):
    """develop to at most k photons per weight: each photon the signed power of two nearest the remainder; the last
    one chooses between the two powers around the remainder by chance (unbiased), or nearest. No photon below
    place `floor` (then the remainder below it is rounded the same way at that place)."""
    r = q.astype(np.int64); v = np.zeros_like(r)
    for step in range(k):
        a = np.abs(r); sg = np.sign(r)
        n = bl(a); lo = np.where(n > 0, np.left_shift(np.int64(1), np.maximum(n - 1, 0)), 0)
        f = np.int64(1) << floor
        small = lo < f                                     # below the floor: the two candidates are 0 and 2^floor
        base = np.where(small, 0, lo); span = np.where(small, f, lo)
        last = step == k - 1
        if chance and last:
            ru = rng.integers(0, 1 << 30, size=a.shape, dtype=np.int64) & np.maximum(span - 1, 0)
            up = ru < (a - base)
        else:
            up = 2 * (a - base) > span
        c = np.where(a > 0, np.where(up, base + span, base), 0) * sg
        v += c; r -= c
    return v.astype(np.int32)

def side_bytes(ncols_bytes_per_col=1, per_plate_bytes=4):
    cols = sum(plate(n).shape[1] for n in NAMES)
    return cols * ncols_bytes_per_col + len(NAMES) * per_plate_bytes

def log(rec):
    try: allr = json.load(open(OUTJ))
    except Exception: allr = []
    allr.append(rec); json.dump(allr, open(OUTJ, "w"), indent=1)

def run_version(name, develop, side_per_col=2, per_plate=4, do_words=True, check=True, note="", ctx=1, extra_side=0):
    """develop(name, q, e) -> (v stored whole numbers int32, s per column or scalar).  Codes all 48 plates, decodes
    and checks the stored numbers, rebuilds the plates (v << s) and asks for the words."""
    if do_words and _tb._TRUTH is None: assert words() == 30   # the machine's own answers, before any plate moves
    t0 = time.time(); tot = 0; allok = True; ph = 0; ph_naf = 0; by = {}; changed = 0
    try:
      for n in NAMES:
          q = plate(n); e = colexp(n)
          v, s = develop(n, q, e)
          nb, ok = store(v, ctx=ctx, check=check)
          tot += nb; allok &= bool(ok) if check else True
          k = n.split(".")[0]; by[k] = by.get(k, 0) + nb
          w = shift_back(v, s); changed += int((w != q).sum())
          _tb.W[n] = (w.astype(np.float64), _tb.W[n][1], _tb.W[n][2])     # the developed plate goes in place
          ph += int(ones(v).sum()); ph_naf += int(naf(v).sum())
    except BaseException:
        restore(); raise
    side = side_bytes(side_per_col, per_plate) + extra_side
    total = tot + side
    rec = {"version": name, "note": note, "coded_bytes": tot, "side_bytes": side, "total_MB": round(total / 1e6, 3),
           "bits_per_value": round(8 * total / NVAL, 4), "roundtrip_ok": allok if check else None,
           "photons_per_value": round(ph / NVAL, 3), "single_wave_photons_per_value": round(ph_naf / NVAL, 3),
           "by_kind_MB": {k: round(b / 1e6, 2) for k, b in by.items()}}
    rec["changed_share"] = round(changed / NVAL, 4)
    try:
        if do_words:
            t1 = time.time(); rec["words"] = int(words()); rec["secs_words"] = round(time.time() - t1, 1)
    finally:
        restore()
    rec["secs"] = round(time.time() - t0, 1)
    log(rec); print(json.dumps(rec), flush=True)
    return rec

def restore():                                             # the machine's own plates back in place
    for n in NAMES: _tb.W[n] = (plate(n).astype(np.float64), _tb.W[n][1], _tb.W[n][2])

def free_tables():                                         # the plates family never reads the word tables
    global E, T
    E = None; T = None; _tb.E = None; _tb.T = None

def set_lim(l): _lib.set_lim(int(l))                     # slowest the coder's chances move: 1/2^l per answer

free_tables()

def place_l2(name, cols=None):
    """each neuron's light for unit light on every line: the bit length of the root of its sum of squares, read in
    true scale (plus the neuron's power of two). Whole numbers: squares, sums, a bit length, a halving."""
    q = plate(name).astype(np.int64)
    if cols is not None: q = q[:, cols]
    ss = (q * q).sum(0); e = colexp(name) if cols is None else colexp(name)[cols]
    return (bl(ss) >> 1) + e

def mode(a):
    vals, cnt = np.unique(a, return_counts=True); return int(vals[np.argmax(cnt)])

def neuron_tp(q):                                          # the place of a neuron's average |weight| (counts, shifts)
    R = q.shape[0]; return bl(np.abs(q.astype(np.int64)).sum(0)) - bl(np.array(R)) + 1

def v7_places(q, a):
    """v7's floor place per weight: each neuron metered on its average |weight| (tp_j), each line offset by how far
    its own average sits from the most common line (tr_i): s_ij = max(0, tp_j - a + tr_i - mode tr)"""
    qa = np.abs(q.astype(np.int64)); R, C = q.shape
    tp = bl(qa.sum(0)) - bl(np.array(R)) + 1
    tr = bl(qa.sum(1)) - bl(np.array(C)) + 1; off = tr - mode(tr)
    return np.maximum(0, (tp - a)[None, :] + off[:, None])

def keep_sums(q, v, s):
    """after developing, give each neuron back its total (sum over lines, the light it passes for a uniform -- DC --
    light): flip the weights whose rounding was closest to a tie, one step each, while a flip still brings the
    neuron's total closer. Whole numbers: the total excess E_j, each weight's step D = 2^s and remainder r."""
    q = q.astype(np.int64); D = np.left_shift(np.int64(1), s.astype(np.int64)) * np.ones_like(q)
    w = np.left_shift(v.astype(np.int64), s.astype(np.int64)) * np.ones_like(q)
    d = w - q                                              # developed minus original, |d| <= D/2
    Ej = d.sum(0)                                          # each neuron's excess
    sgn = np.sign(Ej)[None, :]
    cand = (np.sign(d) == sgn) & (d != 0)                  # weights rounded the way of the excess
    cost = np.where(cand, D - 2 * np.abs(d), np.int64(1) << 40)   # extra squared error per unit moved: D - 2|d|
    order = np.argsort(cost, axis=0, kind="stable")
    Ds = np.take_along_axis(np.where(cand, D, 0), order, 0)
    cum_before = np.cumsum(Ds, 0) - Ds
    flip_sorted = (Ds > 0) & (2 * cum_before + Ds <= 2 * np.abs(Ej)[None, :])   # this flip still lowers |E|
    flip = np.zeros_like(cand); np.put_along_axis(flip, order, flip_sorted, 0)
    v2 = v.astype(np.int64) - np.where(flip, sgn, 0)       # one step back toward the original, in stored units
    return v2.astype(np.int32)

def halfplace(x):
    """place of a whole number in half places: 2 x (top place) + the next place down (lit = upper half of the octave)"""
    x = np.asarray(x, np.int64); n = bl(x)
    nxt = np.where(n >= 2, (x >> np.maximum(n - 2, 0)) & 1, 0)
    return np.where(x > 0, 2 * (n - 1) + nxt, -1000)

def half_places(q, A):
    """v7's metering read to half a place: H_ij = neuron's half place + line's half-place offset - A; the step is the
    whole place under it, s_ij = max(0, H_ij >> 1). A = 2a matches v7 a; odd A sits between."""
    qa = np.abs(q.astype(np.int64)); R, C = q.shape
    tph = halfplace(qa.sum(0)) + 2 * (2 - bl(np.array(R)))
    trh = halfplace(qa.sum(1)); off = trh - mode(trh)
    H = tph[None, :] + off[:, None] - A
    return np.maximum(0, H >> 1)
