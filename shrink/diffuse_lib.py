"""DIFFUSE family -- error diffusion on the plates (halftoning). A printer prints a grey photo with coarse dots by
carrying each dot's rounding error to the next dot, so the errors cancel in any average over neighbours. A neuron's
pile is a sum over its lines, so here each weight's rounding error is carried to the NEXT LINE of the same neuron
(or, as a control, to the next neuron on the same line) while developing.

Exposure is round one's (plates family): per neuron a step term t_j, per line an offset off_i (half places), the step
of weight (i, j) is s_ij = max(0, (t_j + off_i - A) >> 1) -- a shift. Developing at s keeps the places >= s.
  plain (control):  v = (q + half) >> s
  carry:            u = q + c,  v = (u + half) >> s,  e = (v << s) - u  (this weight's rounding error),
                    the next weight in the chain gets c = -e (first order) or -(2e - e_prev) (second order).
  carry in step units: the error is carried as a fraction of the step it was made with (e x 2^(16 - s), a shift)
                    and given to the next weight in ITS step (x 2^(s' - 16), a shift, rounded to nearest): the pile
                    sees the same fraction of a step on every line, whatever each line's exposure.
Everything in developing, coding and decoding is whole numbers: adds, subtracts, shifts, compares, clips.

The file format and the decoder are round one's (plates_best_store.py): per plate a header, each neuron's power of
two, each neuron's step term, each line's offset, and the photon-coded stored numbers. The decoder never needs to know
that the encoder carried errors: it rebuilds s_ij and w = v << s_ij exactly as before. Every version here is written
to a real file, decoded back from that file alone, checked equal to what the encoder developed, and judged ONLY by
words(plates=...) with all 48 plates in place at once. Results go to diffuse_results.json (this family's own).
"""
import sys, os, json, time, struct, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/abundancemachine/streaming-machine"); sys.path.insert(0, HERE)
from plates_lib import *                                    # plate, colexp, encode, decode, shift_back, cut, NAMES, NVAL
import table_bits as _tb
_src = open(os.path.join(HERE, "plates_best_store.py")).read()
exec(_src[_src.index("OUTDIR ="):_src.index("assert words() == 30")])   # NP, terms(n, q, rule), steps(t, off, A)
SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad"
OUTJ_D = os.path.join(HERE, "diffuse_results.json")
P = 16                                                      # fraction places of a carry measured in steps
HDR = "<BHHbhhI"

def half(s):
    s = np.asarray(s, np.int64); return np.where(s > 0, np.left_shift(np.int64(1), np.maximum(s - 1, 0)), 0)

def rnd_shift(x, k):
    """x x 2^-k rounded to nearest (k >= 0), or x x 2^|k| (k < 0): add half, shift"""
    x = np.asarray(x, np.int64); k = np.asarray(k, np.int64)
    up = np.left_shift(x, np.maximum(-k, 0)); kp = np.maximum(k, 0)
    return np.right_shift(up + np.where(kp > 0, np.left_shift(np.int64(1), np.maximum(kp - 1, 0)), 0), kp)

def develop_carry(q, s, axis=0, order=None, kind=1, unit=False, leak=0, reset=None):
    """stored whole numbers v (int32) of a plate developed at steps s with the rounding error carried along a chain.
    axis 0: the chain runs down each neuron's column, line after line (in `order`); axis 1: along each line, neuron
    after neuron. kind 1: carry -e; kind 2: carry -(2e - e_prev). unit: carry in fractions of a step (see top).
    leak k > 0: carry only e - (e >> k) (a shift; a sigma-delta with a leak). reset: boolean per chain position,
    True = the chain starts over there (nothing carried in)."""
    Q = np.asarray(q, np.int64); S = np.broadcast_to(np.asarray(s, np.int64), Q.shape)
    if axis == 1: Q = Q.T; S = S.T
    n, m = Q.shape
    order = np.arange(n) if order is None else np.asarray(order)
    V = np.zeros((n, m), np.int64)
    e1 = np.zeros(m, np.int64); e2 = np.zeros(m, np.int64)    # last two errors (original units, or steps x 2^P)
    for k, i in enumerate(order):
        si = S[i]
        if reset is not None and reset[k]: e1[:] = 0; e2[:] = 0
        f = e1 if kind == 1 else 2 * e1 - e2
        if leak: f = f - (f >> leak)
        c = -f if not unit else -rnd_shift(f, P - si)
        u = Q[i] + c
        v = np.right_shift(u + half(si), si)
        w = np.clip(np.left_shift(v, si), -32768, 32767)
        v = np.right_shift(w, si)                            # what the decoder will rebuild (w is a multiple of 2^s)
        w = np.left_shift(v, si)
        w = np.clip(w, -32768, 32767)
        e = w - u
        if unit: e = np.left_shift(e, P - si)
        e2 = e1; e1 = e
        V[i] = v
    if axis == 1: V = V.T
    return np.ascontiguousarray(V).astype(np.int32)

def develop_plain(q, s):
    return cut(q, s)

# ---------------------------------------------------------------- the file: round one's format, written and read back
def write_file(path, rule, A, develop):
    """develop(n, q, s) -> v. Writes the file; returns (size, {name: developed int16 plate as the encoder sees it},
    [photons, single-wave photons] of the stored numbers)"""
    W16 = {}; ph = [0, 0]
    with open(path, "wb") as f:
        for idx, n in enumerate(NAMES):
            q = plate(n); R, C = q.shape; e = colexp(n)
            t, off = terms(n, q, rule); s = steps(t, off, A)
            v = develop(n, q, s)
            buf = encode(v)
            tb = int(t.min()); ob = int(off.min())
            assert (t - tb).max() < 256 and (off - ob).max() < 256 and e.min() > -128
            f.write(struct.pack(HDR, idx, R, C, A, tb, ob, len(buf)))
            f.write(e.astype(np.int8).tobytes()); f.write((t - tb).astype(np.uint8).tobytes())
            f.write((off - ob).astype(np.uint8).tobytes()); f.write(buf)
            W16[n] = shift_back(v, s); ph[0] += int(ones(v).sum()); ph[1] += int(naf(v).sum())
    return os.path.getsize(path), W16, ph

def read_file(path):
    """the decoder: sees only the file. Returns {name: int16 plate}"""
    data = open(path, "rb").read(); p = 0; hs = struct.calcsize(HDR); out = {}
    for _ in NAMES:
        idx, R, C, A, tb, ob, nb = struct.unpack_from(HDR, data, p); p += hs
        e = np.frombuffer(data, np.int8, C, p).astype(np.int64); p += C
        t = np.frombuffer(data, np.uint8, C, p).astype(np.int64) + tb; p += C
        off = np.frombuffer(data, np.uint8, R, p).astype(np.int64) + ob; p += R
        v = decode(data[p:p + nb], R, C); p += nb
        n = NAMES[idx]; assert np.array_equal(e, colexp(n))
        out[n] = shift_back(v, steps(t, off, A))
    assert p == len(data)
    return out

def logd(rec):
    try: allr = json.load(open(OUTJ_D))
    except Exception: allr = []
    allr.append(rec); json.dump(allr, open(OUTJ_D, "w"), indent=1)

def run(label, rule, A, develop, path=None, note="", keep=False):
    """develop all 48 plates, write the file, read it back with the decoder alone, check, judge by words."""
    if _tb._TRUTH is None: assert words() == 30
    path = path or os.path.join(SP, "diffuse_current.bin")
    t0 = time.time()
    size, Wenc, ph = write_file(path, rule, A, develop)
    Wdec = read_file(path)
    ok = all(np.array_equal(Wenc[n], Wdec[n]) for n in NAMES)
    chg = sum(int((Wdec[n] != plate(n)).sum()) for n in NAMES)
    t1 = time.time(); wd = int(words(plates=Wdec)); restore()
    rec = {"version": label, "rule": rule, "A": A, "note": note, "file": path if keep else None, "file_bytes": size,
           "total_MB": round(size / 1e6, 3), "bits_per_value": round(8 * size / NVAL, 4), "roundtrip_ok": ok,
           "stored_photons_per_weight": round(ph[0] / NVAL, 3),
           "stored_single_wave_photons_per_weight": round(ph[1] / NVAL, 3), "changed_share": round(chg / NVAL, 4),
           "words": wd, "secs": round(time.time() - t0, 1)}
    logd(rec); print(json.dumps(rec), flush=True)
    return rec
