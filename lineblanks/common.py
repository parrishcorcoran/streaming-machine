"""Shared helpers for lineblanks/pairs.py and lineblanks/totals.py (way2.py keeps its own copies of the same code).

Every plate of ../foundation0/base2_pythia-<size>.npz is whole numbers q with one power of two e per neuron (.w the
finished, .s the blank, the same e for both). Sizes compared across neurons with different powers must be real sizes,
so `real` writes every value as a whole number on ONE scale 2^emin: q x 2^(e - emin), exact (e - emin <= 5 here, so
|value| < 2^21 and every sum below stays inside int64; asserted where used).
The fresh draw: each neuron's own spread in grains, the value written as fp16 (as the stored plates were) and put on
the neuron's power of two the way pythia_convert.py put every plate there.
"""
import numpy as np

def ones(X): return np.bitwise_count(np.abs(np.asarray(X, dtype=np.int64))).astype(np.int64)
def naf(X): a = np.abs(np.asarray(X, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)

def spread(w, axis):
    """each neuron's spread in grains (axis 1: neurons are columns; 0: rows), from whole-number sums"""
    s1 = np.zeros(w.shape[axis]); s2 = np.zeros(w.shape[axis]); n = w.shape[1 - axis]
    for i0 in range(0, w.shape[0], 256):
        b = np.asarray(w[i0:i0 + 256]).astype(np.int64)
        if axis == 1: s1 += b.sum(0); s2 += (b * b).sum(0)
        else: s1[i0:i0 + 256] = b.sum(1); s2[i0:i0 + 256] = (b * b).sum(1)
    return np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0))

def fresh(w, e, axis, rng):
    """a fresh bell-curve draw of w's per-neuron spreads, through fp16, onto the neuron's power of two (int16)"""
    sig = spread(w, axis); out = np.empty(w.shape, np.int16)
    def put(g, ee):
        sc = np.exp2(ee.astype(np.float64)); return np.clip(np.round((g * sc).astype(np.float16).astype(np.float64) / sc), -32767, 32767)
    if axis == 1:
        for i0 in range(0, w.shape[0], 64):
            n = min(64, w.shape[0] - i0); out[i0:i0 + n] = put(rng.normal(0.0, 1.0, (n, w.shape[1])) * sig[None, :], e[None, :])
    else:
        for i0 in range(0, w.shape[0], 4096):
            n = min(4096, w.shape[0] - i0); out[i0:i0 + n] = put(rng.normal(0.0, 1.0, (n, w.shape[1])) * sig[i0:i0 + n, None], e[i0:i0 + n, None])
    return out

def real(q, e, axis, emin):
    """q x 2^(e - emin) as int64 (axis 1: e per column; 0: e per row)"""
    sh = np.asarray(e, np.int64) - emin; assert sh.min() >= 0 and sh.max() <= 6
    q = np.asarray(q).astype(np.int64)
    return np.left_shift(q, sh[None, :] if axis == 1 else sh[:, None])

def versions(Z, name, axis, rng_seed):
    """the plate's versions as whole numbers on its own grid (int16): blank, finished, fresh_b, fresh_f"""
    w, s, e = np.asarray(Z[name + ".w"]), np.asarray(Z[name + ".s"]), np.asarray(Z[name + ".e"])
    yield "blank", s
    yield "finished", w
    yield "fresh_b", fresh(s, e, axis, np.random.default_rng(rng_seed + [1]))
    yield "fresh_f", fresh(w, e, axis, np.random.default_rng(rng_seed + [2]))
