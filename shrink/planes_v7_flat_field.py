"""PLANES family, version 7: flat field first (a camera's dark-frame correction). Each line's dark level -- the mean
of its first 32,768 words, a sum and a shift by 15 -- is stored as side information (768 whole numbers) and taken off
every value of that line before the places are made; the reader adds it back. Then the places are coded as
  a) version 3a contexts (higher places, line, sign context line x top place)
  b) version 6b contexts (+ word exposure meter + line-aware places below the top)
Prediction: a shift of a whole line does not change how much that line holds (H(v - d | line) = H(v | line)), so an
ideal per-line coder gains nothing; ours gains only what the fold v -> |v| cost it on lines whose light sits off zero
(the sign context already took most of that: sign 1.000 -> 0.859). A few hundredths at most.
Also prints the per-line floor: each line's own histogram of v >> 7 (the 7 lowest places are coin flips: coded at
1.000 bit each in every version) + 7 bits -- what any coder that knows only the line can reach.
    python3 planes_v7_flat_field.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from planes_lib import *

def H(c):
    c = c[c > 0].astype(np.float64); p = c / c.sum(); return float(-(c * np.log2(p)).sum())

for tb in ["E", "T"]:
    A = load(tb); R, C = A.shape
    # floor references (analysis only, floats allowed here)
    q = (A >> 7).astype(np.int64); q -= q.min()
    f_line = sum(H(np.bincount(q[:, c])) for c in range(C)) / A.size + 7
    _, cnt = np.unique(A, return_counts=True); f0 = H(cnt) / A.size
    print(f"   {tb}: order-0 floor {f0:.4f} bits;  per-line floor (each line its own histogram of v>>7, + 7) {f_line:.4f} bits", flush=True)
    off = ((A[:32768].astype(np.int64).sum(0) + (1 << 14)) >> 15).astype(np.int32)
    side = C * (int(np.abs(off).max()).bit_length() + 1)
    F = A - off[None, :]
    for label, mode in [("v7a flat field + v3a contexts", M_HIGH | M_LINE | M_SIGNC),
                        ("v7b flat field + v6b contexts", M_HIGH | M_LINE | M_SIGNC | M_WEXP | M_REFLINE)]:
        res, data = run(label, tb, F, mode, extra_side=side, note=f"per-line floor {f_line:.4f}, order-0 floor {f0:.4f}")
        # check the full path from bytes: decode, add the dark level back, compare with the original table
        nb = nplanes(F); B = np.zeros_like(F); m2 = np.zeros(nb + 1, np.int64)
        lib().planes(1, B.ctypes.data, R, C, nb, mode, None, data.ctypes.data, len(data), m2.ctypes.data)
        print(f"     from bytes + dark levels back: equal to the original table: {bool(np.array_equal(B + off[None, :], A))}", flush=True)
        del B
    del A, F, q
