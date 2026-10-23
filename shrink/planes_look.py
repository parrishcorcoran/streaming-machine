"""PLANES family, look first (no coding): what each place (bit plane) of the entry table E and the top table T holds.
Reads the same arrays table_bits.py serves, straight from the npz (E = M["wte"], T = M["head.w"].T), so the 1.5 GB
machine is not loaded. Floats appear only in the printed reference numbers (Shannon floors), never in a codec.
    python3 planes_look.py
"""
import numpy as np
NPZ = "/Users/abundancemachine/streaming-machine/base2_gpt2.npz"
M = np.load(NPZ)

def H(counts):
    c = counts[counts > 0].astype(np.float64); p = c / c.sum(); return float(-(c * np.log2(p)).sum())

def look(name, A):
    R, C = A.shape; a = np.abs(A).astype(np.int64); N = A.size
    nb = int(a.max()).bit_length()
    print(f"\n=== {name}: {R} x {C}, max |v| = {a.max()}, {nb} magnitude places + sign; zeros {np.mean(a == 0):.4%}")
    print(" place : share lit (1)  | share lit among values whose top lit place is above this one")
    top = np.zeros(A.shape, np.int8)
    for k in range(nb - 1, -1, -1):
        b = (a >> k) & 1
        above = top > 0
        print(f"   {k:2d}  : {b.mean():.4f}        | {b[above].mean() if above.any() else float('nan'):.4f}   (values lit above: {above.mean():.3f})")
        top[(b == 1) & (top == 0)] = k + 1
    tp = top.astype(np.int64)                      # top lit place + 1 (0 = value is zero)
    print(" top lit place histogram:", np.bincount(tp.ravel(), minlength=nb + 1).tolist())
    s = A < 0
    print(f" sign: share negative {s[a > 0].mean():.4f}")
    # order-0 floor of the whole value, and its chain-rule split: top place + the places below it (all with full prefix)
    _, cv = np.unique(A, return_counts=True); h0 = H(cv) / N
    htop = H(np.bincount(tp.ravel())) / N
    print(f" order-0 floor of the value: {h0:.3f} bits;  of its top place alone: {htop:.3f} bits")
    # top place given line (each line its own histogram): 768 histograms of ~20 bins over 50k values -- small bias
    ht_line = sum(H(np.bincount(tp[:, c], minlength=nb + 1)) for c in range(C)) / N
    ht_word = sum(H(np.bincount(tp[r], minlength=nb + 1)) for r in range(0, R)) / N
    print(f" top place given its line: {ht_line:.3f} bits;  given its word (768 values a word): {ht_word:.3f} bits")
    # mantissa places below the top: share lit at depth 1, 2, 3, 4 below the top lit place
    for d in range(1, 6):
        m = tp - 1 - d >= 0
        bits = (a[m] >> (tp[m] - 1 - d)) & 1
        print(f"   depth {d} below the top: share lit {bits.mean():.4f}  ({m.mean():.3f} of values have this place)")
    # how many places below the top: the "mantissa" bits that are coin flips
    below = np.maximum(tp - 1, 0)
    print(f" places strictly below the top lit place, per value: {below.mean():.3f}  (+ top place {htop:.3f} + sign 1 = {below.mean() + htop + 1:.3f})")
    # per-word and per-line scale spread: top place of the mean |v| (whole numbers: sum then bit length)
    wl = np.array([int(x).bit_length() for x in a.sum(1)]); ll = np.array([int(x).bit_length() for x in a.sum(0)])
    print(f" per-word top place of sum|v|: min {wl.min()} max {wl.max()}, histogram {np.bincount(wl)[wl.min():].tolist()}")
    print(f" per-line top place of sum|v|: min {ll.min()} max {ll.max()}, histogram {np.bincount(ll)[ll.min():].tolist()}")
    # line DC: mean per line
    mean = A.astype(np.int64).sum(0) // R
    print(f" per-line mean (DC) |mean| largest 5: {np.sort(np.abs(mean))[-5:].tolist()}, typical |v| {int(np.median(a))}")

look("E (entry table, 2^16)", M["wte"].astype(np.int32))
look("T (top table)", np.ascontiguousarray(M["head.w"].T).astype(np.int32))
