"""REFERENCE BEAM family, version 7: the reference as a NEGATIVE instead of a subtraction. For each line and each place,
the photon most words have there (lit or dark: the majority over the 50,257 words) is the line's reference photon;
every word stores its places XOR the reference (a place matching the reference goes dark; "every 1111 is 0000" where
the plate is white). The direction (sign) the same way: flipped against the line's majority direction.
Stored per line: one P-place mask + one direction bit (768 x (P+1) bits, counted). Each word's value is kept as one
whole number X = 2 (|v| XOR mask) + (direction XOR majority direction) -- always invertible (XOR twice is nothing).
Applied (a) to the table as stored, (b) to the version-1 residual. Coded with rc line+mant (odds per line).
Photons counted in X (the direction bit counts as a photon when it is lit). Also the floor of this idea: the fewest
photons any per-line, per-place negative can leave = sum over places of min(lit, dark) -- that is what (a) reaches.
    python3 refbeam_v7_negative.py
"""
import time, numpy as np
from refbeam_lib import *

def negative(V):
    a = np.abs(V); P = int(a.max()).bit_length(); N = V.shape[0]
    mask = np.zeros(V.shape[1], np.int64)
    for p in range(P):
        lit = ((a >> p) & 1).sum(0); mask |= (lit * 2 > N).astype(np.int64) << p
    sneg = ((V < 0).sum(0) * 2 > N).astype(np.int64)
    X = 2 * (a ^ mask[None, :]) + ((V < 0).astype(np.int64) ^ sneg[None, :])
    return X, mask, sneg, P

def undo(X, mask, sneg):
    a = (X >> 1) ^ mask[None, :]; s = (X & 1) ^ sneg[None, :]
    return np.where(s == 1, -a, a)

for name in ("E", "T"):
    A = load(name); n = A.size; t = time.time()
    ref = np.sort(A, axis=0)[A.shape[0] // 2]; sref = small(ref); R = A - ref[None, :]
    base_ph = float(ones(A).mean() + (A < 0).mean())
    print(f"   {name}: photons/value as stored, counting a hole's direction as one photon: {base_ph:.3f}")
    for label, V, extra in (("(a) negative of the table", A, 0), ("(b) negative of the v1 residual", R, sref)):
        X, mask, sneg, P = negative(V)
        ok_t = bool(np.array_equal(undo(X, mask, sneg), V))
        side = extra + (768 * (P + 1) + 7) // 8 + 8
        nb, ok = rc(X, "line", True)
        ph = float(ones(X).mean()); before = float(ones(V).mean() + (V < 0).mean())
        report("v7", name, f"{label}, rc line+mant", nb, side, n, ok and ok_t,
               {"photons_before": round(before, 4), "photons_after": round(ph, 4)})
        del X
    del R, A
    print(f"   ({time.time() - t:.0f} s)")
