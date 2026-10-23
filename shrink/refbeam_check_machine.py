"""REFERENCE BEAM family: the full loop for the best version (version 1, the line's most common value as the
reference, range coder with odds per line). Encode E and T, decode them from the coded bytes alone (plus the 768
reference values), add the reference back, and run the machine on the DECODED tables: right next word out of 30.
Lossless, so it must be 30 -- this checks the loop end to end.
    python3 refbeam_check_machine.py
"""
import sys, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine"); sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from refbeam_lib import *

def common(A):
    out = np.zeros(A.shape[1], np.int64)
    for j in range(A.shape[1]):
        u, c = np.unique(A[:, j], return_counts=True); out[j] = u[np.flatnonzero(c == c.max())[0]]
    return out

dec = {}; sizes = {}
for name, mant in (("E", False), ("T", True)):
    A = load(name); ref = common(A)
    nb, ok, back = rc(A - ref[None, :], "line", mant, give_back=True)
    dec[name] = back + ref[None, :]; sizes[name] = nb + small(ref)
    print(f"{name}: {sizes[name]:,} bytes ({8 * sizes[name] / A.size:.4f} b/v), decoded equal: {np.array_equal(dec[name], A)}", flush=True)
    del A, back
import table_bits as tb
w = tb.words(entry=dec["E"], top=dec["T"].astype(np.int16))
print(f"machine on the decoded tables: {w}/30")
