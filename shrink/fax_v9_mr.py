"""FAX family, version 9: the fax 2-D (MR) idea taken literally -- each fax line coded as what CHANGED from the line
before it (adding: the reader keeps a running sum). Two ways:
  words : each word's 768 values minus the word before's (the word above on the page)
  lines : each value minus the one on the line to its left (in the same word)
The differences are then coded exactly like version 6 (JBIG contexts). Prediction before running: if neighbouring
words (lines) share nothing, a difference of two independent values has twice the spread, so it costs half a bit more
per value (log2 of sqrt 2); only lines with a big shared dark level gain. Decoded (running sums) and checked equal.
    python3 fax_v9_mr.py E|T|fc.0
"""
import sys, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from fax_common import *
import fax_v6_jbig as v6

name = sys.argv[1] if len(sys.argv) > 1 else "E"
A = load(name); res = []
for how in ("words", "lines"):
    t = time.time()
    D = np.diff(A, axis=0 if how == "words" else 1, prepend=0).astype(np.int32)   # first word/line kept as it is
    bufs, nb = v6.code(D); nbytes = sum(len(b) for b in bufs); side = 2 * 32 + 5 + 64 * len(bufs)
    B = np.cumsum(v6.decode(bufs, nb, D.shape), axis=0 if how == "words" else 1, dtype=np.int64); del D
    ok = bool(np.array_equal(B, A)); del B
    res.append(report(f"v9 MR: difference from the {how[:-1]} before, JBIG", name, nbytes, side, A.size, ok))
    print(f"     ({time.time() - t:.0f} s)", flush=True)
json.dump(res, open(f"/Users/abundancemachine/streaming-machine/shrink/fax_out_v9_{name}.json", "w"))
