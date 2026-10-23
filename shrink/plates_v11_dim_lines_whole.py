"""PLATES v11 -- are the dim lines the brightest light? v7 keeps every weight to about the same relative precision,
including the lines whose weights are far below the plate's usual line (the gain folded in is small there because the
arriving light is large). If the arriving light on those lines is more than the folded gain makes up for, relative
precision is not enough there. Here: v7, and every line at least `dim` places below the most common line is kept
whole (not developed at all). Stored: as v7.
Prediction before running: if those lines carry the stream's few very bright lines, a=3 rises from 27 toward 29-30
at a cost of well under 1 MB; if not, no change."""
from plates_lib import *
import sys
extra = sum(plate(n).shape[0] for n in NAMES)
a = int(sys.argv[1]) if len(sys.argv) > 1 else 3
dims = [int(x) for x in sys.argv[2:]] or [2]
for dim in dims:
    cnt = [0]
    def dev(n, q, e, dim=dim):
        qa = np.abs(q.astype(np.int64)); C = q.shape[1]
        tr = bl(qa.sum(1)) - bl(np.array(C)) + 1; off = tr - mode(tr)
        s = v7_places(q, a); whole = off <= -dim; cnt[0] += int(whole.sum())
        s = np.where(whole[:, None], 0, s)
        return cut(q, s), s
    r = run_version(f"v11 v7 a={a}, lines >= {dim} places dim kept whole", dev, side_per_col=2, extra_side=extra,
                    note=f"v7 a={a}; lines whose average sits >= {dim} places under the plate's most common line: not developed")
    print("lines kept whole over all plates:", cnt[0])
