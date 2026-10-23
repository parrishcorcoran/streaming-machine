"""PLATES v7 -- meter every weight on both its line and its neuron (an exposure per line AND per neuron).
What v6 missed: the light arriving on a line of qkv/fc is the stream itself, and the plate already tells how bright
each line is -- the gain folded into the plate is small on a line exactly where the arriving light is strong (what
passes through each line is about the same light). A uniform cut zeroes such a line's weights while its light is
the brightest. Rule, from each plate's own lines and neurons (sums, bit lengths -- no ordering, no grouping):
   tp_j = place of neuron j's average |weight|, tr_i = place of line i's average |weight|
   s_ij = max(0, tp_j - a + (tr_i - most common tr))   -- every weight kept to about the same relative precision.
Stored: developed whole numbers, per neuron its power of two and tp_j, per line its offset (1 byte each).
Prediction before running: if the dim lines are the bright-light lines, words at a=4..3 rise clearly above v5/v6
(27 and 24) for about the same size (dim lines are few: ~1-10% of lines)."""
from plates_lib import *
import sys
As = [int(x) for x in sys.argv[1:]] or [5, 4, 3]
extra = sum(plate(n).shape[0] for n in NAMES)
def dev(n, q, e, a):
    qa = np.abs(q.astype(np.int64)); R, C = q.shape
    tp = bl(qa.sum(0)) - bl(np.array(R)) + 1
    tr = bl(qa.sum(1)) - bl(np.array(C)) + 1; off = tr - mode(tr)
    s = np.maximum(0, (tp - a)[None, :] + off[:, None])
    return cut(q, s), s
for a in As:
    run_version(f"v7 line+neuron metering a={a}", lambda n, q, e, a=a: dev(n, q, e, a), side_per_col=2, extra_side=extra,
                note=f"s_ij = tp_j - {a} + (tr_i - mode tr), nearest")
