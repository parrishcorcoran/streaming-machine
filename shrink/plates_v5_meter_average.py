"""PLATES v5 -- each neuron metered on its average, not its brightest (centre-weighted metering instead of exposing
for the highlights). The neuron's typical place tp_j = bit length of (sum of |weights|) minus bit length of the
line count plus 1 (counts and shifts only); keep a places below it: s_j = max(0, tp_j - a). A neuron whose brightest
weight is an outlier is no longer under-exposed. Stored: developed whole numbers, each neuron's power of two and
its s_j (1 byte each).
Prediction before running: the same words as v2 for ~0.3-0.5 bit less (the max/average ratio varies neuron to
neuron; v2 spends places on neurons with a modest brightest weight and starves those with an outlier)."""
from plates_lib import *
import sys
As = [int(x) for x in sys.argv[1:]] or [6, 5, 4]
def dev(n, q, e, a):
    R = q.shape[0]; ssum = np.abs(q.astype(np.int64)).sum(0)
    tp = bl(ssum) - bl(np.array(R)) + 1
    s = np.maximum(0, tp - a)
    return cut(q, s[None, :]), s[None, :]
for a in As:
    run_version(f"v5 average-metered neuron a={a}", lambda n, q, e, a=a: dev(n, q, e, a), side_per_col=2,
                note=f"each neuron keeps {a} places below its average's place, nearest")
