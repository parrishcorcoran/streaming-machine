"""PLATES v6 -- expose each line for the light that arrives on it. The light a plate scatters into a neuron is
sum over lines (incoming light x weight); rounding a weight by d adds (incoming light x d) -- so a weight on a bright
line must be developed finer than one on a dim line (scatter adds in power, line by line).  The light arriving on a
line is the light the plate before it sent: the relation, read from the plates themselves, nothing run:
   proj.L line i   <- neuron i of the v part of qkv.L    (the light a plate sends for unit light in = root sum of squares)
   mproj.L line i  <- neuron i of fc.L
   qkv.L, fc.L     <- the stream after the gauge: every line the same
Line offset o_i = (that neuron's place) - (the most common such place); s_ij = max(0, s_j - o_i) with s_j from v5
(each neuron metered on its average). Stored: developed whole numbers, per neuron its power of two and s_j, per
line of proj/mproj its offset (1 byte).
Prediction before running: proj/mproj lines carry very uneven light (fc neurons' powers of two span ~32x), so at the
same words the plates shrink by ~0.3-0.5 bit/weight, or at the same size v5's 27-28/30 rises."""
from plates_lib import *
import sys
As = [int(x) for x in sys.argv[1:]] or [5, 4, 3]
OFF = {}
for L in range(12):
    OFF[f"proj.{L}"] = place_l2(f"qkv.{L}", np.arange(1536, 2304))
    OFF[f"mproj.{L}"] = place_l2(f"fc.{L}")
for k in OFF: OFF[k] = OFF[k] - mode(OFF[k])
print("line offsets proj.0:", np.unique(OFF["proj.0"], return_counts=True), " mproj.0:", np.unique(OFF["mproj.0"], return_counts=True))
extra = sum(len(v) for v in OFF.values())
def dev(n, q, e, a):
    s = np.maximum(0, neuron_tp(q) - a)[None, :]
    if n in OFF: s = s - OFF[n][:, None]
    s = np.maximum(0, s)
    return cut(q, s), s
for a in As:
    run_version(f"v6 line light + average neuron a={a}", lambda n, q, e, a=a: dev(n, q, e, a), side_per_col=2, extra_side=extra,
                note=f"v5 neuron metering a={a}; proj/mproj lines exposed for the light the plate before sends them")
