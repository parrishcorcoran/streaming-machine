"""PLATES v19 -- fewer photons per weight on top of the metered exposure. v14's development (A), then no weight may
carry more than k single-wave photons: each photon the signed power of two nearest what is left (nearest -- v3/v8/v18
showed rounding by chance only adds scatter). Also the limit of 'only the 1s': ONE photon per weight (every weight
0 or +-2^p) on a fine floor.
Prediction before running: k=2 on A=6/A=7 saves ~0.15-0.25 bit/weight for ~1 word; one photon per weight on the
A=8 floor stores ~3 bits/weight but its scatter (a lone power of two misses a weight by up to a third of it) is
worse than A=4's (23 words) -> about 10-18 words."""
from plates_lib import *
import sys
src = open(os.path.join(HERE, "plates_v14_next_path.py")).read(); src = src[src.index("def next_path"):src.index("for A in As:")]
g = {}; exec("from plates_lib import *\n" + src, g)
extra = sum(plate(n).shape[0] for n in NAMES)
runs = [tuple(int(y) for y in x.split(",")) for x in sys.argv[1:]] or [(6, 2), (7, 2), (8, 1)]
for A, k in runs:
    def dev(n, q, e, A=A, k=k):
        s = g["places"](n, q, A); v = cut(q, s); return topk(v, k, chance=False), s
    run_version(f"v19 v14 A={A} + at most {k} photon(s) per weight", dev, side_per_col=2, extra_side=extra,
                note=f"v14 exposure A={A}, then cap at {k} single-wave photons per weight, nearest")
