"""PLATES v16 -- v14's exposure (fc and v neurons by their next path, everything else own-size, line offsets) with
mproj half a place finer (v13's sharing by kind). Pairs "A,Am": qkv/proj/fc at A, mproj at Am.
Prediction before running: (5,6) ~42 MB at 27-28 (v14 A=5: 26); (7,7) ~51 MB at 29-30; (8,8) ~56 MB at 30."""
from plates_lib import *
import sys
src = open(os.path.join(HERE, "plates_v14_next_path.py")).read(); src = src[src.index("def next_path"):src.index("for A in As:")]
g = {}; exec("from plates_lib import *\n" + src, g)
extra = sum(plate(n).shape[0] for n in NAMES)
pairs = [tuple(int(y) for y in x.split(",")) for x in sys.argv[1:]] or [(7, 7), (8, 8), (5, 6), (6, 7)]
for A, Am in pairs:
    def dev(n, q, e, A=A, Am=Am):
        s = g["places"](n, q, Am if n.startswith("mproj") else A); return cut(q, s), s
    run_version(f"v16 v14 A={A}, mproj A={Am}", dev, side_per_col=2, extra_side=extra,
                note=f"v14 exposure; qkv/proj/fc at A={A}, mproj at A={Am}")
