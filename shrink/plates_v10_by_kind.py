"""PLATES v10 -- which plates can take a coarser development? Everything at v7 a=4 (30/30), and one group of plates
at a coarser a; the words say whether that group's scatter shows. Groups are fixed by what a plate is (kind) or
where it sits (depth) -- never by what its numbers look like.
Prediction before running: the plates writing straight onto the stream after the bend (mproj, 3072 lines, the
largest) and the early layers are the most sensitive; proj the least."""
from plates_lib import *
import sys
extra = sum(plate(n).shape[0] for n in NAMES)
groups = {"qkv": lambda n: n.startswith("qkv"), "proj": lambda n: n.startswith("proj"),
          "fc": lambda n: n.startswith("fc"), "mproj": lambda n: n.startswith("mproj"),
          "layers0-3": lambda n: int(n.split(".")[1]) < 4, "layers4-7": lambda n: 4 <= int(n.split(".")[1]) < 8,
          "layers8-11": lambda n: int(n.split(".")[1]) >= 8}
coarse = int(sys.argv[1]) if len(sys.argv) > 1 else 2
which = sys.argv[2:] or list(groups)
for g in which:
    def dev(n, q, e, g=g):
        s = v7_places(q, coarse if groups[g](n) else 4); return cut(q, s), s
    run_version(f"v10 {g} at a={coarse}, rest a=4", dev, side_per_col=2, extra_side=extra,
                note=f"v7 metering; group {g} at a={coarse}, all other plates at a=4")
