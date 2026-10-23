"""DEVELOP family, version 4: the word-wide place cut of version 3, rounded BY CHANCE (exact on average). This is where
dithering should pay: under a hard cut every value smaller than half a step goes dark together -- a dead zone, a loss
that does not average out over the 768 lines. By chance, a value of a quarter step is lit a quarter of the time, so the
pile over 768 lines keeps it on average (halftone printing: grey survives a black/white press).
Same ladders as version 3, seeds fixed (E: 300 + b, T: 400 + b).
Predicted before running: at the fine cuts (E b=10/8, T b=12/10) equal to or a little worse than version 3; at the
coarse cuts (E b=6/4, T b=8/6) BETTER than version 3 by several words. Stored a little larger than version 3 at the
same b (chance spreads values over more steps).
    python3 develop_v4_cut_dither.py words|store
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from develop_lib import run
SPECS = [dict(name=f"v4 cut b={b} dither E", E=dict(how="cut", b=b, seed=300 + b)) for b in (10, 8, 6, 4)]
SPECS += [dict(name=f"v4 cut b={b} dither T", T=dict(how="cut", b=b, seed=400 + b)) for b in (12, 10, 8, 6)]
if __name__ == "__main__": run(SPECS, sys.argv[1])
