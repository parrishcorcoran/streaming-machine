"""PLATES v17 -- the stream's bright lines set the gauge. v15b: exposing proj/mproj neurons by the next plate's line
(small line there -> coarser) cost 7 words. The reading: a small line in the next plate marks a line whose light is
large (the gain was made small to compensate); the gauge divides every line by the root of the position's total
light, which those few bright lines dominate -- so an error written onto a bright line moves every line, and those
neurons need FINER steps, not coarser. Here: v14, plus proj/mproj neuron d with the opposite sign of v15:
   term = mode(own) + (path_d - mode(path)) x (-1)  ->  smaller next line = finer step.
Prediction before running: if the gauge is why v15b lost, A=5 rises above v14's 26 (to 27-29) at about the same size."""
from plates_lib import *
import sys
src = open(os.path.join(HERE, "plates_v15_next_path_all.py")).read(); src = src[:src.index("for A in As:")]
g = {}; exec(src.replace("As = [int(x) for x in sys.argv[1:]] or [5, 4]", "As = []"), g)
PATH = g["PATH"]
extra = sum(plate(n).shape[0] for n in NAMES)
As = [int(x) for x in sys.argv[1:]] or [5, 6]
P = {}
for L in range(12):
    P[f"fc.{L}"] = PATH[f"fc.{L}"]
    P[f"qkv.{L}"] = [PATH[f"qkv.{L}"][2]]
    P[f"proj.{L}"] = [(c, 2 * mode(I) - I) for c, I in PATH[f"proj.{L}"]]      # reversed: small next line -> finer
    P[f"mproj.{L}"] = [(c, 2 * mode(I) - I) for c, I in PATH[f"mproj.{L}"]]
g["PATH"] = P
for A in As:
    def dev(n, q, e, A=A):
        s = g["places"](n, q, A); return cut(q, s), s
    run_version(f"v17 v14 + bright stream lines finer, A={A}", dev, side_per_col=2, extra_side=extra,
                note=f"v14; proj/mproj neurons finer where the next gauged plate's line is small; A={A}")
