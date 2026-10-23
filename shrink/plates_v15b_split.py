"""PLATES v15b -- which half of v15 hurt? v14 (fc and v neurons by their next path) plus ONE of:
   (qk)     q and k neurons exposed by their partner's size
   (stream) proj and mproj neurons exposed by the next gauged plate's line d
Prediction before running: the stream half hurt -- the stream's brightest lines set the gauge (the root of the total
light of the position divides every line), so an error on a bright line moves every line; the next plate's small
line there (the gain made small to compensate) says the opposite."""
from plates_lib import *
import sys, importlib.util
spec = importlib.util.spec_from_file_location("v15", os.path.join(HERE, "plates_v15_next_path_all.py"))
src = open(os.path.join(HERE, "plates_v15_next_path_all.py")).read(); src = src[:src.index("for A in As:")]
g = {"__name__": "v15lib"}; exec(src.replace("As = [int(x) for x in sys.argv[1:]] or [5, 4]", "As = []"), g)
PATH = g["PATH"]; places_full = g["places"]
extra = sum(plate(n).shape[0] for n in NAMES)
A = int(sys.argv[1]) if len(sys.argv) > 1 else 5
for part in (sys.argv[2:] or ["qk", "stream"]):
    P = {}
    for L in range(12):
        P[f"fc.{L}"] = PATH[f"fc.{L}"]
        P[f"qkv.{L}"] = PATH[f"qkv.{L}"] if part == "qk" else [PATH[f"qkv.{L}"][2]]
        P[f"proj.{L}"] = PATH[f"proj.{L}"] if part == "stream" else []
        P[f"mproj.{L}"] = PATH[f"mproj.{L}"] if part == "stream" else []
    def dev(n, q, e, P=P):
        g["PATH"] = P; s = g["places"](n, q, A); return cut(q, s), s
    run_version(f"v15b v14 + {part} next path, A={A}", dev, side_per_col=2, extra_side=extra,
                note=f"v14 plus the {part} part of v15's next-path rule; A={A}")
