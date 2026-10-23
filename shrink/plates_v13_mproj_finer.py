"""PLATES v13 -- share the exposure by what each plate is. v10 showed the plate after the bend (mproj) loses about
twice as many words per MB saved as the others (0.85 words/MB vs 0.42-0.57). Equal return per byte: if the loss
grows with the scatter power (x4 per place) and mproj is ~2x as sensitive, it should sit half a place finer
(log2(2)/2 = 0.5 place = one half-place step). Rule fixed by kind, not by any number's look:
   mproj plates at A+1, everything else at A (v12 half-place meters).
Prediction before running: (A=5, mproj 6) ~42 MB holds 27-29 where plain A=5 fell to 22; (A=6, mproj 7) ~47 MB at 29-30."""
from plates_lib import *
import sys
extra = sum(plate(n).shape[0] for n in NAMES)
pairs = [tuple(int(y) for y in x.split(",")) for x in sys.argv[1:]] or [(5, 6), (6, 7), (4, 5)]
for A, Am in pairs:
    def dev(n, q, e, A=A, Am=Am):
        s = half_places(q, Am if n.startswith("mproj") else A); return cut(q, s), s
    run_version(f"v13 half-place A={A}, mproj A={Am}", dev, side_per_col=2, extra_side=extra,
                note=f"v12 meters; mproj at A={Am}, qkv/proj/fc at A={A}")
