"""PLATES v4 -- one exposure for the whole plate (the camera: one shutter, one gain for every pixel). In v2 each
neuron is exposed for its own brightest weight; here every neuron of a plate gets the same absolute step, the step of
the plate's most common neuron power of two e* at b places: s_j = max(0, e* + 15 - b - e_j). Bright neurons keep
more places, dim neurons fewer. Stored: developed whole numbers, each neuron's power of two (s_j follows from it and
one number per plate).
Why it could help: the light a plate adds to a neuron is a sum of incoming light x weight; rounding adds the same
scatter to every neuron whose step is the same -- for a plate writing straight onto the stream (proj, mproj) every
neuron's scatter counts the same, so one step everywhere puts the bits where the scatter is (equal-scatter rule).
Prediction before running: at the same stored size, as good as or better than v2 on proj/mproj, possibly worse on
qkv where a dim q neuron multiplies a bright k neuron."""
from plates_lib import *
import sys
bs = [int(x) for x in sys.argv[1:]] or [8, 7, 6]
def dev(n, q, e, b):
    vals, cnt = np.unique(e, return_counts=True); es = vals[np.argmax(cnt)]       # most common power of two (a count)
    s = np.maximum(0, es + 15 - b - e)
    return cut(q, s[None, :]), s[None, :]
for b in bs:
    run_version(f"v4 one exposure per plate b={b}", lambda n, q, e, b=b: dev(n, q, e, b), side_per_col=1,
                note=f"absolute step of the plate's most common neuron at {b} places, nearest")
