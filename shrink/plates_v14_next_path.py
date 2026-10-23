"""PLATES v14 -- expose a neuron for how strongly the next plate reads it (the path the light takes next), not for
its own size. A rounding error on neuron j of fc reaches the stream only through line j of mproj; on neuron d of the
v part of qkv only through line d of proj. Equal scatter on the stream (the scatter of all neurons adds in power)
asks for steps inversely to that next path: step_j ~ 1 / (size of the next plate's line j). The bend is close to
straight-through for large light, so a neuron's own size and its next path trade against each other (scale one up,
the other down, same machine) -- only their product is real, and v12 exposed for the wrong half of it.
Rule (whole numbers: |weight| x 2^(neuron power - smallest power), sums, half places):
   fc.L neuron j      : I_j = half place of sum_k |mproj.L[j, k]| in true scale
   qkv.L v-neuron d   : I_d = half place of sum_k |proj.L[d, k]| in true scale
   neuron term = (most common own half place) - (I_j - most common I);   q, k neurons, proj, mproj: as v12.
Line offsets as v12. Stored: as v12 (per neuron one byte for its step term).
Prediction before running: at A=6 / A=5 the same or more words than v12 (29, 22) at a similar size; if the plates'
neuron sizes and next paths already match (neuron j big <-> next line j small), no change."""
from plates_lib import *
import sys
extra = sum(plate(n).shape[0] for n in NAMES)
As = [int(x) for x in sys.argv[1:]] or [6, 5]
def next_path(name):                                       # each line's size in true scale, in half places
    q = np.abs(plate(name).astype(np.int64)); e = colexp(name)
    return halfplace((q << (e - e.min())[None, :]).sum(1))
NP = {}
for L in range(12):
    NP[f"fc.{L}"] = next_path(f"mproj.{L}")              # 3072 lines of mproj <- 3072 neurons of fc
    NP[f"qkv.{L}"] = next_path(f"proj.{L}")              # 768 lines of proj <- 768 v neurons of qkv
print("fc.0 next-path half places:", np.unique(NP["fc.0"], return_counts=True))
def places(n, q, A):
    qa = np.abs(q.astype(np.int64)); R, C = q.shape
    tph = halfplace(qa.sum(0)) + 2 * (2 - bl(np.array(R)))
    trh = halfplace(qa.sum(1)); off = trh - mode(trh)
    if n in NP:
        I = NP[n]; cols = np.arange(C) if n.startswith("fc") else np.arange(1536, 2304)
        own = tph[cols]; tph = tph.copy(); tph[cols] = mode(own) - (I - mode(I))
    H = tph[None, :] + off[:, None] - A
    return np.maximum(0, H >> 1)
for A in As:
    def dev(n, q, e, A=A):
        s = places(n, q, A); return cut(q, s), s
    run_version(f"v14 next path for fc + v neurons, A={A}", dev, side_per_col=2, extra_side=extra,
                note=f"v12 meters, but fc and v neurons exposed inversely to the next plate's line size; A={A}")
