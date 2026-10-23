"""PLATES v15 -- every neuron exposed for its next path (v14 extended to all plates). Where each neuron's light goes:
   fc.L neuron j          -> line j of mproj.L
   qkv.L v-neuron d       -> line d of proj.L
   qkv.L q-neuron d       -> multiplies k-neuron d of the same head (the score): its path is k-neuron d's size
   qkv.L k-neuron d       -> multiplies q-neuron d: its path is q-neuron d's size
   proj.L neuron d        -> stream line d, read next through the gauge by line d of fc.L
   mproj.L neuron d       -> stream line d, read next by line d of qkv.L+1 (for L = 11: line d of the top table)
Sizes in true scale (|weight| shifted by the neuron's power of two above the plate's smallest), summed, read in half
places. Neuron term = (most common own half place of that part) - (path - most common path). Line offsets as v12.
Prediction before running: A=5 >= v14's 26; A=4 above v12's 23 at about the same size."""
from plates_lib import *
import sys
extra = sum(plate(n).shape[0] for n in NAMES)
As = [int(x) for x in sys.argv[1:]] or [5, 4]
def line_size(w, e):                                       # each line's size, true scale, half places
    q = np.abs(w.astype(np.int64)); return halfplace((q << (e - e.min())[None, :]).sum(1))
def col_size(w, e):                                        # each neuron's size, true scale, half places
    q = np.abs(w.astype(np.int64)); return halfplace(q.sum(0) << (e - e.min()))
PATH = {}
for L in range(12):
    PATH[f"fc.{L}"] = [(np.arange(3072), line_size(plate(f"mproj.{L}"), colexp(f"mproj.{L}")))]
    qk = plate(f"qkv.{L}"); eq = colexp(f"qkv.{L}"); cs = col_size(qk, eq)
    PATH[f"qkv.{L}"] = [(np.arange(0, 768), cs[768:1536]), (np.arange(768, 1536), cs[0:768]),
                        (np.arange(1536, 2304), line_size(plate(f"proj.{L}"), colexp(f"proj.{L}")))]
    PATH[f"proj.{L}"] = [(np.arange(768), line_size(plate(f"fc.{L}"), colexp(f"fc.{L}")))]
    nxt = (plate(f"qkv.{L+1}"), colexp(f"qkv.{L+1}")) if L < 11 else (M["head.w"], M["head.e"].astype(np.int64))
    PATH[f"mproj.{L}"] = [(np.arange(768), line_size(*nxt))]
def places(n, q, A):
    qa = np.abs(q.astype(np.int64)); R, C = q.shape
    tph = halfplace(qa.sum(0)) + 2 * (2 - bl(np.array(R)))
    trh = halfplace(qa.sum(1)); off = trh - mode(trh)
    t = tph.copy()
    for cols, I in PATH[n]:
        t[cols] = mode(tph[cols]) - (I - mode(I))
    H = t[None, :] + off[:, None] - A
    return np.maximum(0, H >> 1)
for A in As:
    def dev(n, q, e, A=A):
        s = places(n, q, A); return cut(q, s), s
    run_version(f"v15 next path for every neuron, A={A}", dev, side_per_col=2, extra_side=extra,
                note=f"every neuron exposed inversely to the size of the path its light takes next; line offsets as v12; A={A}")
