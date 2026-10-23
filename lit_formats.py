"""Fewest lit pixels for the same picture: exact re-writings of the plate and the light, counted in lit-pixel meetings
(lit x lit, the dark free), every bank, the 30 pieces' real light. Single waves = up/down pixels with no two lit side by
side (each value's fewest lit pixels; count = ones(n ^ 3n)). Base 2.      python3 lit_formats.py
"""
import numpy as np
exec(open("rank_gaps.py").read().split("tot = {}")[0])          # base2 machine, pc(), lights per bank
def naf(A):                                                     # lit pixels in the single-wave writing of each value
    A = np.abs(A.astype(np.int64)); return pc(A ^ (3 * A))
tot = {}
for name, xs in lights.items():
    t = tot.setdefault(name.split(".")[0], np.zeros(6, np.int64))
    w = M[name + ".w"].astype(np.int64); order = np.argsort(w, axis=0, kind="stable"); ws = np.take_along_axis(w, order, 0)
    gaps = np.diff(ws, axis=0, prepend=0); pw, nw, pg, ng = pc(w), naf(w), pc(gaps), naf(gaps)
    for x in xs[:6]:
        px, nx = pc(x)[:, None], naf(x)[:, None]
        S = np.cumsum(x.astype(np.int64)[order][::-1], axis=0)[::-1]
        t += [(pw * px).sum(), (nw * px).sum(), (nw * nx).sum(), (pg * pc(S)).sum(), (ng * naf(S)).sum(), w.size]
print(f"  {'bank':6s} | lit-pixel meetings, share of plain:  single waves (plate)   single waves (both)   ranked gaps   ranked gaps + single waves")
for k, (a, b, c, d, e, n) in tot.items():
    print(f"  {k:6s} | {'':36s}{100 * b / a:5.0f}%{'':17s}{100 * c / a:5.0f}%{'':15s}{100 * d / a:5.0f}%{'':9s}{100 * e / a:5.0f}%")
